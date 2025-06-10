import numpy as np
import torch.nn as nn
import torch


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def rotate_weights(kernel):
    """
    Generating the RC complementary kernel
    """

    r_complementary_kernel = torch.flip(kernel, dims=[1, 2])

    return r_complementary_kernel


class ConvolutionalBlock(nn.Module):
    """
    Implements a convolutional block with
    - ReLU activation
    - Conv1D
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super(ConvolutionalBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            dilation=dilation,
        )

    def forward(self, x):

        x = self.relu(x)
        x = self.conv1d(x)
        return x


class RCe_ConvolutionalBlock(nn.Module):
    """
    Implements a convolutional block with
    - Batch Normalization 1D
    - ReLU activation
    - Conv1D
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super(RCe_ConvolutionalBlock, self).__init__()
        self.relu = nn.ReLU()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation

        self.my_weights = nn.Parameter(
            torch.empty(out_channels // 2, in_channels, kernel_size)
        )
        nn.init.kaiming_uniform_(self.my_weights, a=5**0.5)

        self.bias = nn.Parameter(torch.zeros(out_channels // 2))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = self.relu(x)
        kernels = torch.cat(
            [self.my_weights, rotate_weights(self.my_weights).flip(0)], dim=0
        )
        bias = torch.cat([self.bias, self.bias.flip(0)], dim=0)
        x = nn.functional.conv1d(
            x,
            kernels,
            stride=self.stride,
            padding=self.kernel_size // 2,
            dilation=self.dilation,
            bias=bias,
        )
        return x


class myCNN(nn.Module):
    def __init__(
        self,
        n_labels,
        n_ressidual_blocks,
        in_channels,
        out_channels,
        kernel_size,
        max_pooling_kernel_size,
        dropout_rate,
        stride=1,
        RC_equivariant=False,
    ):
        super(myCNN, self).__init__()

        if type(kernel_size) == int:
            kernel_size = [kernel_size] * n_ressidual_blocks
        elif type(kernel_size) == list:
            assert len(kernel_size) == n_ressidual_blocks
        if RC_equivariant:
            assert is_power_of_2(max_pooling_kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_labels = n_labels
        self.residual_blocks = nn.ModuleList()
        self.RC_equivariant = RC_equivariant

        for i in range(n_ressidual_blocks):
            self.residual_blocks.append(
                nn.Sequential(
                    (
                        RCe_ConvolutionalBlock(
                            in_channels, out_channels, kernel_size[i], 1, 1
                        )
                        if RC_equivariant
                        else ConvolutionalBlock(
                            in_channels, out_channels, kernel_size[i], 1, 1
                        )
                    ),
                    Residual(
                        RCe_ConvolutionalBlock(
                            out_channels, out_channels, kernel_size[i], 1, 1
                        )
                        if RC_equivariant
                        else ConvolutionalBlock(
                            out_channels, out_channels, kernel_size[i], 1, 1
                        )
                    ),
                    (
                        nn.MaxPool1d(max_pooling_kernel_size)
                        if i < n_ressidual_blocks - 1
                        else nn.Identity()
                    ),
                    (
                        nn.Dropout(dropout_rate)
                        if i < n_ressidual_blocks - 1
                        else nn.Identity()
                    ),
                )
            )
            in_channels = out_channels

        self.residual_blocks = nn.Sequential(*self.residual_blocks)
        self.inp_length = out_channels // 2 if RC_equivariant else out_channels
        print(self.inp_length)
        self.ffn = nn.Linear(self.inp_length, self.n_labels)
        self.ffn_size = None
        self.global_max_pooling_set = lambda x: nn.MaxPool1d(x.size(2))
        self.global_max_pooling = None

    def forward(self, z):
        assert is_power_of_2(z.size(2))
        z = self.residual_blocks(z)
        # To take the max activation over the forward and reverse complement, hence making it invariant to the orientation of the sequence.
        z = (
            torch.cat([z[:, : z.size(1) // 2], z[:, z.size(1) // 2 :].flip(1)], dim=2)
            if self.RC_equivariant
            else z
        )
        if self.global_max_pooling is None:
            print("Setting the global max pooling")
            print(z.size(2))
            self.global_max_pooling = self.global_max_pooling_set(z)

        z = self.global_max_pooling(z).squeeze(-1)

        base_logits = self.ffn(z)

        return base_logits


if __name__ == "__main__":
    ### TESTING STUFF
    def test_RC_equivariant_Conv1d():
        # generate a one-hot encoded DNA sequence
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        reverse_mapping = {0: "A", 1: "C", 2: "G", 3: "T"}
        kernel = torch.zeros(2, 4, 16)
        original_sequence = "GGGGaaaaaaaattta"
        for i, nuc in enumerate(original_sequence):
            kernel[0, mapping[nuc.upper()], i] = 1
        r_kernel = rotate_weights(kernel)
        new_sequence = []
        for i in range(16):
            new_sequence.append(reverse_mapping[r_kernel[0, :, i].argmax().item()])
            # print(reverse_mapping[r_kernel[0, :, i].argmax().item()], end="")
        assert (
            "".join(new_sequence) == "taaattttttttCCCC".upper()
        )  # check that indeed this is the RC

        kernel[1] = rotate_weights(kernel[0:1])

        return kernel

    x = test_RC_equivariant_Conv1d()
    my_RCNN = myCNN(1, 3, 4, 4, [5, 5, 5], 2, 0, RC_equivariant=True)
    my_RCNN.eval()
    # import pdb; pdb.set_trace()
    # Now check that for every layer we have equivariance in th eintermediate output
    for i, layer in enumerate(my_RCNN.residual_blocks):
        x = layer(x)
        assert torch.allclose(
            x[0], rotate_weights(x[1:2])[0], atol=1e-5
        )  # EQUIVARIANCE

    x = test_RC_equivariant_Conv1d()
    x = my_RCNN(x)
    assert torch.allclose(x[0], x[1], atol=1e-5)  # INVARIANCE

    # Now the same, but more intensive testing
    # generate a random matrix 4 x 4096
    x = torch.randn(2, 4, 4096)
    x[1] = rotate_weights(x[0:1])
    my_RCNN = myCNN(1, 3, 4, 2, [5, 3, 7], 8, 0, RC_equivariant=True)
    my_RCNN.eval()
    # Now check that for every layer we have equivariance in th eintermediate output
    for i, layer in enumerate(my_RCNN.residual_blocks):
        x = layer(x)
        assert torch.allclose(
            x[0], rotate_weights(x[1:2])[0], atol=1e-5
        )  # EQUIVARIANCE

    x = torch.randn(2, 4, 4096)
    x[1] = rotate_weights(x[0:1])
    x = my_RCNN(x)
    assert torch.allclose(x[0], x[1], atol=1e-5)  # INVARIANCE
    print(x[0], x[1])
    print("All tests passed")

    # NOTE THIS WORKS EVEN WITH NON-ZERO BIASES!!!

    # initialize tw networks one rc and another not rc and count the number of parameters
    my_RCNN = myCNN(1, 4, 4, 10, [5, 3, 7, 5], 8, 0, RC_equivariant=True)
    my_RCNN.eval()
    my_NN = myCNN(1, 4, 4, 10, [5, 3, 7, 5], 8, 0, RC_equivariant=False)
    my_NN.eval()
    print(
        "Number of parameters in RC network",
        sum(p.numel() for p in my_RCNN.parameters()) * 2,
    )
    print(
        "Number of parameters in non-RC network",
        sum(p.numel() for p in my_NN.parameters()),
    )
