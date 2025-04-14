import sys
sys.path.append(".")
from Code.CNN_model.res_CNN import myCNN
from Code.dataset_utils.CustomDataset import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score, )
from torch.optim.lr_scheduler import LambdaLR

def mse_loss(output, target, weights=None):
    #return F.mse_loss(output, target, reduction="mean")
    return F.l1_loss(output, target, reduction="mean", weight=weights)

def binary_cross_entropy(output, target, weights=None):
    # this is the simplest one
    mask = torch.ones(target.shape[0], output.shape[1]).to(target.device)
    mask[target == 3] = 0
    loss = nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")(output, target)
    return (loss * mask).sum() / mask.sum()

def loss_fnc(loss_name):
    # Intended usage is
    # loss = loss_fnc("mse")(output, target)
    if loss_name == "log2FC" or loss_name == "amplitude":
        return mse_loss
    elif loss_name == "DE_per_treatment" or loss_name == "quantiles_per_treatment":
        return binary_cross_entropy
    else:
        raise ValueError("Loss function not recognized")
    
class metrics:
    """
    Accumulate the training metrics in a sensible way.
    Provide functionality to prent summary statistics.
    """

    def __init__(self, problem_type):
        assert problem_type in [
            "log2FC",
            "DE_per_treatment",
            "LR",
            "quantiles_per_treatment",
            "amplitude"
        ]
        self.metrics = {}
        self.problem_type = problem_type
        self.metrics["loss"] = {}
        self.metrics["train_loss"] = {}
        self.metrics["accuracy"] = {}
        self.metrics["MCC"] = {}
        self.metrics["AUC"] = {}
        self.metrics["F1"] = {}
        self.metrics["pearson_correlation"] = {}
        self.metrics["R2"] = {}
        self.metrics["1_vs_0_acc"] = {}
        self.metrics["MCC_per_class"] = {}
        self.outputs = {}
        self.targets = {}
        self.loss = {}

    def add_batch(self, epoch, output, target, loss):
        # if the problem is GSR, then the target is a matrix of batch x 1 probabilities
        # the ouptut is the same

        # if the problem is log2FC, then the target is a matrix of batch x n_classes real values
        # the output is the same

        # if the problem is direction, then the target is list of n_heads tensors of batch x n_classes
        # the output is a matrix batch x n_heads, with values from 0 to n_classes

        # convert to numpy
        if (
            self.problem_type == "log2FC" or self.problem_type == "amplitude"
            or self.problem_type == "DE_per_treatment"
            or self.problem_type == "quantiles_per_treatment"
        ):
            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            if (
                self.problem_type == "DE_per_treatment"
                or self.problem_type == "quantiles_per_treatment"
            ):
                # apply sigmoid
                # mask = (target != 3).all(axis=1)
                output = 1 / (1 + np.exp(-output))  # [mask]
                target = target  # [mask]

        if epoch not in self.outputs:
            self.outputs[epoch] = []
            self.targets[epoch] = []
            self.loss[epoch] = []

        self.outputs[epoch].append(output)
        self.targets[epoch].append(target)
        self.loss[epoch].append(loss)

    def _R2(self, output, target):
        # R2 = 1 - SS_res/SS_tot
        SS_res = np.sum((output - target) ** 2)
        SS_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - SS_res / SS_tot

    def concatenate_epoch_logs(self, epoch, train_loss=None):
        # concatenate the outputs and targets for the epoch
        loss = np.mean(self.loss[epoch])
        if (
            self.problem_type == "log2FC" or self.problem_type == "amplitude"
            or self.problem_type == "DE_per_treatment"
            or self.problem_type == "quantiles_per_treatment"
        ):
            output = np.concatenate(self.outputs[epoch], axis=0)
            target = np.concatenate(self.targets[epoch], axis=0)

        # remove the key from the dictionary
        del self.outputs[epoch]
        del self.targets[epoch]

        # calculate the metrics
        if (
            self.problem_type == "DE_per_treatment"
            or self.problem_type == "quantiles_per_treatment"
        ):
            
            self.metrics["MCC_per_class"][epoch] = {}
            for i in range(target.shape[1]):
                mask = target[:, i] != 3
                self.metrics["MCC_per_class"][epoch][i] = matthews_corrcoef(
                    target[mask, i], output[mask, i] > 0.5
                )

            # calculate average correlation between the probabilities of the output
            corrs = np.triu(np.corrcoef(output.T), k=1)
            corrs = corrs[corrs != 0]
            print(f"Correlation between outcome columns: {corrs.mean()}")

            # I need to put in long format
            target = target.ravel()
            mask = target != 3
            target = target[mask]
            output = output.ravel()
            output = output[mask]

            self.metrics["loss"][epoch] = loss
            self.metrics["train_loss"][epoch] = train_loss
            self.metrics["accuracy"][epoch] = accuracy_score(target, output > 0.5)
            self.metrics["MCC"][epoch] = matthews_corrcoef(target, output > 0.5)
            self.metrics["AUC"][epoch] = roc_auc_score(target, output)
            self.metrics["F1"][epoch] = f1_score(target, output > 0.5)

        elif self.problem_type == "log2FC" or self.problem_type == "amplitude":
            # this requires evaluation per class/column
            self.metrics["loss"][epoch] = loss
            self.metrics["train_loss"][epoch] = train_loss
            self.metrics["pearson_correlation"][epoch] = {}
            self.metrics["R2"][epoch] = {}
            # print correlation of output columns
            corrs = np.triu(np.corrcoef(output.T), k=1)
            corrs = corrs[corrs != 0]
            print(f"Correlation between columns: {corrs.mean()}")
            # now the same but for target
            corrs = np.triu(np.corrcoef(target.T), k=1)
            corrs = corrs[corrs != 0]
            print(f"Correlation between target columns: {corrs.mean()}")
            for i in range(target.shape[1]):
                self.metrics["pearson_correlation"][epoch][i] = np.corrcoef(
                    output[:, i], target[:, i]
                )[0, 1]
                self.metrics["R2"][epoch][i] = self._R2(output[:, i], target[:, i])
        # print
        print(f"Epoch {epoch} - Loss: {loss}")
        if (
            self.problem_type == "DE_per_treatment"
            or self.problem_type == "quantiles_per_treatment"
        ):
            print(f"Accuracy: {self.metrics['accuracy'][epoch]}")
            print(f"MCC: {self.metrics['MCC'][epoch]}")
            print(f"AUC: {self.metrics['AUC'][epoch]}")
            print(f"F1: {self.metrics['F1'][epoch]}")

        elif self.problem_type == "log2FC" or self.problem_type == "amplitude":
            for i in range(target.shape[1]):
                print(
                    f"Head {i} - Pearson correlation: {self.metrics['pearson_correlation'][epoch][i]}"
                )

    def plot_training_logs(self, folder, name):
        # plot the training logs

        if (
            self.problem_type == "DE_per_treatment"
            or self.problem_type == "quantiles_per_treatment"
        ):
            # make a grid plot of the statistics
            # loss, accuracy, MCC, AUC, F1
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            for i, metric in enumerate(["loss", "accuracy", "MCC", "AUC", "F1"]):
                ax = axs[i // 3, i % 3]
                if metric == "loss":
                    ax.plot(
                        list(self.metrics[metric].keys()),
                        list(self.metrics[metric].values()),
                        label="Validation",
                    )
                    ax.plot(
                        list(self.metrics["train_loss"].keys()),
                        list(self.metrics["train_loss"].values()),
                        label="Train",
                    )
                    ax.legend()
                else:
                    ax.plot(
                        list(self.metrics[metric].keys()),
                        list(self.metrics[metric].values()),
                    )
                ax.set_title(metric)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
            # the last one for the MCC per class in the case of DE_per_treatment
        
            ax = axs[1, 2]
            for i in range(len(self.metrics["MCC_per_class"][0])):
                ax.plot(
                    list(self.metrics["MCC_per_class"].keys()),
                    [x[i] for x in list(self.metrics["MCC_per_class"].values())],
                    label=f"Class {i}",
                )
            ax.set_title("MCC_per_class")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MCC_per_class")
            ax.legend()

            plt.savefig(f"{folder}/{name}_training_metrics.png")

        elif self.problem_type == "log2FC" or self.problem_type == "amplitude":
            # make a grid plot of the statistics
            # loss, pearson_correlation, R2
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            ax = axs[0]
            ax.plot(
                list(self.metrics["loss"].keys()),
                list(self.metrics["loss"].values()),
                label="Validation",
            )
            ax.plot(
                list(self.metrics["train_loss"].keys()),
                list(self.metrics["train_loss"].values()),
                label="Train",
            )
            ax.legend()
            ax.set_title("loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("loss")
            for i, metric in enumerate(["pearson_correlation", "R2"]):
                ax = axs[i + 1]
                for j in range(len(self.metrics[metric][0])):
                    ax.plot(
                        list(self.metrics[metric].keys()),
                        [x[j] for x in list(self.metrics[metric].values())],
                        label=f"Class {j}",
                    )
                ax.set_title(metric)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.legend()
           
            plt.savefig(f"{folder}/{name}_training_metrics.png")
        plt.close("all")

def handle_batch(batch, device):
    TTS = batch["TTS"]
    TSS = batch["TSS"]
    padding_20bp = np.zeros((TSS.shape[0], 20, 4))
    DNA = np.concatenate((TSS, padding_20bp, TTS), axis=1)
    DNA = torch.tensor(DNA).float().to(device).transpose(1, 2)
    labels = torch.tensor(batch["values"]).float().to(device)
    return DNA, labels

def get_cnn(n_labels, equivariant):
    '''
    Wrapping to handle either enformer or a CNN
    '''
    model = myCNN(
        n_labels=n_labels,
        n_ressidual_blocks=5,
        in_channels=4,
        out_channels=300,
        kernel_size=[5, 3, 3, 3, 3],
        max_pooling_kernel_size=4,
        dropout_rate=0.25,
        stride=1,
        RC_equivariant=equivariant,
    )

    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    return model

def get_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
        """
        Returns a LambdaLR scheduler for warm-up followed by a linear decay.
        
        Args:
        - optimizer: The optimizer to apply the scheduler to.
        - num_warmup_steps: Number of warm-up steps.
        - num_training_steps: Total number of training steps.
        
        Returns:
        - A LambdaLR scheduler.
        """
        print("Using warmup scheduler")
        print(f"Num warmup steps: {num_warmup_steps}")
        print(f"Num training steps: {num_training_steps}")
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warm-up
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                # Linear decay until 10% of the learning rate
                return max(
                    0.1,
                    float(num_training_steps - current_step)
                    / float(max(1, num_training_steps - num_warmup_steps)),
                )
                

        return LambdaLR(optimizer, lr_lambda)

def train_cnn_model(
    model,
    train_loader,
    val_loader,
    training_specs,
    batch_handler,
    device,
    store_folder,
    name,
    weights=None,
):
    """
    Trains the model up to early stopping based on validation.
    """
    os.makedirs(store_folder, exist_ok=True)
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_specs["lr"],
        weight_decay=training_specs["weight_decay"],
    )
    scheduler = get_warmup_scheduler(optimizer, 100, training_specs["n_epochs"] * len(train_loader))
    my_metrics = metrics(training_specs["problem_type"])
    my_metrics_val = metrics(training_specs["problem_type"])
    my_loss_fnc = loss_fnc(training_specs["problem_type"])

    # Training loop
    best_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(training_specs["n_epochs"]):
        model.train()
        train_loss = []
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            DNA, labels = batch_handler(batch, device)
            output = model(DNA)            
            #l1_norm_last_layer = model.ffn.weight.norm(p=1) # TMP will only work for GSR and CNN
            loss = my_loss_fnc(output, labels, weights) #+ 0.001 * l1_norm_last_layer
            loss.backward()
            optimizer.step()
            scheduler.step()
            my_metrics.add_batch(epoch, output, labels, loss.item())
            train_loss.append(loss.item())
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("TRAIN:")
        my_metrics.concatenate_epoch_logs(epoch)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

       
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                DNA, labels = batch_handler(batch, device)
                output = model(DNA)
                loss = my_loss_fnc(output, labels, weights)
                my_metrics_val.add_batch(epoch, output, labels, loss.item())
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("VAL:")
        my_metrics_val.concatenate_epoch_logs(epoch, np.mean(train_loss))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        my_metrics_val.plot_training_logs(store_folder, name)
        # early stopping
        if my_metrics_val.metrics["loss"][epoch] < best_loss:
            best_loss = my_metrics_val.metrics["loss"][epoch]
            best_model = model.state_dict()
            print(f"New best model with loss: {best_loss}")
            # save
            torch.save(best_model, os.path.join(store_folder, "best_model.pt"))
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= training_specs["patience"]:
                print("Early stopping!")
                break
        
    return model, my_metrics, my_metrics_val

def get_loader(mRNA, TSS_sequences, TTS_sequences, batch_size, shuffle):
    loader = get_dataloader(
        mRNA,
        TSS_sequences,
        TTS_sequences,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader

# Now the idea is to construct a function that will take the training specs and the model specs
# and return the model, the training metrics and the validation metrics

def train_cnn_model_from_specs(
    model,
    training_specs,
    TSS_sequences,
    TTS_sequences,
    mRNA_train,
    mRNA_validation,
    device,
    store_folder,
    name,
):  
    os.makedirs(store_folder, exist_ok=True)
    training_loader = get_loader(
        mRNA_train,
        TSS_sequences,
        TTS_sequences,
        training_specs["batch_size"],
        True
    )
    validation_loader = get_loader(
        mRNA_validation,
        TSS_sequences,
        TTS_sequences,
        training_specs["batch_size"],
        False,
    )
    weights = None
    if training_specs["problem_type"] == "DE_per_treatment" or training_specs["problem_type"] == "quantiles_per_treatment":
        # get the proportions
        weights = []
        for i in mRNA_train.columns[1:-1]:
            mask = mRNA_train[i] != 3
            ps_clas = np.sum(mRNA_train[i][mask].values)
            print(f"Class {i} - Positive instances: {ps_clas}")
            ns_clas = mRNA_train[mask].shape[0] - ps_clas
            print(f"Class {i} - Negative instances: {ns_clas}")
            weight_i = ns_clas / ps_clas
            weights.append(weight_i)
        # concat in a vector
        weights = torch.tensor(weights).float().to(device)
        print("Weights:")
        print(weights)
    
    # train the model
    model, my_metrics, my_metrics_val = train_cnn_model(
        model,
        training_loader,
        validation_loader,
        training_specs,
        handle_batch,
        device,
        store_folder,
        name,
        weights,
    )

    return model, my_metrics, my_metrics_val


def test_cnn(   model,
                training_specs,
                TSS_sequences,
                TTS_sequences,
                mRNA_test,
                device,
                store_folder,
                treatments,
                save_results = True,
                return_model = False,
                return_metrics = False):
    
    model.eval()
    model = model.to(device)

    # Construct loader
    test_loader = get_loader(
        mRNA_test,
        TSS_sequences,
        TTS_sequences,
        training_specs["batch_size"],
        False,
    )

    my_metrics = metrics(training_specs["problem_type"])
    outputs = []
    inputs = []
    my_loss_fn = loss_fnc(training_specs["problem_type"])
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            DNA, labels = handle_batch(batch, device)
            output = model(DNA)
            loss = my_loss_fn(output, labels, None) # The loss is not important anyway!
            my_metrics.add_batch(0, output, labels, loss.item())
            outputs.append(output.cpu().detach().numpy())
            inputs.append(labels.cpu().detach().numpy())
    
    my_metrics.concatenate_epoch_logs(0)
    Y_hat = np.concatenate(outputs, axis=0)
    Y = np.concatenate(inputs, axis=0)
    
    if training_specs['problem_type'] in ["log2FC", "amplitude"]:
        # get R2 and pearson correlation
        m = {}
        m["R2"] = []
        m["Pearson"] = []
        m['sign_accuracy'] = []
        m['Spearman'] = []
        for i in range(Y.shape[1]):
            m["R2"].append(my_metrics._R2(Y_hat[:, i], Y[:, i]))
            m["Pearson"].append(np.corrcoef(Y_hat[:, i], Y[:, i])[0, 1])
            m['sign_accuracy'].append(np.mean(np.sign(Y_hat[:, i]) == np.sign(Y[:, i])))
            m['Spearman'].append(stats.spearmanr(Y_hat[:, i], Y[:, i]).statistic)

        print(m)
    elif training_specs['problem_type'] in ["DE_per_treatment", "quantiles_per_treatment"]:
        # get MCC per class
        m = {}
        m["MCC"] = []
        m["Accuracy"] = []
        m["AUC"] = []
        m["F1"] = []
        # apply sigmoid
        Y_hat = 1 / (1 + np.exp(-Y_hat))
        for i in range(Y.shape[1]):
            mask = Y[:, i] != 3
            m["MCC"].append(matthews_corrcoef(Y[mask, i], Y_hat[mask, i] > 0.5))
            m["Accuracy"].append(accuracy_score(Y[mask, i], Y_hat[mask, i] > 0.5))
            m["AUC"].append(roc_auc_score(Y[mask, i], Y_hat[mask, i]))
            m["F1"].append(f1_score(Y[mask, i], Y_hat[mask, i] > 0.5))
        print(m)
    
    # construct DF with m
    #import pdb; pdb.set_trace()
    m = pd.DataFrame(m).T
    m.columns = treatments
    if save_results:
        m.to_csv(f"{store_folder}/test_metrics.csv")
    else:
        print(m)

    if return_model:
        if return_metrics:
            return Y_hat, Y, model, m
        return Y_hat, Y, model
    else:
        if return_metrics:
            return Y_hat, Y, m
        return Y_hat, Y