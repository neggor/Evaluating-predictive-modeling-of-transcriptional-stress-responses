import sys
import numpy as np
sys.path.append(".")
import os
import torch
import pandas as pd
from tqdm import tqdm
from Code.Training.train_cnn import handle_batch
from Code.Training.get_performance import load_data
from Code.CNN_model.res_CNN import myCNN
import shap
import argparse
import json


def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    From: https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """

    seq_len, one_hot_dim = seq.shape
    arr = one_hot_to_tokens(seq)

    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    all_results = np.empty(
        (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
    )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)

    # enfornce that there are no 0s! That seriusly messes up things
    shape_missing = all_results[all_results.sum(-1) == 0, :].shape

    # Generate random one-hot encoded values
    random_indices = np.random.randint(
        0, 4, size=shape_missing[0]
    )  # Random indices for one-hot encoding
    one_hot_filled = np.zeros(shape_missing, dtype=int)  # Initialize zero matrix
    one_hot_filled[np.arange(shape_missing[0]), random_indices] = 1  # Fill with 1s

    all_results[all_results.sum(-1) == 0, :] = one_hot_filled
    assert np.sum(all_results.sum(-1) == 0) == 0
    return all_results if num_shufs else all_results[0]


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    From: https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def one_hot_to_tokens(one_hot):
    """
    From: https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py


    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def do_shap_analysis(model, query, background, device, split=False, hypothetical=True):
    """
    Gets actual and hypothetical attribution scores given a model, query and background

    1 x 4 x L tensor query
    N x 4 x L tensor background
    """
    # convert to torch tensors and put in device
    if type(query) == np.ndarray:
        query = torch.tensor(query).to(device)
    if type(background) == np.ndarray:
        background = torch.tensor(background).to(device)

    # if split, split the background in 4 parts and do the shap analysis for each part
    # and then average the results
    shap_values = []
    if split:
        background = torch.split(background, 30, dim=0)
        # hypothetical_scores = []
        for bg in background:
            my_shap_explainer = shap.DeepExplainer(model, torch.tensor(bg).to(device))
            _shap = my_shap_explainer.shap_values(
                query, check_additivity=False, hypothetical=hypothetical
            )
            shap_values.append(_shap)
            # hypothetical_scores.append(hyp_)
        # shap_values = np.mean(shap_values, axis=0)
        # hypothetical_scores = np.mean(hypothetical_scores, axis=0)

    else:
        my_shap_explainer = shap.DeepExplainer(
            model, torch.tensor(background).to(device)
        )
        _shap = my_shap_explainer.shap_values(
            query, check_additivity=False, hypothetical=hypothetical
        )
        shap_values.append(_shap)
        # hypothetical_scores = hyp_

    return shap_values  # , hypothetical_scores


def get_shap(
    TSS_sequences,
    TTS_sequences,
    mRNA_df,
    outcome_type,
    treatment_names,
    model,
    device,
    n_background=10,
    main_folder="Results/Interpretation",
):
    """
    Get shap values for all the data in the loader
    Make sure that the loadr has batch size 1, because otherwise the meomry will not be enough
    """

    for gene in tqdm(mRNA_df["Gene"]):
        TSS = TSS_sequences[gene][np.newaxis]
        TTS = TTS_sequences[gene][np.newaxis]
        b = {
            "TSS": TSS,
            "TTS": TTS,
            "values": mRNA_df[mRNA_df["Gene"] == gene].iloc[:, 1:-1].values,
        }
        DNA, _ = handle_batch(b, device)
        background = dinuc_shuffle(DNA.squeeze().T.cpu().detach().numpy(), n_background)
        background = np.transpose(background, (0, 2, 1))
        # make sure that the background has all 0s in the positions where the query has all 0s (padding stuff)
        background = background * DNA.cpu().detach().numpy().sum(1)[:, np.newaxis, :]
        shap_ = do_shap_analysis(model, DNA, background, device, split=False)[
            0
        ]  # if split is flase it is a list of 1 element (TODO)
        # Deep explainer returns a list anyway, for one element per head
        # save output
        for i, treatment in enumerate(treatment_names):
            os.makedirs(
                f"{main_folder}/{outcome_type}/{treatment}/hypothetical_scores",
                exist_ok=True,
            )
            os.makedirs(
                f"{main_folder}/{outcome_type}/{treatment}/queries", exist_ok=True
            )
            np.save(
                f"{main_folder}/{outcome_type}/{treatment}/hypothetical_scores/{gene}_hyp_scores.npy",
                shap_[i],
            )
            np.save(
                f"{main_folder}/{outcome_type}/{treatment}/queries/{gene}_query.npy",
                DNA.cpu().detach().numpy(),
            )

    return shap_


def main():
    parser = argparse.ArgumentParser()
    DNA_specs = [814, 200, 200, 814]
    outcome_types = [
        "log2FC",
        "amplitude",
        "quantiles_per_treatment",
        "DE_per_treatment",
    ]
    treatments = ["B", "C", "D", "G", "H", "X", "Y", "Z", "W", "V", "U", "T"]
    mapping = {
        "B": "MeJA",
        "C": "SA",
        "D": "SA+MeJA",
        "G": "ABA",
        "H": "ABA+MeJA",
        "X": "3-OH10",
        "Y": "chitooct",
        "Z": "elf18",
        "W": "flg22",
        "V": "nlp20",
        "U": "OGs",
        "T": "Pep1",
    }

    for outcome_type in outcome_types:
        config_file = f"Results/CNN/{outcome_type}/2048/exons_masked_False/config.json"

        with open(config_file, "r") as f:
            cnn_config = json.load(f)

        dna_specs = [
            cnn_config["upstream_TSS"],
            cnn_config["downstream_TSS"],
            cnn_config["upstream_TTS"],
            cnn_config["downstream_TTS"],
        ]
        print(f"DNA specs: {dna_specs}")
        _, mRNA_validation, mRNA_test, TSS_sequences, TTS_sequences, metadata = (
            load_data(
                0.8,  # Important, cuz we only use validation and test.
                0.1,  # Important, cuz we only use validation and test.
                treatments=cnn_config["treatments"],
                problem_type=cnn_config["problem_type"],
                dna_format=cnn_config["dna_format"],
                DNA_specs=dna_specs,
                mask_exons=cnn_config["mask_exons"],
                kmer_rc=False,  # it does not apply to the CNN
            )
        )
        model = myCNN(
            n_labels=cnn_config["n_labels"],
            n_ressidual_blocks=cnn_config["n_ressidual_blocks"],
            in_channels=cnn_config["in_channels"],
            out_channels=cnn_config["out_channels"],
            kernel_size=cnn_config["kernel_size"],
            max_pooling_kernel_size=cnn_config["max_pooling_kernel_size"],
            dropout_rate=cnn_config["dropout_rate"],
            stride=cnn_config["stride"],
            RC_equivariant=cnn_config["equivariant"],
        )

        model.load_state_dict(
            torch.load(
                f"Results/CNN/{outcome_type}/2048/exons_masked_False/model_0/best_model.pth"
            )
        )
        model.eval()
        model.to("cuda")

        get_shap(
            TSS_sequences,
            TTS_sequences,
            pd.concat([mRNA_validation, mRNA_test]),
            outcome_type,
            [mapping[t] for t in treatments],
            model,
            "cuda",
            n_background=100,
        )


if __name__ == "__main__":
    main()
