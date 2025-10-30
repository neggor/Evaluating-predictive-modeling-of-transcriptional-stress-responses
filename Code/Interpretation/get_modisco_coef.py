'''
Relate each pattern to a known TFBM.
This will allow to compare patterns across different treatments.
Eventually we construct a TFBM x coefficient kind of matrices.
'''
import h5py
from typing import Dict, Optional
import pandas as pd
import re
import numpy as np
from tqdm import tqdm

def parse_meme_file_to_pwm_dict(meme_path: str) -> Dict[str, np.ndarray]:
    """
    Parse a MEME-format file and extract PWMs into a dictionary keyed by TF name.

    Args:
        meme_path (str): Path to the MEME file.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping TF name (uppercase) to PWM numpy array
                               with shape (4, width), rows correspond to A, C, G, T.
    """
    pwm_dict: Dict[str, np.ndarray] = {}
    with open(meme_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF"):
            parts = line.split()
            tf_name = parts[-1].upper()

            i += 1
            # Find the line starting with "letter-probability matrix:"
            while i < len(lines) and not lines[i].strip().startswith(
                "letter-probability matrix:"
            ):
                i += 1
            if i == len(lines):
                break

            matrix_line = lines[i].strip()
            match = re.search(r"w\s*=\s*(\d+)", matrix_line)
            if not match:
                raise ValueError(f"Cannot find width 'w' for motif {tf_name}")
            w = int(match.group(1))

            i += 1
            pwm_rows = []
            for _ in range(w):
                vals = lines[i].strip().split()
                if len(vals) != 4:
                    raise ValueError(
                        f"Expected 4 columns in PWM line, got {len(vals)}: '{lines[i].strip()}'"
                    )
                pwm_rows.append(list(map(float, vals)))
                i += 1

            pwm_array = np.array(pwm_rows).T  # Shape: (4, w)
            pwm_dict[tf_name] = pwm_array
        else:
            i += 1

    return pwm_dict

def load_symbol_to_gene_id_map(filename: str) -> pd.DataFrame:
    """
    Load a tab-separated file containing gene mapping information and drop rows with NaNs.

    Args:
        filename (str): Path to the tab-separated mapping file.

    Returns:
        pd.DataFrame: DataFrame with columns including 'Gene_id' and 'symbol'.
    """
    return pd.read_table(filename).dropna()

def find_gene_id_for_tf(df: pd.DataFrame, tf_name: str) -> Optional[str]:
    """
    Find the Gene_id for a given TF name by searching in the 'symbol' column of the DataFrame.

    The 'symbol' column contains comma-separated lists of TF names.

    Args:
        df (pd.DataFrame): DataFrame with 'Gene_id' and 'symbol' columns.
        tf_name (str): Transcription factor name to search for.

    Returns:
        Optional[str]: The corresponding Gene_id if found, else None.
    """
    tf_name = tf_name.upper()
    mask = (
        df["symbol"]
        .str.upper()
        .str.split(",")
        .apply(lambda syms: tf_name in [s.strip() for s in syms])
    )
    gene_ids = df.loc[mask, "Gene_id"]
    if not gene_ids.empty:
        return gene_ids.iloc[0]
    return None

def generate_PWM_dict(
    tf_id_mapping: str, jaspar: str
) -> Dict[Optional[str], np.ndarray]:
    """
    Generate a dictionary mapping Gene_id to PWM numpy arrays from a JASPAR MEME file.

    Args:
        tf_id_mapping (str): Path to the gene ID mapping file (tab-separated).
        jaspar (str): Path to the JASPAR MEME motif file.

    Returns:
        Dict[Optional[str], np.ndarray]: Dictionary mapping Gene_id (or None if not found)
                                        to PWM arrays.
    """
    symbol_to_gene = load_symbol_to_gene_id_map(tf_id_mapping)
    pwm_dict = parse_meme_file_to_pwm_dict(jaspar)
    return {find_gene_id_for_tf(symbol_to_gene, key): pwm_dict[key] for key in pwm_dict}

def get_patterns_seqlets(modisco_results: str) -> dict:
    '''
    Digests .h5 results file. 
    Returns patterns with a tuple of PWM and n_seqlets.
    '''
    patterns = {}
    with h5py.File(modisco_results, "r") as f:
        for direction in ["neg_patterns", "pos_patterns"]:
            print("Direction:", direction)
            if direction not in f:
                continue
            # for pattern in direction 
            for key in f[direction].keys():
                PWM = np.array(f[direction][key]["sequence"][:])
                CWM = np.array(f[direction][key]["contrib_scores"][:])
                PWM, CWM = filter_based_on_importance(PWM, CWM, 0.3)
                n_seqlets = str(f[direction][key]["seqlets"]["n_seqlets"][0])
                patterns[direction + "_" + key] = (PWM, n_seqlets)
    
    return patterns

def filter_based_on_entropy(PWM, CWM, threshold=0.6):
    PWM_tmp = PWM + 1e-7  # Add a small value to avoid log(0)
    # Calculate the shannon entropy
    shannon_entropy = -np.sum(PWM_tmp * np.log2(PWM_tmp), axis=1)
    # Find the indices of the positions that pass the threshold
    pass_inds = np.where((1 / shannon_entropy) > threshold)[0]
    if pass_inds.shape[0] == 0:
        return None, None
    # Trim the PWM
    trimmed_pwm = PWM[np.min(pass_inds) : np.max(pass_inds) + 1]
    trimmed_cwm = CWM[np.min(pass_inds) : np.max(pass_inds) + 1]

    return trimmed_pwm, trimmed_cwm

def filter_based_on_importance(PWM, CWM, threshold=0.3):
    score = np.sum(np.abs(CWM), axis=1)
    trim_thresh = (
        np.max(score) * threshold
    )  # Cut off anything less than 30% of max score
    pass_inds = np.where(score >= trim_thresh)[
        0
    ]  # find the indices of the positions that pass the threshold
    # Trim the PWM and CWM
    trimmed_pwm = PWM[
        np.min(pass_inds) : np.max(pass_inds) + 1
    ]  # Between the min and max indices such that the score is above the threshold
    trimmed_cwm = CWM[np.min(pass_inds) : np.max(pass_inds) + 1]

    return trimmed_pwm, trimmed_cwm

def get_argmax_similarities(patterns: dict, tfbms_data: str):
    '''
    Compute similarity between pattern PWMs and TF PWMs, reporting best match per pattern.

    Parameters
    ----------
    patterns : dict[str, tuple[np.ndarray, Any]]
        Dictionary where each value is a tuple (PWM, n_seqlets or similar)
    tfbms_data : str
        Path to MEME/JASPAR file containing TF PWMs

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Full similarity matrix (patterns × TFs) and summary DataFrame.
    '''

    pwm_dict = parse_meme_file_to_pwm_dict(tfbms_data)
    matrix_sim = {}

    # Compute convolution similarities
    for pattern_name, (pattern_PWM, _) in tqdm(patterns.items()):
        row_scores = {}
        for tf_name, tf_PWM in pwm_dict.items():
            # pattern_PWM[0].T because pattern_PWM[0] is (L,4) — we want (4,L)
            row_scores[tf_name] = pwm_convolve_pair(pattern_PWM.T, tf_PWM)
        matrix_sim[pattern_name] = row_scores

    # Similarity matrix: rows = patterns, cols = TFs
    df = pd.DataFrame(matrix_sim).T

    # Identify best-matching TF per pattern
    argmax_TF = df.idxmax(axis=1)
    max_val = df.max(axis=1)

    # Collect pattern-level info
    pattern_values = []
    pattern_lengths = []
    best_TF_lengths = []

    for pattern_name, (pattern_PWM, val) in patterns.items():
        sign = -1 if pattern_name.startswith("neg") else 1
        pattern_values.append(sign * int(val))
        pattern_lengths.append(pattern_PWM.shape[0])  # PWM shape (L,4)

        # get best TF and its length
        best_tf_name = argmax_TF[pattern_name]
        best_TF_lengths.append(pwm_dict[best_tf_name].shape[1])

    # Build summary DataFrame
    result = pd.DataFrame({
        "Best_TF": argmax_TF,
        "Max_Score": max_val,
        "n-seqlets": pattern_values,
        "Pattern_Len": pattern_lengths,
        "Best_TF_Len": best_TF_lengths
    })

    aggregated = (
        result.groupby("Best_TF")
        .agg({
            "Max_Score": "mean",       # average activation strength
            "n-seqlets": "sum",        # total number of seqlets
            "Pattern_Len": "mean",     # mean pattern length
            "Best_TF_Len": "first"     # same for all rows per TF
        })
        .reset_index()
        .sort_values("n-seqlets", ascending=False)
    )

    return df, aggregated

def construct_matrix(augmented_patterns: dict):
    '''
    Takes patterns with the associated TFBM.
    Constructs a .csv of two columns TFBM , n_seqlets
    '''
    pass


def pwm_convolve_pair(pwm1: np.ndarray, pwm2: np.ndarray) -> float:
    """
    Compute the convolution between two PWMs (motif1 and motif2),
    allowing partial overlaps.

    Parameters
    ----------
    pwm1 : np.ndarray
        PWM matrix of shape (4, n1)
    pwm2 : np.ndarray
        PWM matrix of shape (4, n2)

    Returns
    -------
    float
        Maximum convolution score between the two motifs.
    """
    n1 = pwm1.shape[1]
    n2 = pwm2.shape[1]
    start_min = -n1 + 1
    start_max = n2

    scores = []
    for offset in range(start_min, start_max):
        # Compute overlapping region
        pwm1_start = max(0, -offset)
        pwm1_end = min(n1, n2 - offset)
        pwm2_start = max(0, offset)
        pwm2_end = min(n2, offset + n1)

        if pwm1_end > pwm1_start and pwm2_end > pwm2_start:
            slice1 = pwm1[:, pwm1_start:pwm1_end]
            slice2 = pwm2[:, pwm2_start:pwm2_end]
            score = np.sum(slice1 * slice2)
            scores.append(score)
        else:
            scores.append(0.0)

    return np.max(scores)


if __name__ == "__main__":
    import os

    os.makedirs("Results/Interpretation/tfmodisco_coef")
    for treatment in os.listdir("Results/Interpretation/log2FC"):
        if treatment == "queries":
            continue
        
        patterns = get_patterns_seqlets(f"Results/Interpretation/log2FC/{treatment}/modisco_run/modisco_results.h5")
        df, result = get_argmax_similarities(patterns, "Data/RAW/JASPAR2024_CORE_non-redundant_pfms_meme.txt")
        result.to_csv(f"Results/Interpretation/tfmodisco_coef/{treatment}.csv")
