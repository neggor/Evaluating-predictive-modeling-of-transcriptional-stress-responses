import h5py
import numpy as np
import logomaker as lm
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import subprocess
from tqdm import tqdm
import matplotlib.colors as mcolors
import requests
from bs4 import BeautifulSoup
import time

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


def plot_pattern(PWM: np.ndarray, folder, name):
    # Create a DataFrame
    pwm_df = pd.DataFrame(PWM)
    pwm_df.columns = ["A", "C", "G", "T"]
    # Visualize it as a sequence logo
    logo = lm.Logo(pwm_df)
    plt.title(name)
    plt.savefig(f"{folder}/{name}.png")
    plt.close("all")


def plot_pattern_ax(PWM: np.ndarray, ax):
    # Create a DataFrame
    pwm_df = pd.DataFrame(PWM)
    pwm_df.columns = ["A", "C", "G", "T"]
    # Visualize it as a sequence logo
    logo = lm.Logo(pwm_df, ax=ax)

def construct_seqlet_histogram(
    h5_file, store_path=".", offset=1, DNA_specs=[814, 200, 200, 814], binning_size=1
):
    post_bins = {}
    neg_bins = {}

    with h5py.File(h5_file, "r") as f:
        for direction in ["neg_patterns", "pos_patterns"]:
            for pattern_n in f[direction].keys():
                if f[direction][pattern_n]["seqlets"]["n_seqlets"][0] < 1000:
                    continue
                starts = np.array(f[direction][pattern_n]["seqlets"]["start"])
                ends = np.array(f[direction][pattern_n]["seqlets"]["end"])

                # Adjust for offset
                starts[starts > 814 + 200 - offset] += offset * 2 + 20
                ends[ends > 814 + 200 - offset] += offset * 2 + 20

                # Construct histogram line
                line = construct_line(starts, ends, DNA_specs, binning_size, offset)
                if direction == "neg_patterns":
                    neg_bins[pattern_n] = line
                else:
                    post_bins[pattern_n] = line

    fig, ax = plt.subplots(figsize=(20, 5))

    # Plot the lines
    plot_lines = []
    labels = []
    motif_images = {}

    for pattern_n, line in post_bins.items():
        plot_lines.append(ax.plot(line, label=f"pos {pattern_n}")[0])
        labels.append(f"pos {pattern_n}")
        motif_images[f"pos {pattern_n}"] = (
            f"{store_path}/pos_patterns_{pattern_n}_PWM.png"
        )

    for pattern_n, line in neg_bins.items():
        plot_lines.append(ax.plot(-line, label=f"neg {pattern_n}")[0])
        labels.append(f"neg {pattern_n}")
        motif_images[f"neg {pattern_n}"] = (
            f"{store_path}/neg_patterns_{pattern_n}_PWM.png"
        )

    # Set y-axis ticks to always be positive
    max_count = int(
        np.max(
            (
                np.max([post_bins[pattern_n] for pattern_n in post_bins]),
                np.max([neg_bins[pattern_n] for pattern_n in neg_bins]),
            )
        )
    )

    # Add vertical reference lines for TSS and TTS
    ax.axvline(DNA_specs[0], color="black", linestyle="dashed", linewidth=1)
    ax.axvline(
        DNA_specs[0] + DNA_specs[1] + DNA_specs[2] + 20,
        color="black",
        linestyle="dashed",
        linewidth=1,
    )

    # Adjust x-axis ticks for readability
    ax.set_xticks(
        [
            0,
            DNA_specs[0],
            DNA_specs[0] + DNA_specs[1] + 10,
            DNA_specs[0] + DNA_specs[1] + DNA_specs[2] + 20,
            sum(DNA_specs) + 20,
        ]
    )
    ax.set_xticklabels(
        [
            "0",
            f"{DNA_specs[0]} (TSS)",
            "Padding",
            f"{DNA_specs[0] + DNA_specs[1] + DNA_specs[2] + 20} (TTS)",
            f"{sum(DNA_specs) + 20}",
        ]
    )

    # Explicitly break the plot in the middle
    ax.axvline(1010, color="black", linestyle="dashed", linewidth=1)
    ax.axvline(1038, color="black", linestyle="dashed", linewidth=1)

    ax.set_xlabel("Nucleotide position")

    ax.set_yticks(np.arange(-max_count, max_count, 100))
    ax.set_yticklabels(np.abs(np.arange(-max_count, max_count, 100)))
    ax.legend(plot_lines, labels, loc="upper right", bbox_to_anchor=(1.1, 1.1))
    plt.savefig(f"{store_path}/seqlet_histogram.png")


def construct_line(starts, ends, DNA_specs, bining_size, offset):
    """
    Construct a line of 0s and 1s
    """
    line = np.zeros((sum(DNA_specs) + 20) // bining_size)
    for start, end in zip(starts, ends):
        bin_start = start // bining_size
        bin_end = end // bining_size
        if bin_start <= 814 + 200 - offset and bin_end >= 814 + 200 + 20 + offset:
            continue
            # Because these are patters that don not make much sense,
            # are a combinations of the two extremes
            # TODO fix this, hypothetical scores to very very low numbers
            # and the queries to random values in the padding?
        line[bin_start:bin_end] += 1
    return line


def create_meme_file(group_pwms, group_name, output_dir):
    """
    Create a .meme file for a group of PWMs.
    """
    meme_file = f"{output_dir}/{group_name}.meme"
    with open(meme_file, "w") as f:
        f.write("MEME version 5\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies (from uniform background):\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        for i, pwm in enumerate(group_pwms):
            f.write(f"MOTIF {group_name}_motif{i}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]}\n")
            for row in pwm:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
            f.write("\n")

    return meme_file


def run_tomtom(meme_file, jaspar_db, output_dir):
    """
    Run Tomtom to compare the .meme file against the JASPAR database.
    """
    tomtom_output = f"{output_dir}"
    command = [
        "tomtom",
        "-dist",
        "kullback",
        "-thresh",
        "0.05",
        "-oc",
        output_dir,
        meme_file,
        jaspar_db,
    ]
    subprocess.run(command, check=True)
    return tomtom_output + "/tomtom.tsv"


def parse_tomtom_output(tomtom_output):
    """
    Parse the Tomtom output and return the TF with the most hits.
    """
    # Load the Tomtom results
    results = pd.read_csv(tomtom_output, sep="\t", comment="#")
    print(results)
    # check if there are any results
    if results.shape[0] == 0:
        return "No match", 1, "No match", "No match"
    # these are all signifficant hits!
    most_hits = results["Target_ID"].value_counts().idxmax()
    # import pdb; pdb.set_trace()
    most_hits_row = (
        results[results["Target_ID"] == most_hits].sort_values("Overlap").iloc[-1]
    )
    # get lower q-value
    # most_hits_row = results.sort_values("q-value").iloc[0]
    return (
        most_hits_row["Target_ID"],
        most_hits_row["q-value"],
        most_hits_row["Query_consensus"],
        most_hits_row["Target_consensus"],
    )


def extract_patterns(h5_file, treat_name, seqlet_threshold=500):
    """
    Extract patterns from the h5 file
    """
    with h5py.File(h5_file, "r") as f:
        patterns = {}
        for direction in ["neg_patterns", "pos_patterns"]:
            print("Direction:", direction)
            if direction not in f:
                continue
            for key in f[direction].keys():
                # get n_seqlets
                if f[direction][key]["seqlets"]["n_seqlets"][0] < seqlet_threshold:
                    continue
                PWM = np.array(f[direction][key]["sequence"][:])
                CWM = np.array(f[direction][key]["contrib_scores"][:])
                n_seqlets = str(f[direction][key]["seqlets"]["n_seqlets"][0])
                PWM, CWM = filter_based_on_importance(PWM, CWM, 0.45)
                PWM, CWM = filter_based_on_entropy(PWM, CWM, 0.7)
                if PWM is None or PWM.shape[0] < 3:
                    continue

                patterns[treat_name + "_" + direction + "_" + key + "_" + n_seqlets] = (
                    PWM
                )

    return patterns


def pwm_similarity(pwm1, pwm2):
    """
    Compute the similarity between two PWMs of different lengths using
    sliding window Pearson correlation, considering both forward and reverse complement strands.

    Parameters:
    pwm1, pwm2: np.ndarray
        PWMs represented as numpy arrays of shape (L, 4), where L is the length
        and 4 corresponds to A, C, G, T frequencies.

    Returns:
    max_corr: float
        Maximum Pearson correlation found over all alignments.
    """

    def reverse_complement(pwm):
        """
        Compute the reverse complement of a PWM.

        Parameters:
        pwm: np.ndarray
            PWM represented as a numpy array of shape (L, 4).

        Returns:
        rev_comp_pwm: np.ndarray
            Reverse complement of the input PWM.
        """
        return pwm[::-1, ::-1]

    len1, len2 = pwm1.shape[0], pwm2.shape[0]

    # Ensure pwm1 is the longer one for consistency
    if len1 < len2:
        pwm1, pwm2 = pwm2, pwm1
        len1, len2 = len2, len1

    max_corr = -1  # Minimum possible correlation

    # Slide the shorter PWM over the longer one
    for i in range(len1 - len2 + 1):
        sub_pwm1 = pwm1[i : i + len2]  # Extract submatrix of same length as pwm2

        # Flatten and compute Pearson correlation for both strands
        corr_fwd = pearsonr(sub_pwm1.flatten(), pwm2.flatten())[0]
        corr_rev = pearsonr(sub_pwm1.flatten(), reverse_complement(pwm2).flatten())[0]
        max_corr = max(max_corr, corr_fwd, corr_rev)

    return max_corr


def parse_jaspar_meme(file_path):
    """
    Parses a JASPAR MEME .txt file to extract a dictionary mapping
    motif ID to transcription factor (TF) name.

    Parameters:
    file_path : str
        Path to the MEME .txt file.

    Returns:
    dict
        Dictionary where keys are motif IDs and values are TF names.
    """
    motif_dict = {}

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("MOTIF"):  # Identify motif definition lines
                parts = line.strip().split()
                if len(parts) >= 3:
                    motif_id = parts[1]  # e.g., "MA0001.1"
                    tf_name = " ".join(parts[2:])  # e.g., "AGL3"
                    motif_dict[motif_id] = tf_name

    return motif_dict


def cluster_patterns(n_clusters, output_dir, jaspar, seqlet_threshold=500):
    """
    Cluster the patterns
    """
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

    pat_dict = {}
    for treat in mapping.values():
        if os.path.exists(
            f"Results/Interpretation/log2FC/{treat}/modisco_run/modisco_results.h5"
        ):
            h5_file = (
                f"Results/Interpretation/log2FC/{treat}/modisco_run/modisco_results.h5"
            )
            patterns = extract_patterns(h5_file, treat, seqlet_threshold)
            pat_dict.update(patterns)

    # Compute similarity matrix
    similarity_matrix = np.zeros((len(pat_dict), len(pat_dict)))
    for i, (name1, pwm1) in tqdm(enumerate(pat_dict.items()), total=len(pat_dict)):
        for j, (name2, pwm2) in enumerate(pat_dict.items()):
            similarity_matrix[i, j] = pwm_similarity(pwm1, pwm2)

    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_matrix
    # check if symmetric
    assert np.allclose(distance_matrix, distance_matrix.T)
    # enforce symmetry over num. errors
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    # make diag 0
    np.fill_diagonal(distance_matrix, 0)
    # Convert square distance matrix to condensed distance matrix
    condensed_distance_matrix = squareform(distance_matrix)

    # Perform hierarchical clustering with 20 clusters
    linkage_matrix = linkage(condensed_distance_matrix, method="complete")
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    # Add cluster assignments to the patterns
    pattern_names = list(pat_dict.keys())
    cluster_assignments = {
        name: cluster for name, cluster in zip(pattern_names, clusters)
    }
    # print wich patterns are in each cluster
    for i in range(1, n_clusters + 1):
        print(
            f"Cluster {i}: {', '.join([name for name, cluster in cluster_assignments.items() if cluster == i])}"
        )
    # Create a table with treatments as columns and clusters as rows
    treatments = list(mapping.values())
    cluster_table = pd.DataFrame(
        0,
        index=range(1, n_clusters + 1),
        columns=treatments + ["best_match", "query_consensus"],
    )

    # Populate the table
    for pattern_name, cluster in cluster_assignments.items():
        treat = pattern_name.split("_")[0]  # Extract treatment name
        direction = pattern_name.split("_")[1]  # Extract direction (pos/neg)
        n_seqlets = int(pattern_name.split("_")[-1])  # Extract number of seqlets
        if direction == "pos":
            # add the number of seqlets
            cluster_table.loc[cluster, treat] += n_seqlets  # Positive pattern

        elif direction == "neg":
            cluster_table.loc[cluster, treat] -= n_seqlets  # Negative pattern

    # average over the pamps/damps
    cluster_table["PTI"] = cluster_table[
        ["3-OH10", "chitooct", "elf18", "flg22", "nlp20", "OGs", "Pep1"]
    ].mean(axis=1)
    # and drop the columns
    cluster_table.drop(
        columns=["3-OH10", "chitooct", "elf18", "flg22", "nlp20", "OGs", "Pep1"],
        inplace=True,
    )
    # reorder columns
    cluster_table = cluster_table[
        [
            "MeJA",
            "SA",
            "SA+MeJA",
            "ABA",
            "ABA+MeJA",
            "PTI",
            "best_match",
            "query_consensus",
        ]
    ]
    print(cluster_table)
    # if a cluster does not have more than 1000 seqlets, in total we remove it:
    cluster_table = cluster_table[cluster_table.iloc[:, :-2].abs().sum(axis=1) > 100]
    # Now enrichment for each group
    os.makedirs(output_dir, exist_ok=True)
   
    cluster_dict = {}
    for i, (name, cluster) in enumerate(cluster_assignments.items()):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(pat_dict[name])

    print(cluster_dict)
    motif_to_tf = parse_jaspar_meme(jaspar)
    for group_name, group_pwms in cluster_dict.items():
        # if cluster name is not in the table, we skip it
        if group_name not in cluster_table.index:
            continue
        meme_file = create_meme_file(group_pwms, f"cluster_{group_name}", output_dir)
        tomtom_output = run_tomtom(
            meme_file, jaspar, output_dir + f"/cluster_{group_name}"
        )
        best_match, q_value, query_consensus, target_consensus = parse_tomtom_output(
            tomtom_output
        )
        print(
            f"Cluster {group_name}: {best_match} (q-value: {q_value}), {query_consensus} vs {target_consensus}"
        )
        # now we put the query consensus in the table corresponding to the group
        cluster_table.loc[group_name, "best_match"] = (
            motif_to_tf[best_match] if best_match in motif_to_tf else "No match"
            #map_tf_to_family(best_match)  if best_match in motif_to_tf else "No match"
        )
        cluster_table.loc[group_name, "query_consensus"] = query_consensus
        if best_match == "No match":
            consensus = "".join("ACGT"[np.argmax(row)] for row in group_pwms[0])
            cluster_table.loc[group_name, "query_consensus"] = consensus

    # remvoe if No match
    # cluster_table = cluster_table[cluster_table["best_match"] != "No match"]
    cluster_table.reset_index(inplace=True)
    # drop the index
    cluster_table.drop(columns=["index"], inplace=True)
    cluster_table.to_csv(f"{output_dir}/cluster_table.csv")
    # display a nice latex table
    print(cluster_table)

def map_tf_to_family(TF_ID):
    url = f"https://jaspar.elixir.no/matrix/{TF_ID}/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Loop through all rows in the metadata table
        for row in soup.select("#matrix-detail tr"):
            key_cell = row.find('td')
            if key_cell and 'Family:' in key_cell.text:
                value_cell = key_cell.find_next_sibling('td')
                return value_cell.text.strip() if value_cell else 'Unknown'
        
        return 'Unknown'
    except Exception as e:
        print(f"Error with {TF_ID}: {e}")
        return 'Unknown'

if __name__ == "__main__":
    cluster_patterns(
        5,
        "Results/Interpretation/cluster_patterns",
        "Data/RAW/MOTIF/JASPAR_2024_PLANT_motifs.txt",
        30,
    )
