# the idea here is to take coordinates in the genome and dap seq data and provide a binary arrya
# This command indeed merges the stuff very fast:
# bedtools intersect
# -a Data/RAW/DNA/Ath/custom_promoter_coordinates_999up_1500down_TTS.bed
# -b Data/RAW/DAPseq/dap_data_v4/peaks/BBRBPC_tnt/BPC1_col_a/chr1-5/chr1-5_GEM_events.narrowPeak -wa -wb > overlaps.txt

import os
import pandas as pd
import numpy as np
import subprocess
import glob
from tqdm import tqdm


def aggregate_tf_binding(dap_data_folder, DNA_specs=[814, 200, 200, 814]):
    """
    Aggregates TF binding signals across all TFs and plots the average binding per genomic position.

    Parameters:
    - dap_data_folder: str, folder containing BED-like intersection files for each TF.
    - DNA_specs: list, lengths of the upstream and downstream regions for TSS and TTS.
    """
    # load the TSS and TTS coordinates
    TSS_coordinates_file = (
        "Data/RAW/DNA/Ath/custom_promoter_coordinates_*up_*down_TSS.bed"
    )
    TSS_coordinates_file = glob.glob(TSS_coordinates_file)[0]
    TTS_coordinates_file = (
        "Data/RAW/DNA/Ath/custom_promoter_coordinates_*up_*down_TTS.bed"
    )
    TTS_coordinates_file = glob.glob(TTS_coordinates_file)[0]
    df_TSS = pd.read_csv(
        TSS_coordinates_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "gene", "dot", "strand"],
    )
    df_TTS = pd.read_csv(
        TTS_coordinates_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "gene", "dot", "strand"],
    )
    files = os.listdir(dap_data_folder)
    genes = df_TSS["gene"].unique()
    n_genes = len(genes)
    position_counts_TSS = np.zeros(
        (n_genes, len(files), DNA_specs[0] + DNA_specs[1]), dtype=bool
    )
    position_counts_TTS = np.zeros(
        (n_genes, len(files), DNA_specs[2] + DNA_specs[3]), dtype=bool
    )
    TF_names = {}  # List to store TF names

    # Iterate over all DAP-seq intersection files
    for i, file in tqdm(enumerate(files), total=len(files)):  # iterates over TFs
        if not file.endswith(".bed"):  # Skip non-BED files
            continue

        TF_name = file.split(".")[0]  # Extract TF name
        # check if TSS or TTS is in the name, generate a dummy variable for that
        if "TSS" in TF_name:
            TSS = True
        elif "TTS" in TF_name:
            TSS = False
        else:
            print("file name does not contain TSS or TTS")
            continue
        try:
            df = pd.read_csv(os.path.join(dap_data_folder, file), sep="\t", header=None)
        except:
            print("Problems with: ", file)
            continue
        TF_names[i] = TF_name

        # Assuming BED format: (chrom, start, end, ..., score column at 8)
        for _, row in df.iterrows():  # iterates over the genes
            gene = row[13]  # Get gene name
            # get gene index
            gene_idx = np.where(genes == gene)[0][0]
            # get TSS and TTS coordinates
            if TSS:
                gene_coordinates = df_TSS[df_TSS["gene"] == gene]
            else:
                gene_coordinates = df_TTS[df_TTS["gene"] == gene]

            # get the relative position of the match
            if gene_coordinates.shape[0] == 0:
                continue
            start_gene = gene_coordinates["start"].values[0]
            end_gene = gene_coordinates["end"].values[
                0
            ]  # Actually it should always be in between
            # get the relative position of the match
            tf_match_position = max(row[1] - start_gene, 0)
            tf_end_position = min(row[2] - start_gene, end_gene - start_gene)

            for pos in range(tf_match_position, tf_end_position):
                if TSS:
                    position_counts_TSS[gene_idx, i, pos] = 1
                else:
                    position_counts_TTS[gene_idx, i, pos] = 1

    # store the arrays
    np.save("Data/Processed/TF_binding_TSS.npy", position_counts_TSS)
    np.save("Data/Processed/TF_binding_TTS.npy", position_counts_TTS)
    # it is important, however, to know which dimension obeys to which TF
    # save the dictionary as csv
    with open("Data/Processed/TF_names.csv", "w") as f:
        for key in TF_names.keys():
            f.write("%s,%s\n" % (key, TF_names[key]))


def calculate_verlap(df):
    # columns 1, 2 and 11, 12 define the coordinates. make a function to calculate the overlap
    coordinates = df[[1, 2, 11, 12]]
    # get the maximum start and the minimum end
    df["start"] = coordinates[[1, 11]].max(axis=1)
    df["end"] = coordinates[[2, 12]].min(axis=1)
    # now calculate how many bp overlap
    df["overlap"] = df["end"] - df["start"]
    return df


def _generate_peaks(genes, dap_data_folder, cutoff_percentile, output_folder):
    # iterate over the TSS and the TTS (are in the same folder)
    for file in os.listdir(dap_data_folder):
        assert file is not None
        TF_name = file.split(".")[0]
        try:
            df = pd.read_csv(dap_data_folder + "/" + file, sep="\t", header=None)
            # if genes are there several times, take the one with the highest score
            df = df.groupby(13)[8].max().reset_index()
            df = df[[8, 13]]
            df.columns = [TF_name, "gene"]
            # assert all genes are in genes
            assert df["gene"].isin(genes["gene"]).all()
            # assert they are unique
            assert df["gene"].nunique() == df.shape[0]
            genes = pd.merge(genes, df, left_on="gene", right_on="gene", how="left")
        except:
            print("Problems with: ", file)
            continue

    # get quantiles over all entries
    genes = genes.set_index("gene")
    values_peaks = genes.values
    cutoff = np.nanquantile(values_peaks, float(cutoff_percentile))
    # convert NaN to 0
    genes = genes.fillna(0)
    # make 0 values below the cutoff
    genes = genes.applymap(lambda x: 0 if x <= cutoff else 1)
    # Lastly, because I am taking TSS and TTS, merge those columns using an OR
    # first remove the _TSS and _TTS
    # genes.columns = genes.columns.str.replace("_TSS", "").str.replace("_TTS", "")
    # genes = genes.groupby(level = 0, axis = 1).max()

    # If ==> Insetad of merging into one taking the or, maintain the position, which means, besically leaving it as it is

    genes_TTS = genes.filter(like="_TTS")
    genes_TSS = genes.filter(like="_TSS")
    # save the files
    genes_TSS.to_csv(output_folder + "/genes_TSS.csv")
    genes_TTS.to_csv(output_folder + "/genes_TTS.csv")


def generate_peaks(
    TSS_file,
    TTS_file,
    DAPseq_folder_raw,
    DAPseq_folder_raw_output,
    DAPseq_folder,
    cutoff_percentile=0.25,
):
    # first call the bash script
    cmd = [
        "DAPseq_parsing/intersect_dap_seq.sh",
        TSS_file,
        TTS_file,
        DAPseq_folder_raw,
        DAPseq_folder_raw_output,
    ]
    subprocess.run(cmd, check=True)
    genes = pd.read_csv(TSS_file, sep="\t", header=None)[3].reset_index()
    genes = genes.rename(columns={3: "gene"})
    genes.drop(columns="index", inplace=True)
    _generate_peaks(genes, DAPseq_folder_raw_output, cutoff_percentile, DAPseq_folder)


if __name__ == "__main__":
    aggregate_tf_binding("Data/RAW/DAPseq/dap_overlap", bin_size=10)
