# the idea here is to take coordinates in the genome and dap seq data and provide a binary arrya
# This command indeed merges the stuff very fast:
# bedtools intersect
# -a Data/RAW/DNA/Ath/custom_promoter_coordinates_999up_1500down_TTS.bed
# -b Data/RAW/DAPseq/dap_data_v4/peaks/BBRBPC_tnt/BPC1_col_a/chr1-5/chr1-5_GEM_events.narrowPeak -wa -wb > overlaps.txt

import os
import pandas as pd
import numpy as np
import subprocess


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
    # now in both files we must have the same TFs
    colanmes_TSS = [col.split("_")[0] for col in genes_TSS.columns]
    colanmes_TTS = [col.split("_")[0] for col in genes_TTS.columns]
    #  select the ones outside the intersection
    intersect = set(colanmes_TSS).intersection(set(colanmes_TTS))
    # select only the ones that in the intersection
    genes_TSS = genes_TSS.loc[:, [col +"_TSS" for col in intersect]]
    genes_TTS = genes_TTS.loc[:, [col +"_TTS" for col in intersect]]

    # order the columns (which is critical)
    genes_TSS = genes_TSS.reindex(sorted(genes_TSS.columns), axis=1)
    genes_TTS = genes_TTS.reindex(sorted(genes_TTS.columns), axis=1)

    os.makedirs(output_folder, exist_ok=True)
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
        "Code/DAPseq_processing/intersect_dap_seq.sh",
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
