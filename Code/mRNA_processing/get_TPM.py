import sys
sys.path.append(".")
import pandas as pd
import numpy as np
import subprocess

def generate_TPM(raw_mRNA, gene_lengths):
    '''
    1- Divide the read counts by the length of each gene in kilobases. This gives you reads per kilobase (RPK).
    2- Count up all the RPK values in a sample and divide this number by 1,000,000. This is your “per million” scaling factor.
    3- Divide the RPK values by the “per million” scaling factor. This gives you TPM.
    '''
    # Transform gene length in kb (RPK)
    gene_lengths["length"] = gene_lengths["length"] / 1000
    # Divide the raw counts
    raw_mRNA = raw_mRNA.merge(gene_lengths, left_index= True, right_index= True)
    raw_mRNA = raw_mRNA.dropna()
    raw_mRNA = raw_mRNA.div(raw_mRNA['length'], axis=0).drop("length", axis=1)
    # Get sample-wise counts
    count_sums = raw_mRNA.sum(0)
    # Divide number by 1_000_000
    count_sums = count_sums/1_000_000
    # Divide RPK values to obtain TPM
    raw_mRNA = raw_mRNA.div(count_sums, 1)
    return raw_mRNA

def generate_high_low(Y: pd.DataFrame) -> pd.DataFrame:
    """
    Classify rows based on row-wise max into low/high categories.
    - Low (<25th percentile) -> 0
    - Middle (25-75th percentile) -> 3
    - High (>75th percentile) -> 1
    
    Returns a DataFrame with only low and high rows (middle removed).
    """
    # take log_10
    Y = np.log10(Y+1)
    row_mean = Y.max(axis=1)

    low_thresh = row_mean.quantile(0.25)
    high_thresh = row_mean.quantile(0.75)

    classification = pd.Series(3, index=Y.index, name="class")  # middle = 3
    classification[row_mean < low_thresh] = 0
    classification[row_mean > high_thresh] = 1

    # Keep only low and high
    classification = classification[classification != 3]

    # Return as DataFrame (keeps 2D shape)
    return classification.to_frame()

def load_datasets():
    # get gene_lengths
    cmd = [ "Code/DNA_processing/gene_lenghts.sh",
            "Data/RAW/DNA/Ath/TAIR10_GFF3_genes.gff", 
            "Data/RAW/DNA/Ath/gene_lengths.txt"]
    subprocess.run(cmd, check=True)
    # Hormones
    df = pd.read_csv("Data/RAW/mRNA/hormone_treatments_raw/raw_count_matrix.txt", sep='\t', index_col=0)
    df = df.astype(float)
    columns_old = df.columns
    df = df.iloc[:, :-1]
    df.columns = columns_old[1:]
    # PAMP_DAMP
    df_pamp= pd.read_csv("Data/RAW/mRNA/PTI_raw/PRJEB25079_UncorrectedCounts.csv", index_col=0)
    df_pamp = df_pamp.astype(float)
    ## this works as R1_Col_3-OH-FA_000 for _ separated characteristics
    ## being replicate, accession, stimulus, time unit, being 0 the reference.
    ## Take only Col samples
    df_pamp = df_pamp.loc[:, df_pamp.columns.str.contains("Col")]
    df_pamp = df_pamp.merge(df, left_index= True, right_index= True)


    ## Load Gene lengths
    gene_lengths = pd.read_table("Data/RAW/DNA/Ath/gene_lengths.txt", header=None)
    gene_lengths.columns = ["gene", "length"]
    gene_lengths = gene_lengths.set_index("gene")
    gene_lengths = gene_lengths.astype(float)
    # Derive TCPM 
    TPM = generate_TPM(raw_mRNA = df.copy(), gene_lengths=gene_lengths.copy())
    Y = generate_high_low(TPM)
    TPM.to_csv("Data/Processed/Basal/TPM.csv")
    Y.to_csv("Data/Processed/Basal/up_down_q_tpm.csv")

if __name__ == "__main__":
    load_datasets()
