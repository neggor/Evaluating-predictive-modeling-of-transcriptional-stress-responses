# To concatenate the leaf hormone treatments with the seddlings PAMPs/DAMPs treatments.
import pandas as pd

hormone = pd.read_csv(
    "Data/Processed/mRNA/DESeq2_padj_results_Hormone.csv", index_col=0
)
dmp = pd.read_csv("Data/Processed/mRNA/DESeq2_padj_results_PTI.csv", index_col=0)

# concatenate
# Incorporating all
df = pd.concat([hormone, dmp], axis=0)


# save
df.to_csv("Data/Processed/mRNA/DESeq2_padj_results_ALL.csv")
