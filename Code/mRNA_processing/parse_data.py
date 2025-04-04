import pandas as pd
import numpy as np

X = pd.read_csv("Data/RAW/mRNA/PTI_raw/PRJEB25079_UncorrectedCounts.csv", index_col=0)
# this works as R1_Col_3-OH-FA_000 for _ separated characteristics
# being replicate, accession, stimulus, time unit, being 0 the reference.
# Take only Col samples
X = X.loc[:, X.columns.str.contains("Col")]

# make a .csv per each stimulus
stimulus = X.columns.str.extract(r"Col_(.*)_")[0].unique()

print(stimulus)
names = ["control", "X", "Y", "Z", "W", "V", "U", "T"]
datasets = []
for i, s in enumerate(['mock', '3-OH10', 'chitooct', 'elf18', 'flg22', 'nlp20', 'OGs', 'Pep1']):
    tmpX = X.loc[:, X.columns.str.contains(s)]
    # For some reason the number of the replicate is the last one (sic!)
    # remove the col info
    tmpX.columns = tmpX.columns.str.extract(r"Col_(.*)")[0]
    # only if the last number after the last _ has 4 digits it is indicating the replicate, otherwise is the time
    time = tmpX.columns.str.extract(rf"{s}_(.*)")[0]
    # remove the last digit if the number has more than 3 digits
    replicate = time.str.extract(r"(\d{4})")[0]
    time = time.str.extract(r"(\d{3})")[0]
    # get the last digit in each number 
    replicate = replicate.str[-1]
    replicate = replicate.fillna("0")
    # Now generate new colnames using time and replicate
    tmpX.columns = names[i] + "_" + time + "_" + replicate 
    datasets.append(tmpX)

# concatenate
X = pd.concat(datasets, axis=1)
# save 
X.to_csv("Data/RAW/mRNA/PTI_raw/RAW_COUNTS_PTI.csv")

