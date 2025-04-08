# 🌿 Evaluating predictive modeling of transcriptional stress responses in Arabidopsis thaliana 🌱

## 🛠️ Set-Up

Before starting, ensure the following tools are **installed and available in your `PATH`**:

- [`BEDTools`]
- [`SAMtools`]
- [`BLASTp`]

Then, run the following to install all the required dependencies in a new conda environment.
```bash
conda create -n epmsAT python=3.8.18 r-base=4.4.3 -c conda-forge -y

conda activate epmsAT

pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

conda install -c bioconda bioconductor-deseq2

conda install conda-forge::r-stringr
```
## 📁 Gather the data.
- Download mRNA-seq data: 
    - Hormone data: ???. 
    - PAMPs/DAMPs: From https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-9694, download "Processed Data" into Data/RAW/mRNA/PTI_raw/PRJEB25079_UncorrectedCounts.csv
- Prepare the rest of the DNA and DAP-seq data with:
```bash
bash scripts/prepare_data.sh
```
Which takes care of conducting the differential expression analysis with DESeq2, downloading the DNA, DAP-seq, protein sequences and processing it, including generating the protein families file which will be used for family-wise splitting. This should take less than 1h.

## 📊  Run model performance evaluation

## 🧠 Run TF-Modisco Analysis

## 📈 Construct plots

## Overview code
