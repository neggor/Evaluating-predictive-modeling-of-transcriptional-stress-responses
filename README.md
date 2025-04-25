# 🌿 Evaluating predictive modeling of transcriptional stress responses in Arabidopsis thaliana 🌱

## 🛠️ Set-Up

Before starting, ensure the following tools are installed:

- [`BEDTools`]
- [`SAMtools`]
- [`BLASTp`]

Then, run the following to install all the required dependencies in a new conda environment.
```bash
# ✅ Create the environment with specific Python and R versions
conda create -n epmsAT python=3.8.18 r-base=4.4.3 -c conda-forge -y

# ✅ Activate the environment
conda activate epmsAT

# ✅ Install Python dependencies
pip install -r requirements.txt

# ✅ Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ✅ Install DESeq2 
conda install -c conda-forge -c bioconda bioconductor-deseq2 -y

# ✅ Install stringr 
conda install conda-forge::r-stringr -y

```
## 📁 Gather the data.
- Download mRNA-seq data: 
    - Hormone data: ???. 
    - PAMPs/DAMPs: From https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-9694, download "Processed Data" into Data/RAW/mRNA/PTI_raw/PRJEB25079_UncorrectedCounts.csv
- Prepare the rest of the DNA and DAP-seq data with:
```bash
bash Code/prepare_data.sh
```
Which takes care of conducting the differential expression analysis with DESeq2, downloading the DNA, DAP-seq, protein sequences and processing it, including generating the protein families file which will be used for family-wise splitting. This should take less than 1h.

## 📊  Run model performance evaluation
```bash
```
### Check relationship with average expression
```bash
```
## 🧠 Run TF-Modisco Analysis
```bash
python Code/Interpretation/get_shap.py # get shap values for log2FC and Quartile classification
python Code/Interpretation/run_modisco.py
python Code/Interpretation/process_modisco_report.py
```
## 📈 Construct plots
```bash
```
## Overview code
