# 🌿 Evaluating predictive modeling of transcriptional stress responses in Arabidopsis thaliana 🌱

## 🛠️ Set-Up

Before starting, ensure the following tools are installed:

- [`BEDTools`]
- [`SAMtools`]
- [`BLASTp`]
- [`TOMTOM` (meme suite)]

Then, run the following to install all the required dependencies in a new conda environment.
```bash
# ✅ Create the environment with specific Python versions
conda create -n epmsAT python=3.8.18 -c conda-forge -y

# ✅ Activate the environment
conda activate epmsAT

# ✅ Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ✅ Install Python dependencies
pip install -r requirements.txt

```
## 📁 Gather the processed data:
mRNA counts are already processed into the time-series summaries and are available at https://figshare.com/articles/dataset/DESeq2_padj_results_ALL_csv_zip/29457890?file=55920620 . 

Store it (DESeq2_padj_results_ALL.csv) in Data/Processed.

- Prepare the rest of the DNA and DAP-seq data with:
```bash
bash Code/prepare_data.sh
```
This takes care of downloading the DNA, DAP-seq, protein sequences and processing it, including generating the protein families file which will be used for family-wise splitting.

## 📄 Dataset description:

We do not provide the raw counts files, however, we provide the processed summaries of the RNA-seq time series. Those are constructed using the .R files in Code/mRNA_processing .

This dataset can be used to construct a set of independent variables to try out different models using Arabidopsis thaliana genome information.

- **`DESeq2_padj_results_ALL.csv`** is a table of the time-series summaries in long format:

- **`gene`**  : Arabidopsis thaliana (col-0, TAIR10 annotation) identifier

- **`padj`**  : Bonferoni corrected p-value (per number of genes) of the likelihood ratio test between full model incorporating treatment effects and a null model incorporating only time effects.

- **`stat`**  : Likelihood ratio test statistic

- **`treatment`**  : Hormone or PAMP/DAMP applied:

        B: MeJA
        C: SA
        D: SA + MeJA
        G: ABA
        X: 3-OH-C10
        Y: chitooctaose
        Z: elf18
        W: flg22
        V: nlp20
        U: OGs
        T: Pep1

- **`Log2_Fold_Change`**  : The $\log_2$ fold change of the ratio of area under the curve of the fitted counts between treatment and control.

- **`amplitude`**  : maximum $\log_2$ fold change at a particular time point.

- **`average`**  : Average counts over treatment and control and all time points.

- **`neg_log10_pvalue`**  : Negative $\log_{10}$ pvalue of padj column.

## 📊  Run model performance evaluation
```bash
python Code/Training/run_analysis.py
python Code/Training/generate_results_table.py
```

## 🧠 Run TF-Modisco Analysis
```bash
python Code/Interpretation/get_shap.py # get shap values for log2FC and Quartile classification
python Code/Interpretation/run_modisco.py
python Code/Interpretation/process_modisco_report.py
```

## 📈 Construct plots
```bash
python Code/Plotting/make_plots.py
```
