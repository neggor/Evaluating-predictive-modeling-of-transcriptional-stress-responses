# 🌿 Evaluating predictive modeling of transcriptional stress responses in Arabidopsis thaliana 🌱

## 🛠️ Set-Up

Before starting, ensure the following tools are installed:

- [`BEDTools`]
- [`SAMtools`]
- [`BLASTp`]
- [`TOMTOM` (meme suite)]

Then, run the following to install all the required dependencies in a new conda environment.
```bash
# ✅ Create the environment with specific Python and R versions
conda create -n epmsAT python=3.8.18 -c conda-forge -y

# ✅ Activate the environment
conda activate epmsAT

# ✅ Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ✅ Install Python dependencies
pip install -r requirements.txt

```
## 📁 Gather the processed data:
mRNA counts are already processed into the time-series summaries and are available at ... ?
- Prepare the rest of the DNA and DAP-seq data with:
```bash
bash Code/prepare_data.sh
```
This takes care of downloading the DNA, DAP-seq, protein sequences and processing it, including generating the protein families file which will be used for family-wise splitting.


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
