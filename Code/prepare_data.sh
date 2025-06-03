#!/bin/bash

# mRNA stuff (This assumes you have access to the raw data!)
mkdir -p Data/Processed/mRNA

#echo "Parsing PTI data..."
#python Code/mRNA_processing/parse_data.py
#echo "Running R script..."
#echo "Differential expression analysis..."
#Rscript "Code/mRNA_processing/get_mRNA_statistics.R"
#echo "Differential expression analysis... completed."
#echo "Merging hormone and PTI data..."
#python Code/mRNA_processing/merge_treatments.py #
#echo "mRNA processing done!"
# The assumption is that the only thing made available is Data/Processed/mRNA/DESeq2_padj_results_ALL.csv

# DNA stuff
# Chromosome file
mkdir -p Data/RAW/DNA/Ath
wget -O Data/RAW/DNA/Ath/TAIR10_chr_all.fas.gz https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas.gz
# Unzip the chromosome file
gunzip Data/RAW/DNA/Ath/TAIR10_chr_all.fas.gz
# GFF file
wget -O Data/RAW/DNA/Ath/TAIR10_GFF3_genes.gff https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff

# Get protein files
mkdir -p Data/RAW/Protein
wget -O Data/RAW/Protein/TAIR10_pep_20110103_representative_gene_model https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_blastsets/TAIR10_pep_20110103_representative_gene_model_updated
# generate protein families file
python Code/DNA_processing/get_gene_families.py
# Get DAP-seq files
mkdir -p Data/RAW/DAPseq
wget -O Data/RAW/DAPseq/Dapseq.zip http://neomorph.salk.edu/dap_web/pages/dap_data_v4/fullset/dap_download_may2016_peaks.zip
# Unzip DAP-seq files
unzip Data/RAW/DAPseq/Dapseq.zip -d Data/RAW/DAPseq/
# Remove the zip file
rm Data/RAW/DAPseq/Dapseq.zip
# Fix stuff with upper case
# Loop through all .narrowPeak files in the folder
for TF_family in Data/RAW/DAPseq/dap_data_v4/peaks/*; do
    echo "Processing files in $TF_family"
    for TF in "$TF_family"/*; do
        # check if there is a chr1-5 named folder
        if [ -d "$TF/chr1-5" ]; then
            echo "Processing files in $TF/chr1-5"
            for file in "$TF/chr1-5"/*; do
                # Create a temporary file for the output
                echo "Processing $file"
                temp_file="${file}.tmp"
                
                # Convert only the first letter of the chromosome names to uppercase
                awk 'BEGIN {OFS="\t"} { $1 = toupper(substr($1, 1, 1)) substr($1, 2); print }' "$file" > "$temp_file"
                
                # Replace the original file with the modified file
                mv "$temp_file" "$file"
            done
        fi
    done
done
# Donwload the JASPAR FILE
mkdir -p Data/RAW/MOTIF
wget -O Data/RAW/MOTIF/JASPAR_2024_PLANT_motifs.txt https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt

echo "ALL done!"