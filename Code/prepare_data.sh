#!/bin/bash

# mRNA stuff
echo "Parsing PTI data..."
python Code/mRNA_processing/parse_data.py
echo "Running R script..."
echo "Differential expression analysis..."
Rscript "Code/mRNA_processing/get_mRNA_statistics.R"
echo "Differential expression analysis... completed."
echo "Merging hormone and PTI data..."

echo "mRNA processing done!"

# DNA stuff
# Chromosome file
wget -O Data/RAW/DNA/Ath/TAIR10_chr_all.fas https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas.gz
# GFF file
wget -O Data/RAW/DNA/Ath/TAIR10_GFF3_genes.gff https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff
# Get protein files
wget -O Data/RAW/Protein/TAIR10_pep_20110103_representative_gene_model https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_blastsets/TAIR10_seq_20110103_representative_gene_model_updated
# Get DAP-seq files
wget -O Data/RAW/DAPSeq/Dapseq.zip http://neomorph.salk.edu/dap_web/pages/dap_data_v4/fullset/dap_download_may2016_peaks.zip
# Unzip DAP-seq files
unzip Data/RAW/DAPSeq/Dapseq.zip -d Data/RAW/DAPSeq/
