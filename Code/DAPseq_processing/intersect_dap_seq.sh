#!/bin/bash


if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
	echo "Usage: $0 <TSS_coordinates_path> <TTS_coordinates_path> <DAPseq_folder_path> <storage_folder>"
	exit 1
fi

# Check that bedtools is installed
if ! command -v bedtools &> /dev/null; then
	echo "bedtools is not installed. Please install bedtools before running this script."
	exit 1
fi

# Assign input parameters to variables
TSS_coordinates_path="$1"
TTS_coordinates_path="$2"
dap_seq_folder_path="$3"
storage_folder="$4"

echo "TSS coordinates path: $TSS_coordinates_path"
echo "TTS coordinates path: $TTS_coordinates_path"
echo "DAPseq folder path: $dap_seq_folder_path"
echo "Storage folder: $storage_folder"
echo "-----------------------------------------------------------------------------------------------------------------"
# if folder does not exist, create it
if [ ! -d "$storage_folder" ]; then
    mkdir -p "$storage_folder"
fi


# Loop through all .narrowPeak files in the folder
for TF_family in "$dap_seq_folder_path"/*; do
    echo "Processing files in $TF_family"
    for TF in "$TF_family"/*; do
        # check if there is a chr1-5 named folder
        if [ -d "$TF/chr1-5" ]; then
            echo "Processing files in $TF/chr1-5"
            for file in "$TF/chr1-5"/*; do
                # if the file contains a "colmap" in the name, skip it, this is ignoring methylation 
                if [[ $file == *"colmap"* ]]; then
                    echo "Skipping $file"
                else
                   # now, call bedtools to do the intersect with the TSS and store it
                    echo  "Processing $file"
                    file_name=$(basename "$TF" | sed 's/_.*//')
                    bedtools intersect -a "$file" -b "$TSS_coordinates_path" -wa -wb > "$storage_folder/${file_name}_TSS.bed"
                    # now, call bedtools to do the intersect with the TTS and store it
                    bedtools intersect -a "$file" -b "$TTS_coordinates_path" -wa -wb > "$storage_folder/${file_name}_TTS.bed"
                fi
            done
        fi
    done
done


