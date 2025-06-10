#!/bin/bash


# This is a general script for extracting the desired sequences for an arbitrary .fast DNA file and a gff3 file

# Check if input file, upstream, downstream, upstream TTS, downstream TTS and directory are provided
# The folder for the input also will be used for the output, this is, the sequences per gene will be stored in the same folder as the input files
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
	echo "Usage: $0 <upstream_bp_TSS> <downstream_bp_TSS> <upstream_bp_TTS> <downstream_bp_TTS> <folder> <mask_exons>"
	exit 1
fi

# Check that bedtools is installed
if ! command -v bedtools &> /dev/null; then
	echo "bedtools is not installed. Please install bedtools before running this script."
	exit 1
fi
# Check that samtools is installed
if ! command -v samtools &> /dev/null; then
    echo "samtools is not installed. Please install samtools before running this script."
    exit 1
fi

# Assign input parameters to variables
UPSTREAM_BP="$1"
DOWNSTREAM_BP="$2"
UPSTREAM_BP_TTS="$3"
DOWNSTREAM_BP_TTS="$4"
FOLDER="$5"
MASK_EXONS="$6"

# First remove files from previous runs
rm -f ${FOLDER}/custom_promoter_coordinates_*up_*down_TSS.bed
rm -f ${FOLDER}/custom_promoter_coordinates_*up_*down_TTS.bed
rm -f ${FOLDER}/promoters_*up_*down_TSS.fasta
rm -f ${FOLDER}/promoters_*up_*down_TTS.fasta
rm -f ${FOLDER}/intron_coordinates.bed
rm -f ${FOLDER}/*_chromosome_lengths.txt
rm -f ${FOLDER}/*.fai
rm -f ${FOLDER}/masked_exons_chromosome.fasta
rm -f ${FOLDER}/exons_coordinates.bed

#### File handling
## Check that the .fasta and .gff/gff3 files are in the folder
## Each genome must be accompanied by a gff3 file, together in a folder with the name of the genome
## display the name

# List the files in the folder
echo "-----------------------------------------------------------------------------------------------------------------"
# Get .fasta file name
FASTA_O=$(ls "$FOLDER"/*.{fa,fas,fasta,fsa,fna} 2>/dev/null | head -n 1)

if [[ -z "$FASTA_O" ]]; then
    echo "No .fasta file found in $FOLDER."
else
    echo "Found: .fasta file: $FASTA_O"
fi

# Get .gff3 or .gff file name
GFF=$(ls "$FOLDER"/*.gff{,3} 2>/dev/null | head -n 1)

if [[ -z "$GFF" ]]; then
    echo "No .gff or .gff3 file found in $FOLDER."
else
    echo "Found: GFF file: $GFF"
fi

# Print wich name is the genome to be processed
NAME=$(basename "$(ls "$FOLDER"/*.{fa,fas,fasta,fsa,fna} 2>/dev/null | head -n 1)" | sed 's/\..*//')
echo "Processing genome: $NAME"
echo "-----------------------------------------------------------------------------------------------------------------"

if [[ "$MASK_EXONS" == "True" ]]; then
	echo "Masking exons with Ns"
	# Now, get exon coordinates and masked with an N:
	FASTA="${FOLDER}/masked_exons_chromosome.fasta"

	# Extract exon coordinates from the GFF file into BED format
	EXONS_BED="${FOLDER}/exons_coordinates.bed"

	awk 'BEGIN {FS=OFS="\t"} $3 == "exon" {
		split($9, id_array, /[=;]/)
		print $1, $4-1, $5, id_array[2], ".", $7
	}' "$GFF" > "$EXONS_BED"

	echo "Extracted exon coordinates saved to $EXONS_BED."

	# Mask the FASTA sequence for the exons
	bedtools maskfasta -fi "$FASTA_O" -bed "$EXONS_BED" -fo "$FASTA" -mc "N"

	echo "Masked exons saved to $FASTA."
else
	#echo "Not masking exons"
	FASTA="$FASTA_O"
fi




# Index the FASTA file
samtools faidx "$FASTA"

# Extract the chromosome lengths from the index file and save to a text file
cut -f1,2 "$FASTA.fai" > "$FOLDER/${NAME}_chromosome_lengths.txt"

# Substract 1 from DOWNSTREAM_BP_TSS and UPSTREAM_BP_TTS
DOWNSTREAM_BP=$((DOWNSTREAM_BP - 1))
UPSTREAM_BP_TTS=$((UPSTREAM_BP_TTS - 1))

#Convert GFF3 to BED with upstream and downstream regions
# || $3 == "pseudogene" || $3 == "transposable_element_gene"
awk -v upstream_bp="$UPSTREAM_BP" -v downstream_bp="$DOWNSTREAM_BP" 'BEGIN{FS = OFS ="\t"} $3 == "gene" {
if($7 == "+") {
	upstream_start = ($4 > upstream_bp) ? $4 - upstream_bp -1 : 0
	downstream_end = $4 + downstream_bp 
} else if($7 == "-") {
upstream_start = ($5 > downstream_bp) ? $5 - downstream_bp -1 : 0
downstream_end = $5 + upstream_bp 
} 
match($9, /Name=([^;]+)/, name_array)
    if (name_array[1] != "") {
        print $1, upstream_start, downstream_end, name_array[1], ".", $7
    }
}' "${GFF}" > "${FOLDER}/tmp_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

# Now limit the size to chromosome length to avoid bedtools skipping these requests:
awk 'BEGIN{OFS="\t"} ($3 == "chromosome") {chrom_lengths[$1] = $2} 
NR==FNR{chrom_lengths[$1]=$2; next} 
($1 in chrom_lengths) {if ($3 > chrom_lengths[$1]) $3 = chrom_lengths[$1]; print}'\
	"${FOLDER}/${NAME}_chromosome_lengths.txt" "${FOLDER}/tmp_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed" >\
	"${FOLDER}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

rm "${FOLDER}/tmp_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

echo "Conversion completed. Output saved to ${FOLDER}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

bedtools getfasta -fi ${FASTA} \
	-bed ${FOLDER}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed\
	-name -fo ${FOLDER}/promoters_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.fasta \
	-s # Force strandedness. If the feature occupies the antisense strand, the sequence will be reverse complemented. Absolutely crucial for position sensitive analysis.

echo "Fasta file saved to ${FOLDER}/promoters_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.fasta, good to go!"


# Convert GFF3 to BED with upstream and downstream regions
# || $3 == "pseudogene" || $3 == "transposable_element_gene"
awk -v upstream_bp="$UPSTREAM_BP_TTS" -v downstream_bp="$DOWNSTREAM_BP_TTS" 'BEGIN{FS = OFS ="\t"} $3 == "gene"{
if($7 == "+") {
	upstream_start = ($5 > upstream_bp) ? $5 - upstream_bp -1 : 0
	downstream_end = $5 + downstream_bp
} else if($7 == "-") {
upstream_start = ($4 > downstream_bp) ? $4 - downstream_bp -1 : 0
downstream_end = $4 + upstream_bp
} 
match($9, /Name=([^;]+)/, name_array)
    if (name_array[1] != "") {
        print $1, upstream_start, downstream_end, name_array[1], ".", $7
    }
}' "${GFF}" > "${FOLDER}/tmp_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

awk 'BEGIN{OFS="\t"} ($3 == "chromosome") {chrom_lengths[$1] = $2} 
NR==FNR{chrom_lengths[$1]=$2; next} 
($1 in chrom_lengths) {if ($3 > chrom_lengths[$1]) $3 = chrom_lengths[$1]; print}'\
	"${FOLDER}/${NAME}_chromosome_lengths.txt" \
	"${FOLDER}/tmp_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed" > \
	"${FOLDER}/custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

rm "${FOLDER}/tmp_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

echo "Conversion completed. Output saved to ${FOLDER}/custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

bedtools getfasta -fi ${FASTA}\
	-bed ${FOLDER}/custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed\
	-name -fo ${FOLDER}/promoters_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.fasta -s # Absolutely crucial for position sensitive analysis.

if [ "$INTRONS" == "true" ]; then
    echo "Very annoying, TODO"
fi

echo "Fasta file saved to ${FOLDER}/promoters_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.fasta, good to go!"
