#!/bin/bash

# Usage: ./gene_exon_length_gff.sh input.gff output.txt

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input.gff> <output_file>"
    exit 1
fi

gff_file="$1"
output_file="$2"

awk '
$3 == "exon" {
    # Extract Parent gene ID from attributes column
    match($9, /Parent=([^;]+)/, arr)
    parent_id = arr[1]
    
    # Strip isoform suffix like .1, .2, etc.
    sub(/\.[0-9]+$/, "", parent_id)
    
    exon_length = $5 - $4 + 1
    gene_lengths[parent_id] += exon_length
}

END {
    for (gene in gene_lengths) {
        print gene "\t" gene_lengths[gene]
    }
}
' "$gff_file" > "$output_file"

echo "Gene exon lengths aggregated over isoforms written to $output_file"