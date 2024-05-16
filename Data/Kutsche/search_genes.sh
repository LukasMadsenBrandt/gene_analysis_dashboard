#!/bin/bash

# The file with gene names
gene_list="gene_names.txt"

# The file to search through
data_file="Kutsche_Counts.txt"

# The file to append results to
output_file="genes.txt"

# Check if the gene list and data files exist
if [ ! -f "$gene_list" ]; then
    echo "Error: Gene list file '$gene_list' does not exist."
    exit 1
fi

if [ ! -f "$data_file" ]; then
    echo "Error: Data file '$data_file' does not exist."
    exit 1
fi

# Add the first line of data_file to output_file
head -n 1 "$data_file" > "$output_file"

# Read each gene from the gene list file
while IFS= read -r gene; do
    if ! grep -P "\b$gene\b(?![\-])" "$data_file" >> "$output_file"; then
        echo "No matches found for gene: $gene"
    else
        echo "Matches found for gene: $gene and appended to $output_file"
    fi
done < "$gene_list"

echo "Search complete. Results appended to $output_file."
