# Define file paths
file1_path = '/mnt/data/genes_for_HP_0001249_intellectual_disability'
file2_path = '/mnt/data/genes_for_HP_0000252_microcephaly_phenotype'
output_file_path = '/mnt/data/unique_genes.txt'

# Set to store unique genes
unique_genes = set()

# Read the first file and extract the right-most gene name
with open(file1_path, 'r') as file1:
    for line in file1:
        gene = line.strip().split()[-1]  # Take the last element after splitting by whitespace
        if gene:
            unique_genes.add(gene)

# Read the second file and extract the right-most gene name
with open(file2_path, 'r') as file2:
    for line in file2:
        gene = line.strip().split()[-1]  # Take the last element after splitting by whitespace
        if gene:
            unique_genes.add(gene)

# Write the unique genes to the output file
with open(output_file_path, 'w') as output_file:
    for gene in sorted(unique_genes):
        output_file.write(gene + '\n')
