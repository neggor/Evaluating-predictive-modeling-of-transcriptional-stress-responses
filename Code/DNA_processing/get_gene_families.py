import subprocess
import os
import pandas as pd
import markov_clustering as mc
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import requests

# CODE ADAPTED FROM: https://www.pnas.org/doi/epub/10.1073/pnas.2319811121
# Running this creates a gene_families.csv file in the Data/Processed folder.
# This will be used to do the gene-wise splitting of the data.


def generate_gene_families():
    """
    It is used to create the gene families from the protein sequences of the Arabidopsis Thaliana proteome.
    It uses the blastp command to compare the sequences and then the markov clustering algorithm to create
    the families. The families are then saved in a csv file in the Data/Processed folder.

    To be used blastp must be isntalled in the system!
    """

    input_fasta_file = "Data/RAW/Protein/TAIR10_pep_20110103_representative_gene_model"

    # load gene names from fasta file
    with open(input_fasta_file, "r", encoding="utf-8") as f:
        gene_names = [
            line[1:].strip().split()[0]
            for line in f.readlines()
            if line.startswith(">")
        ]

    gene_names = [name.split(".")[0] for name in gene_names]
    if len(gene_names) != len(set(gene_names)):
        print("Gene names are not unique after removing isoform. Exiting.")

    # makeblast command
    mkblst_cmd = [
        "makeblastdb",
        "-in",
        input_fasta_file,
        "-dbtype",
        "prot",
        "-out",
        "Data/RAW/Protein_sequences/tair_db",
    ]
    blast_output_file = "Data/RAW/blast_output"

    # Define the blastp command with the provided arguments
    blastp_cmd = [
        "blastp",
        "-query",
        input_fasta_file,
        "-db",
        "Data/RAW/Protein_sequences/tair_db",
        "-out",
        blast_output_file,
        "-outfmt",
        "6 delim= \t qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
        "-evalue",
        "0.01",  # This removes quite a lot of pairs, to the point that some genes might not appear. It some
        # genes do not appear this is not a problem, we consider that they have no family, so we can put them anywhere
        "-num_threads",
        "15",
    ]
    # On this setting all evalues above 10 are discarded, that is why the result is not the number of genes squared

    # If the blast output file already exists, skip the blastp command
    if os.path.exists(blast_output_file):
        print("SKIPED BLAST, WAS ALREADY DONE")
    else:
        # Execute the blastp command using subprocess
        try:
            if not os.path.exists("Data/RAW/Protein_sequences/tair_db.psq"):
                print("Executing makeblastdb command...")
                subprocess.run(mkblst_cmd, check=True)
                print("makeblastdb command executed successfully.")
            print("Executing BLASTP command...")
            subprocess.run(blastp_cmd, check=True)
            print("BLASTP command executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error while executing BLASTP command:", e)

    blast_res = pd.read_csv("Data/RAW/blast_output", sep="\t")
    blast_res.columns = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
    ]
    # For the first two columns remove the isoform information
    blast_res["qseqid"] = blast_res["qseqid"].str.split(".").str[0]
    blast_res["sseqid"] = blast_res["sseqid"].str.split(".").str[0]
    # this makes duplicates!
    print(blast_res.head())
    print(blast_res.shape)
    # Drop duplicates A->B and B->A
    # Concatenate based qsqid and sseqid based on alphanumerical order
    # This works if indeed there is two copies of each pair.
    blast_res["sorted_ids"] = blast_res.apply(
        lambda row: tuple(sorted([row["qseqid"], row["sseqid"]])), axis=1
    )
    print(blast_res)
    blast_res = blast_res.drop_duplicates(subset="sorted_ids")
    blast_res = blast_res.drop(columns="sorted_ids")

    # Filter results by evalue < 0.01, bitscore > 40 and qsqid != sseqid
    # print("Number of pairs before filtering:", len(blast_res))
    blast_res = blast_res[
        (blast_res["evalue"] < 1e-5)
        & (blast_res["bitscore"] > 40)
        & (blast_res["qseqid"] != blast_res["sseqid"])  # this removes self-references
    ]

    G = nx.Graph()
    # Add edges with weights from DataFrame
    for _, row in blast_res.iterrows():
        G.add_edge(row["qseqid"], row["sseqid"], weight=row["bitscore"])
    print("Number of nodes:", len(G.nodes()))
    nodes = list(G.nodes())
    matrix = nx.to_scipy_sparse_array(G)
    get_family(matrix, 1.1, nodes)


def cluster_to_family(clusters, nodes):
    gene_families = []
    genes_added = set()
    for i, tup in enumerate(clusters):
        for item in tup:
            # tuple (aka family) i contains gene item
            if nodes[item] not in genes_added:
                genes_added.add(nodes[item])
            else:
                continue  # Basically, take the first family to get hard clusters
            # see https://github.com/GuyAllard/markov_clustering/issues/15#issuecomment-471309023
            gene_families.append([i, nodes[item]])

    gene_families = pd.DataFrame(gene_families, columns=["family_id", "gene_id"])

    # assert that gene_ids are unique, no gene belongs to (or more) two families
    assert gene_families["gene_id"].nunique() == len(
        gene_families
    ), "Genes belong to more than one family."
    gene_families.to_csv("Data/Processed/gene_families.csv", index=False)


def get_family(matrix, inflation, nodes):
    """
    Run MCL. Stores the result in "Data/Processed/gene_families.csv
    """
    print("Running clustering with inflation:", inflation)
    mcl_result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(mcl_result)  # returns a list of tuples,
    cluster_to_family(clusters, nodes)


def add_all_genes(gene_families_url: str, gene_names: list):
    """
    Take the gene names form the promoter data, and add them to the gene families. This is done to ensure that
    all genes are present in the gene families, even if they have no family. A gene added belongs to an exclusive
    family with only itself.
    """
    names_promoter_data = gene_names
    gene_families = pd.read_csv(gene_families_url)

    # Check which genes in names_promoter_data are not in gene_families
    # assert that gene families gene_ids are unique
    assert gene_families["gene_id"].nunique() == len(gene_families)
    genes_in_families = gene_families["gene_id"]
    genes_not_in_families = set(names_promoter_data) - set(genes_in_families)
    # Add genes not in families to gene_families, the gene family id starts adding form the last family id
    new_rows = [
        {"family_id": i + 1 + np.max(gene_families["family_id"]), "gene_id": gene}
        for i, gene in enumerate(genes_not_in_families)
    ]
    if len(new_rows) == 0:
        print("No new genes to add.")
        return
    new_rows = pd.DataFrame(new_rows)
    gene_families = pd.concat(
        [gene_families, new_rows], ignore_index=True
    )  # Add the rows.
    assert gene_families["gene_id"].nunique() == len(gene_families)
    # assert all genes in promoter are a subset of gene_families
    assert set(names_promoter_data) <= set(gene_families["gene_id"])
    # plot distribution of cluster size, except for the ones of size 1
    gene_families.to_csv(gene_families_url, index=False)
    print("All genes added to gene families. The gene families file has been updated!")


if __name__ == "__main__":
    generate_gene_families()
