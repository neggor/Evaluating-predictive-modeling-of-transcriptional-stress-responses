import numpy as np
import os
from tqdm import tqdm
from itertools import product


def load_promoters(promoter_file_url, exclude_non_ACGT=False):
    """
    Reads promoter region from a FASTA like file.
    Returns a dictionary of gene names and promoter sequences.

    Parameters:
    promoter_file_url (str): URL of the promoter file.
    exclude_non_ACGT (bool): If True, sequences with non ACGT letters are excluded.

    Returns:
    dict: Dictionary of gene names and promoter sequences.

    """
    # Create a dictionary of gene names and promoter sequences
    promoters = {}
    i = 0
    file = open(promoter_file_url, "r", encoding="utf-8")
    gene_name = None
    try:
        current_sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Line contains header (gene name)
                if current_sequence:
                    # If the current sequence is not empty,
                    # add it to the list of sequences
                    # Check letters different from A, C, G, T
                    if (
                        set(current_sequence).issubset(set(["A", "C", "G", "T"]))
                        is False
                    ):
                        i += 1
                        if exclude_non_ACGT:
                            current_sequence = ""
                            continue

                    promoters[gene_name] = current_sequence
                    current_sequence = ""
                    # Split by :: to get the gene name
                gene_name = line[1:].split("::")[0].strip()
            else:
                current_sequence += line
        # Add last entry
        promoters[gene_name] = current_sequence
    finally:
        file.close()
    print("Number of promoters with some non ACGT letters: ", i)
    print("Number of promoters: ", len(promoters))
    return promoters

def encode(sequence, max_length):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    one_hot = np.zeros((max_length, 4), dtype=np.int8)
    # iterate over the rows and select the column that belongs to that base
    for i, base in enumerate(sequence):
        if i >= max_length:
            break
        if base in mapping:
            one_hot[i, mapping[base]] = 1
        

    return one_hot

def generate_one_hot_representation(promoters: dict, folder: str):
    """
    Generates one hot representation of the promoters. Genes are unique identifiers.

    Parameters:

    promoters (dict): Dictionary of gene names and promoter sequences.
    folder (str): Folder to save the one hot representations.

    Returns:
    dict: Dictionary of gene names and URL of the one hot representations.
    """
    os.makedirs(folder, exist_ok=True)
    # print sequences length
    lengths = [len(promoters[gene]) for gene in promoters]
    print("Max length: ", max(lengths))
    max_length = max(lengths)
    # Even in the case of not being equal this would make all sequences the same length
    # Iterate over the sequences and concatenate them into the array
    urls = {}
    for i, gene in enumerate(promoters):
        # loop over every gene and store its one hot representation
        sequence = promoters[gene]
        encoded_sequence = encode(sequence, max_length)
        # assert shape is correcct
        assert encoded_sequence.shape[0] == max_length
        assert encoded_sequence.shape[1] == 4
        np.save(f"{folder}/{gene}.npy", encoded_sequence)
        urls[gene] = f"{folder}/{gene}.npy"
        # print(f"Saved {gene} in {folder}/{gene}.npy")

    return urls

def _make_kmer(promoters: dict, k: int):
    """
    Generates kmer representation of the promoters.

    Parameters:
    promoters (dict): Dictionary of gene names and promoter sequences.
    k (int): Length of the kmers.

    Returns:
    dict: Dictionary of gene names and kmer representation of the promoters.

    """
    # Create a dictionary of gene names and kmer representation of the promoters
    kmer_rep = {}

    for gene_name, promoter in promoters.items():
        kmer_rep[gene_name] = []
        for i in range(len(promoter) - k + 1):
            # skip if non ACGT letters are present
            if set(promoter[i : i + k]).issubset(set(["A", "C", "G", "T"])) is False:
                continue
            kmer_rep[gene_name].append(promoter[i : i + k])

    return kmer_rep

def _make_reverse_complement(dna_sequence: str):
    """
    Generates the reverse complement of a DNA sequence.

    Parameters:
    dna_sequence (str): DNA sequence.

    Returns:
    str: Reverse complement of the DNA sequence.

    """
    # Create a dictionary of complements
    complements = {"A": "T", "C": "G", "G": "C", "T": "A"}

    # Reverse the sequence
    dna_sequence = dna_sequence[::-1]

    # Generate the reverse complement
    reverse_complement = ""
    for letter in dna_sequence:
        try:
            reverse_complement += complements[letter]
        except KeyError:
            reverse_complement += letter

    return reverse_complement

def unique_kmers(k, use_rev_comp=True):
    kmers = set()
    for kmer_tuple in product("ACGT", repeat=k):
        kmer = "".join(kmer_tuple)
        if use_rev_comp:
            rev_comp = _make_reverse_complement(kmer)
            if rev_comp not in kmers:
                kmers.add(kmer)
        else:
            kmers.add(kmer)
    return sorted(kmers)  # Sorting ensures consistent order

def generate_kmer_count_vector(
    promoters: dict, folder, k = 6, reverse_complement=True
):
    """
    Generate normalized count vectors for each promoter sequence

    Parameters:
    promoters (dict): Dictionary of gene names and promoter sequences.
    k (int): Length of the kmers.
    folder (str): Folder to save the one hot representations.
    reverse_complement (bool): If True, the reverse complement of the kmers is also considered.

    Returns:
    dict: Dictionary of gene names and URL of the one hot representations.

    """
    # generate kmers:
    kmer_rep = _make_kmer(promoters, k)

    if reverse_complement is False:
        # generate a set for all possible kmers
        ordered_kmer_list = unique_kmers(k, use_rev_comp=False)
        print(f"Number of unique kmers: {len(ordered_kmer_list)}")
        # Generate a dictionary of kmers and their index
        kmer_index = {}
        i = 0
        for kmer in ordered_kmer_list:
            kmer_index[kmer] = i
            i += 1
        kmer_set = set(ordered_kmer_list)
        # For each promoter generate a count vector
        count_vectors = {}
        for gene_name, kmer_list in tqdm(kmer_rep.items()):
            # Initialize 0 vector
            count_vectors[gene_name] = np.zeros(len(kmer_set), dtype=np.float16)
            for kmer in kmer_list:
                # Populate the vector witht the counts
                # add 1 if the kmer is present
                count_vectors[gene_name][kmer_index[kmer]] += 1

        urls = {}
        for gene_name, count_vector in count_vectors.items():
            if np.sum(count_vector) == 0:
                print(f'{gene_name} has no counts')
            norm = np.linalg.norm(count_vector)
            if norm == 0:
                count_vectors[gene_name] = np.zeros(count_vector.shape)
            else:
                count_vectors[gene_name] = count_vector #/ norm

            np.save(f"{folder}/{gene_name}.npy", count_vectors[gene_name])
            urls[gene_name] = f"{folder}/{gene_name}.npy"

        return urls, kmer_index

    else:
        ordered_kmer_list = unique_kmers(k, use_rev_comp=True)
        print(f"Number of unique kmers: {len(ordered_kmer_list)}")
        # in the case of reverse complement there will not be exactly half, 
        # there are palindromic sequences that will be the same
        # Generate a dictionary of kmers and their index
        kmer_index = {}
        i = 0
        for kmer in ordered_kmer_list:
            kmer_index[kmer] = i
            i += 1
        # now transform to a set to speed up the search
        kmer_set = set(ordered_kmer_list)
        # For each promoter generate a count vector
        count_vectors = {}
        for gene_name, kmer_list in tqdm(kmer_rep.items()):
            # Initialize 0 vector
            count_vectors[gene_name] = np.zeros(len(kmer_set), dtype=np.float16)
            for kmer in kmer_list:
                # Populate the vector witht the counts
                # add 1 if the kmer is present
                if kmer in kmer_set:
                    count_vectors[gene_name][kmer_index[kmer]] += 1
                else:
                    count_vectors[gene_name][
                        kmer_index[_make_reverse_complement(kmer)]
                    ] += 1

        # Normalize the count vectors
        urls = {}
        for gene_name, count_vector in count_vectors.items():
            if np.sum(count_vector) == 0:
               print(f'{gene_name} has no counts')
            norm = np.linalg.norm(count_vector)
            if norm == 0:
                count_vectors[gene_name] = np.zeros(count_vector.shape)
            else:
                count_vectors[gene_name] = count_vector# / norm

            np.save(f"{folder}/{gene_name}.npy", count_vectors[gene_name])
            urls[gene_name] = f"{folder}/{gene_name}.npy"

        return urls, kmer_index
    
