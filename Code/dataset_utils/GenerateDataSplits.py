import sys

sys.path.append(".")
from Code.DNA_processing import get_representation
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import os
import numpy as np
import subprocess
from Code.DNA_processing.get_gene_families import add_all_genes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from Code.DAPseq_processing.generateDAP_seq import generate_peaks
import pickle
import statsmodels.api as sm


def family_wise_train_test_splitting(
    gene_families: np.array, test_size=0.2, random_state=42
):
    """
    Split the data into training and testing sets, ensuring that the gene families are not split between the two sets.

    gene_families: The gene families
    test_size: The proportion of the data to be used as the test set
    random_state: The random state for the shuffle split
    ---

    returns: Training and test indices

    """
    if test_size == 0:
        print("No splitting is done.")
        return np.arange(len(gene_families)), np.arange(len(gene_families))

    X = np.zeros(shape=(len(gene_families), 1))
    Y = np.zeros(shape=(len(gene_families), 1))

    # Create the split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Split the data
    train_index, test_index = next(gss.split(X, Y, groups=gene_families))

    # assert no family is present in both sets
    assert (
        len(
            set(gene_families[train_index]).intersection(set(gene_families[test_index]))
        )
        == 0
    )

    # Return the split data indices
    return train_index, test_index


class DataHandler:
    """
    This is the main class to hanlde the data generation and storage. Gets the
    desired representation of data and performs all the required operations.

    It constructs dictionaries for the data and dataframes with the dictionary keys for training, validation and testing.

    Parameters:
    DNA_specification: list
        [<upstream_bp_TSS> <downstream_bp_TSS> <upstream_bp_TTS> <downstream_bp_TTS>]
        Being all integers.
    gene_families_file: str
        The file containing the gene families.
    data_path: str
        The path to the data.
    train_proportion: float
        The proportion of the data to be used for training. Default is 0.8.
    validation_proportion: float
        The proportion of the data to be used for validation. Default is 0.1.
    random_state: int
        The random state for the shuffle split. Default is 42.
    dna_format: str
        The format of the DNA sequences. Default is "One_Hot_Encoding". Options: "One_Hot_Encoding", "String", "6-mer", "DAPseq".
    dna_folder: str
        The folder containing the DNA sequences. Default is "Data/RAW/DNA/Ath".
    mRNA_file: str
        The file containing the mRNA data. Default is "Data/Processed/mRNA/DESeq2_padj_results_ALL.csv".
    mask_exons: bool
        Whether to mask exons in the DNA sequences. Default is False.
    kmer_rc: bool
        Whether to use reverse complement for k-mers. Default is False.

    Attributes:
    TSS_dna: dict
        The dictionary of TSS DNA sequences.
    TTS_dna: dict
        The dictionary of TTS DNA sequences.
    gene_names: list
        The list of gene names.
    mRNA: pd.DataFrame
        The mRNA data, contains references to the.
    """

    def __init__(
        self,
        DNA_specification: list,
        gene_families_file: str,
        data_path: str,
        train_proportion: float = 0.8,
        validation_proportion: float = 0.1,
        random_state=42,
        dna_format="One_Hot_Encoding",
        dna_folder="Data/RAW/DNA/Ath",
        mRNA_file="Data/Processed/mRNA/DESeq2_padj_results_ALL.csv",
        mask_exons=False,
        kmer_rc=False,
    ):

        self.gene_families_file = gene_families_file
        self.data_path = data_path
        self.train_proportion = train_proportion
        self.validation_proportion = validation_proportion
        self.random_state = random_state
        self.dna_format = dna_format
        assert self.dna_format in ["One_Hot_Encoding", "String", "6-mer", "DAPseq"]
        self.mRNA_file = mRNA_file
        self.kmer_rc = kmer_rc
        self.metadata = {}
        self.mapping = {
            "B": "MeJA",
            "C": "SA",
            "D": "SA+MeJA",
            "G": "ABA",
            "X": "3-OH10",
            "Y": "chitooct",
            "Z": "elf18",
            "W": "flg22",
            "V": "nlp20",
            "U": "OGs",
            "T": "Pep1",
        }
        # Get fasta files:
        TSS_dna_url = f"{dna_folder}/promoters_{DNA_specification[0]}up_{DNA_specification[1]-1}down_TSS.fasta"
        TTS_dna_url = f"{dna_folder}/promoters_{DNA_specification[2]-1}up_{DNA_specification[3]}down_TTS.fasta"

        # read previous run dna format
        if os.path.exists(f"{dna_folder}/previous_run_DNA_format.txt"):
            with open(f"{dna_folder}/previous_run_DNA_format.txt", "r") as f:
                previous_dna_format = f.read()
        else:
            previous_dna_format = None

        if (
            os.path.exists(TSS_dna_url)
            and os.path.exists(TTS_dna_url)
            and previous_dna_format
            == self.dna_format
            + "_"
            + str(mask_exons)
            + "_"
            + dna_folder
            + "_"
            + str(self.kmer_rc)
        ):
            # Check if the above files exist
            print("Requested DNA files already available.")

            if (
                self.dna_format == "One_Hot_Encoding"
                and "One_Hot_Encoding" in previous_dna_format
            ):

                # load sequences from the .npy files in the data path
                TSS_files = os.listdir(f"{data_path}/DNA/One_Hot_Encoding/TSS")
                TTS_files = os.listdir(f"{data_path}/DNA/One_Hot_Encoding/TTS")

                self.urls_dict_TSS = {
                    gene.split(".")[0]: f"{data_path}/DNA/One_Hot_Encoding/TSS/{gene}"
                    for gene in TSS_files
                }
                self.urls_dict_TTS = {
                    gene.split(".")[0]: f"{data_path}/DNA/One_Hot_Encoding/TTS/{gene}"
                    for gene in TTS_files
                }

                self.TSS_sequences = {
                    gene: np.load(url) for gene, url in self.urls_dict_TSS.items()
                }
                self.TTS_sequences = {
                    gene: np.load(url) for gene, url in self.urls_dict_TTS.items()
                }

            elif self.dna_format == "6-mer" and "6-mer" in previous_dna_format:

                TSS_files = os.listdir(f"{data_path}/DNA/6-mer/TSS")
                TTS_files = os.listdir(f"{data_path}/DNA/6-mer/TTS")

                self.urls_dict_TSS = {
                    gene.split(".")[0]: f"{data_path}/DNA/6-mer/TSS/{gene}"
                    for gene in TSS_files
                }
                self.urls_dict_TTS = {
                    gene.split(".")[0]: f"{data_path}/DNA/6-mer/TTS/{gene}"
                    for gene in TTS_files
                }

                self.TSS_sequences = {
                    gene: np.load(url) for gene, url in self.urls_dict_TSS.items()
                }
                self.TTS_sequences = {
                    gene: np.load(url) for gene, url in self.urls_dict_TTS.items()
                }
                # load the kmer index
                with open(f"{data_path}/DNA/6-mer/kmer_index_TSS.pkl", "rb") as f:
                    self.kmer_index_TSS = pickle.load(f)

                with open(f"{data_path}/DNA/6-mer/kmer_index_TTS.pkl", "rb") as f:
                    self.kmer_index_TTS = pickle.load(f)

                self.metadata["kmer_dict_TSS"] = self.kmer_index_TSS
                self.metadata["kmer_dict_TTS"] = self.kmer_index_TTS

            elif self.dna_format == "DAPseq" and "DAPseq" in previous_dna_format:
                # load the DAP-seq data
                DAPseq_TSS = pd.read_csv(
                    "Data/Processed/DAPseq/genes_TSS.csv",
                    sep=",",
                    header=0,
                    index_col=0,
                )
                DAPseq_TTS = pd.read_csv(
                    "Data/Processed/DAPseq/genes_TTS.csv",
                    sep=",",
                    header=0,
                    index_col=0,
                )
                self.DAPseq_TSS_dict = {
                    gene: DAPseq_TSS.loc[gene].values for gene in DAPseq_TSS.index
                }
                self.DAPseq_TTS_dict = {
                    gene: DAPseq_TTS.loc[gene].values for gene in DAPseq_TTS.index
                }

                self.metadata["TFs_TSS"] = list(DAPseq_TSS.columns)
                self.metadata["TFs_TTS"] = list(DAPseq_TTS.columns)

            elif self.dna_format == "String" and "String" in previous_dna_format:

                TSS_dna = get_representation.load_promoters(TSS_dna_url)
                TTS_dna = get_representation.load_promoters(TTS_dna_url)

                self.TSS_sequences = TSS_dna
                self.TTS_sequences = TTS_dna

        else:
            cmd = [
                "Code/DNA_processing/parse_dna_all.sh",  # Accepts a folder with a .fasta and a .gff file
                str(DNA_specification[0]),
                str(DNA_specification[1]),
                str(DNA_specification[2]),
                str(DNA_specification[3]),
                dna_folder,
                str(mask_exons),
            ]

            subprocess.run(cmd, check=True)

            if self.dna_format == "DAPseq":

                print("DAPseq data requested. Calling maker of peaks")
                # create a dap_seq directory
                os.makedirs(f"{data_path}/DAPseq", exist_ok=True)
                generate_peaks(
                    f"{dna_folder}/custom_promoter_coordinates_{DNA_specification[0]}up_{DNA_specification[1]-1}down_TSS.bed",
                    f"{dna_folder}/custom_promoter_coordinates_{DNA_specification[2]-1}up_{DNA_specification[3]}down_TTS.bed",
                    "Data/RAW/DAPseq/dap_data_v4/peaks",
                    "Data/RAW/DAPseq/dap_overlap",
                    "Data/Processed/DAPseq",
                    cutoff_percentile=0.25,  # as per Niels et al. (2024).
                )

                DAPseq_TSS = pd.read_csv(
                    "Data/Processed/DAPseq/genes_TSS.csv",
                    sep=",",
                    header=0,
                    index_col=0,
                )
                DAPseq_TTS = pd.read_csv(
                    "Data/Processed/DAPseq/genes_TTS.csv",
                    sep=",",
                    header=0,
                    index_col=0,
                )
                # assert that the columns are the same (first remove the _TSS and _TTS)
                colanmes_TSS = [col.split("_")[0] for col in DAPseq_TSS.columns]
                colanmes_TTS = [col.split("_")[0] for col in DAPseq_TTS.columns]
                assert all(
                    [col1 == col2 for col1, col2 in zip(colanmes_TSS, colanmes_TTS)]
                ), "Columns of TSS and TTS do not match. Please check the DAP-seq data."

                # now, transform the dataset into a dictionary using index as keys and the values will be a np.array
                # with the values of the DAP-seq data
                self.DAPseq_TSS_dict = {
                    gene: DAPseq_TSS.loc[gene].values for gene in DAPseq_TSS.index
                }
                self.DAPseq_TTS_dict = {
                    gene: DAPseq_TTS.loc[gene].values for gene in DAPseq_TTS.index
                }

                self.metadata["TFs_TSS"] = list(DAPseq_TSS.columns)
                self.metadata["TFs_TTS"] = list(DAPseq_TTS.columns)

            else:
                # Now, get the dictionaries with the raw sequences
                TSS_dna = get_representation.load_promoters(TSS_dna_url)
                TTS_dna = get_representation.load_promoters(TTS_dna_url)
                if self.dna_format == "One_Hot_Encoding":
                    # Get DNA representation
                    self.urls_dict_TSS, self.urls_dict_TTS = (
                        self.generate_onehot_representation_DNA(TSS_dna, TTS_dna)
                    )

                    self.TSS_sequences = {
                        gene: np.load(url) for gene, url in self.urls_dict_TSS.items()
                    }
                    self.TTS_sequences = {
                        gene: np.load(url) for gene, url in self.urls_dict_TTS.items()
                    }

                elif self.dna_format == "String":
                    self.TSS_sequences = TSS_dna
                    self.TTS_sequences = TTS_dna

                elif self.dna_format == "6-mer":
                    (
                        self.urls_dict_TSS,
                        self.urls_dict_TTS,
                        self.kmer_index_TSS,
                        self.kmer_index_TTS,
                    ) = self.generate_kmer_representation_DNA(TSS_dna, TTS_dna, k=6)

                    self.TSS_sequences = {
                        gene: np.load(url) for gene, url in self.urls_dict_TSS.items()
                    }
                    self.TTS_sequences = {
                        gene: np.load(url) for gene, url in self.urls_dict_TTS.items()
                    }

                    # assert that the indices are in the same order
                    assert all(
                        [
                            gene1 == gene2
                            for gene1, gene2 in zip(
                                sorted(self.kmer_index_TSS.keys()),
                                sorted(self.kmer_index_TTS.keys()),
                            )
                        ]
                    ), "The k-mer indices are not in the same order. Please check the DAP-seq data."

                    # save the kmer index in a pickle file
                    os.makedirs(f"{data_path}/DNA/6-mer", exist_ok=True)
                    with open(f"{data_path}/DNA/6-mer/kmer_index_TSS.pkl", "wb") as f:
                        pickle.dump(self.kmer_index_TSS, f)

                    with open(f"{data_path}/DNA/6-mer/kmer_index_TTS.pkl", "wb") as f:
                        pickle.dump(self.kmer_index_TTS, f)

                    self.metadata["kmer_dict_TSS"] = self.kmer_index_TSS
                    self.metadata["kmer_dict_TTS"] = self.kmer_index_TTS

        with open(f"{dna_folder}/previous_run_DNA_format.txt", "w") as f:
            f.write(
                self.dna_format
                + "_"
                + str(mask_exons)
                + "_"
                + dna_folder
                + "_"
                + str(self.kmer_rc)
            )

    def generate_onehot_representation_DNA(self, TSS_dna, TTS_dna):
        """
        Generate the desired representation of the DNA sequences. Populates the data_path with the
        desired representation. Returns the urls of the generated data.

        Parameters:
        TSS_dna: dict
            The dictionary of TSS DNA sequences.
        TTS_dna: dict
            The dictionary of TTS DNA sequences.
        intron_dna: dict (optional)
            The dictionary of intron DNA sequences.

        Returns:
        urls_dict_TSS: dict
            The dictionary of urls for the TSS DNA sequences.
        urls_dict_TTS: dict
            The dictionary of urls for the TTS DNA sequences.
        urls_dict_intron: dict
            The dictionary of urls for the intron DNA sequences.
        """

        # TTS

        os.makedirs(self.data_path + "/DNA/One_Hot_Encoding/TSS", exist_ok=True)

        urls_dict_TSS = get_representation.generate_one_hot_representation(
            promoters=TSS_dna,
            folder=self.data_path + "/DNA/One_Hot_Encoding/TSS",
        )

        # TTS
        os.makedirs(self.data_path + "/DNA/One_Hot_Encoding/TTS", exist_ok=True)
        urls_dict_TTS = get_representation.generate_one_hot_representation(
            promoters=TTS_dna,
            folder=self.data_path + "/DNA/One_Hot_Encoding/TTS",
        )

        return urls_dict_TSS, urls_dict_TTS

    def generate_kmer_representation_DNA(self, TSS_dna, TTS_dna, k=6):

        os.makedirs(self.data_path + f"/DNA/{k}-mer/TSS", exist_ok=True)
        os.makedirs(self.data_path + f"/DNA/{k}-mer/TTS", exist_ok=True)

        print("Generating k-mer representation of the DNA sequences.")
        print(f"Reverse complement: {self.kmer_rc}")
        urls_dict_TSS, kmer_index_TSS = get_representation.generate_kmer_count_vector(
            promoters=TSS_dna,
            folder=self.data_path + f"/DNA/{k}-mer/TSS",
            k=k,
            reverse_complement=self.kmer_rc,
        )

        urls_dict_TTS, kmer_index_TTS = get_representation.generate_kmer_count_vector(
            promoters=TTS_dna,
            folder=self.data_path + f"/DNA/{k}-mer/TTS",
            k=k,
            reverse_complement=self.kmer_rc,
        )

        return urls_dict_TSS, urls_dict_TTS, kmer_index_TSS, kmer_index_TTS

    def get_splits(self):
        """
        Handles the missing genes in the gene families file. Then
        Provides a family wise split of the data.

        Parameters:
        gene_names: list
            The list of gene names.

        Returns:
        train_index: list
            The training indices.

        validation_index: list
            The validation indices.

        test_index: list
            The testing indices.
        """

        self.gene_names = self.mRNA["Gene"].values
        if self.train_proportion != 1:
            add_all_genes(
                self.gene_families_file, self.gene_names
            )  # add genes that are not in gene families as individual families
            gene_families = pd.read_csv(self.gene_families_file)
            gene_families = gene_families.rename(columns={"gene_id": "Gene"})

            self.mRNA = pd.merge(self.mRNA, gene_families, on="Gene")
        else:  # do not do anything if there is no splitting going on
            gene_families = pd.DataFrame(
                {"Gene": self.gene_names, "family_id": np.arange(len(self.gene_names))}
            )
            self.mRNA = pd.merge(self.mRNA, gene_families, on="Gene")

        # Get the splits
        self.family_ids = self.mRNA["family_id"].values

        # Split the data
        train_index, test_index = family_wise_train_test_splitting(
            self.family_ids,
            test_size=1 - self.train_proportion,
            random_state=self.random_state,
        )
        if self.validation_proportion == 0 and self.train_proportion != 1:

            assert (
                len(
                    set(self.family_ids[train_index]).intersection(
                        set(self.family_ids[test_index])
                    )
                )
                == 0
            )

            return train_index, None, test_index

        elif self.validation_proportion == 0 and self.train_proportion == 1:
            return train_index, None, None

        train_i_index, validation_i_index = family_wise_train_test_splitting(
            self.family_ids[train_index],
            test_size=self.validation_proportion,
            random_state=self.random_state,
        )

        validation_index = train_index[validation_i_index]
        train_index = train_index[train_i_index]

        # assert that there is no genes in the same family in both sets
        assert (
            len(
                set(self.family_ids[train_index]).intersection(
                    set(self.family_ids[validation_index])
                )
            )
            == 0
        )
        assert (
            len(
                set(self.family_ids[train_index]).intersection(
                    set(self.family_ids[test_index])
                )
            )
            == 0
        )
        assert (
            len(
                set(self.family_ids[validation_index]).intersection(
                    set(self.family_ids[test_index])
                )
            )
            == 0
        )

        return train_index, validation_index, test_index

    def _get_class(self, treatment, problem_type):
        mRNA_data_matrix = pd.read_csv(self.mRNA_file)
        mRNA_data_matrix = mRNA_data_matrix.loc[
            mRNA_data_matrix["treatment"].isin(treatment)
        ]

        # remove genes that have NaN in the pvalue
        print("Number of gene instances in the dataset: ", len(mRNA_data_matrix))
        print(
            "Number of gene instances with NA in pvalue: ",
            sum(mRNA_data_matrix["padj"].isna()),
        )
        mRNA_data_matrix = mRNA_data_matrix.dropna(subset=["padj"])
        print(
            "Number of gene instances after removing NA in pvalue: ",
            len(mRNA_data_matrix),
        )

        if problem_type == "log2FC":
            mRNA_data_matrix = mRNA_data_matrix.dropna(subset=["padj"])
            wide_lgFC = mRNA_data_matrix.pivot(
                index="gene", columns="treatment", values="Log2_Fold_Change"
            )
            wide_lgFC = wide_lgFC.dropna()
            wide_lgFC = wide_lgFC.reset_index()
            wide_lgFC = wide_lgFC.rename(columns={"gene": "Gene"})

            self.metadata["n_labels"] = len(treatment)
            return wide_lgFC

        if problem_type == "amplitude":
            mRNA_data_matrix = mRNA_data_matrix.dropna(subset=["padj"])
            wide_amp = mRNA_data_matrix.pivot(
                index="gene", columns="treatment", values="amplitude"
            )
            wide_amp = wide_amp.dropna()
            wide_amp = wide_amp.reset_index()
            wide_amp = wide_amp.rename(columns={"gene": "Gene"})
            print("Number of genes in the dataset: ", len(wide_amp))

            self.metadata["n_labels"] = len(treatment)
            return wide_amp

        elif problem_type == "DE_per_treatment":
            fdr = 0.01
            print("FDA threshold: ", fdr)
            mRNA_data_matrix["padj"] = mRNA_data_matrix["padj"].fillna(1)
            mRNA_data_matrix["class"] = 3
            for h in treatment:
                mRNA_data_matrix.loc[mRNA_data_matrix["treatment"] == h, "class"] = 0
                mRNA_data_matrix.loc[
                    (mRNA_data_matrix["treatment"] == h)
                    & (mRNA_data_matrix["padj"] < fdr),
                    "class",
                ] = 1
            wide_de = mRNA_data_matrix.pivot(
                index="gene", columns="treatment", values="class"
            )
            wide_de = wide_de.fillna(3)
            wide_de = wide_de.astype(int)
            wide_de = wide_de.reset_index()
            wide_de = wide_de.rename(columns={"gene": "Gene"})
            self.metadata["n_labels"] = len(treatment)
            return wide_de

        elif problem_type == "quantiles_per_treatment":

            mRNA_data_matrix["padj"] = mRNA_data_matrix["padj"].fillna(1)
            fdr = 0.01
            print("FDA threshold: ", fdr)
            for h in treatment:
                mRNA_data_matrix.loc[mRNA_data_matrix["treatment"] == h, "class"] = 3
                # get percentile
                q_25 = mRNA_data_matrix.loc[
                    mRNA_data_matrix["treatment"] == h, "stat"
                ].quantile(0.25)
                q_75 = mRNA_data_matrix.loc[
                    mRNA_data_matrix["treatment"] == h, "stat"
                ].quantile(0.75)
                print(f"Quantiles for {h}: {q_25}, {q_75}")
                mRNA_data_matrix.loc[
                    (mRNA_data_matrix["treatment"] == h)
                    & (mRNA_data_matrix["stat"] <= q_25),
                    "class",
                ] = 0
                mRNA_data_matrix.loc[
                    (mRNA_data_matrix["treatment"] == h)
                    & (mRNA_data_matrix["stat"] >= q_75)
                    & (mRNA_data_matrix["padj"] < fdr),
                    "class",
                ] = 1

            wide_de = mRNA_data_matrix.pivot(
                index="gene", columns="treatment", values="class"
            )

            wide_de = wide_de.fillna(3)
            wide_de = wide_de.astype(int)
            # print the number of DE genes per hormon
            # put gene as column
            wide_de = wide_de.reset_index()
            wide_de = wide_de.rename(columns={"gene": "Gene"})
            self.metadata["n_labels"] = len(
                treatment
            )  # bacause this is going to be elementwise sigmoid activation, there can be several 1s at the same time
            return wide_de

        elif problem_type == "TPM_cuartiles":
            Y = pd.read_csv("Data/Processed/Basal/up_down_q_tpm.csv")
            Y.columns = ["Gene", "up_down_q_TPM"]
            return Y



    def get_data(self, treatments: list, problem_type: str):
        """
        Get the data for training, validation and testing.

        hormone:
            List with the hormones to be used

        class_type:
            The type of classification to be done. Options: "log2FC", "DE_per_treatment", "quantiles_per_treatment", "LR"
        """

        self.mRNA = self._get_class(treatments, problem_type)
        # ENFORCE that the columns follow the order of the treatments
        # VERY IMPORTANT!
        if problem_type != "TPM_cuartiles" and problem_type != "TPM":
            self.mRNA = self.mRNA[["Gene"] + treatments]  #!!! IMPORTANT

        if self.dna_format == "DAPseq":
            # enforcing it TODO
            self.mRNA = self.mRNA[self.mRNA["Gene"].isin(self.DAPseq_TTS_dict.keys())]
            self.mRNA = self.mRNA[self.mRNA["Gene"].isin(self.DAPseq_TSS_dict.keys())]
            self.DAPseq_TTS_dict = {
                gene: self.DAPseq_TTS_dict[gene] for gene in self.mRNA["Gene"]
            }
            self.DAPseq_TSS_dict = {
                gene: self.DAPseq_TSS_dict[gene] for gene in self.mRNA["Gene"]
            }
        else:
            print("Removing genes not in the DNA sequences")
            n_genes_before = len(self.mRNA)
            self.mRNA = self.mRNA[self.mRNA["Gene"].isin(self.TSS_sequences.keys())]
            self.mRNA = self.mRNA[self.mRNA["Gene"].isin(self.TTS_sequences.keys())]
            n_genes_after = len(self.mRNA)
            print(
                f"Number of (pseudogenes and trans. element genes) removed at last step: {n_genes_before - n_genes_after}"
            )
            self.TSS_sequences = {
                gene: self.TSS_sequences[gene] for gene in self.mRNA["Gene"]
            }
            self.TTS_sequences = {
                gene: self.TTS_sequences[gene] for gene in self.mRNA["Gene"]
            }
            print("Number of genes in the mRNA data: ", len(self.mRNA))
            # TMP I do not know what is happening here
        self.train_index, self.validation_index, self.test_index = self.get_splits()
        mRNA_train = self.mRNA.iloc[self.train_index]

        if self.validation_index is not None:
            assert self.validation_proportion != 0
            mRNA_validation = self.mRNA.iloc[self.validation_index]
        else:
            assert self.validation_proportion == 0
            mRNA_validation = None

        mRNA_test = (
            self.mRNA.iloc[self.test_index] if self.test_index is not None else None
        )
        # assert that the genes in mRNA are in the self.TSS_sequences
        if self.dna_format == "DAPseq":
            assert all(
                [gene in self.DAPseq_TTS_dict.keys() for gene in mRNA_train["Gene"]]
            ), "probably using different references for mRNA and DNA! or something with DAP-seq data"
            if mRNA_validation is not None:
                assert all(
                    [
                        gene in self.DAPseq_TTS_dict.keys()
                        for gene in mRNA_validation["Gene"]
                    ]
                )
            assert all(
                [gene in self.DAPseq_TTS_dict.keys() for gene in mRNA_test["Gene"]]
            )

        else:
            assert all(
                [gene in self.TSS_sequences.keys() for gene in mRNA_train["Gene"]]
            ), "probably using different references for mRNA and DNA!"
            if mRNA_validation is not None:
                assert all(
                    [
                        gene in self.TSS_sequences.keys()
                        for gene in mRNA_validation["Gene"]
                    ]
                )
            if mRNA_test is not None:
                assert all(
                    [gene in self.TSS_sequences.keys() for gene in mRNA_test["Gene"]]
                )

        if mRNA_validation is not None:
            assert (
                len(
                    set(mRNA_train["family_id"]).intersection(
                        set(mRNA_validation["family_id"])
                    )
                )
                == 0
            )
            assert (
                len(
                    set(mRNA_validation["family_id"]).intersection(
                        set(mRNA_test["family_id"])
                    )
                )
                == 0
            )
        if mRNA_test is not None:
            assert (
                len(
                    set(mRNA_train["family_id"]).intersection(
                        set(mRNA_test["family_id"])
                    )
                )
                == 0
            )

        # assert that the order of the columns is the same as the treatments
        if problem_type != "TPM_cuartiles" and problem_type != "TPM":
            assert all(mRNA_train.columns[1:-1] == treatments)
            if mRNA_validation is not None:
                assert all(mRNA_validation.columns[1:-1] == treatments)
            if mRNA_test is not None:
                assert all(mRNA_test.columns[1:-1] == treatments)

        if self.dna_format == "DAPseq":
            return (
                mRNA_train,
                mRNA_validation,
                mRNA_test,
                self.DAPseq_TSS_dict,
                self.DAPseq_TTS_dict,
                self.metadata,
            )

        else:
            return (
                mRNA_train,
                mRNA_validation,
                mRNA_test,
                self.TSS_sequences,
                self.TTS_sequences,
                self.metadata,
            )


if __name__ == "__main__":
    # To test
    DNA_specification = [50, 50, 50, 15]
    gene_families_file = "Data/Processed/gene_families.csv"
    data_path = "Data/Processed"
    train_proportion = 0.85
    validation_proportion = 0.1
    dna_format = "6-mer"

    data_handler = DataHandler(
        DNA_specification,
        gene_families_file,
        data_path,
        train_proportion,
        validation_proportion,
        dna_format=dna_format,
        mask_exons=False,
    )

    mRNA_train, mRNA_validation, mRNA_test, TSS_sequences, TTS_sequences, metadata = (
        data_handler.get_data(
            treatments=["B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"],
            problem_type="log2FC",
        )
    )
    print(mRNA_train)
    print(mRNA_train.shape, mRNA_validation.shape, mRNA_test.shape)
    exit()
    # make the TSS_sequences a dataframe with the gene names as index
    TSS_sequences = pd.DataFrame.from_dict( TSS_sequences, orient="index")
    # put metahada TFs_TSS
    TSS_sequences.columns = metadata["kmer_dict_TSS"]
    TSS_sequences.index.name = "Gene"
    TSS_sequences = TSS_sequences.reset_index()
    # now TTS
    TTS_sequences = pd.DataFrame.from_dict( TTS_sequences, orient="index")
    # put metahada TFs_TTS
    TTS_sequences.columns = metadata["kmer_dict_TTS"]
    TTS_sequences.index.name = "Gene"
    TTS_sequences = TTS_sequences.reset_index()
    # save the TSS and TTS 
    TSS_sequences.to_csv(f"TSS_sequences_6mer_1500-500.csv", index=False)
    TTS_sequences.to_csv(f"TTS_sequences_6mer_1500-500.csv", index=False)