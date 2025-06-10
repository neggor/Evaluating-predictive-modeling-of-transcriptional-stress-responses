from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CustomDataset(Dataset):
    def __init__(
        self,
        mRNA: pd.DataFrame,
        TSS_dna: dict,
        TTS_dna: dict,
    ):

        self.mRNA = mRNA  # This should be already the train/test/validation split
        self.TSS_dna = TSS_dna  # Dictionary with ALL the genes
        self.TTS_dna = TTS_dna

    def __len__(self):
        return self.mRNA.shape[0]

    def __getitem__(self, idx):
        gene_name = self.mRNA["Gene"].iloc[idx]  # Keys for DNA
        TSS = self.TSS_dna[gene_name]
        TTS = self.TTS_dna[gene_name]
        vectors = np.array(
            self.mRNA.iloc[idx, 1:-1].values
        )  # This 1:-1 because I remove Gene and family_id columns
        return {
            "TSS": TSS,
            "TTS": TTS,
            "values": vectors,
        }


def my_custom_collate(batch):
    # This has to handle an array genes x N_tfs
    batch_TSS = np.stack([sample["TSS"] for sample in batch])
    batch_TTS = np.stack([sample["TTS"] for sample in batch])
    batch_class = np.stack([sample["values"].astype(np.float32) for sample in batch])
    return {
        "TSS": batch_TSS,
        "TTS": batch_TTS,
        "values": batch_class,
    }


def get_dataloader(
    mRNA,
    TSS_dna,
    TTS_dna,
    batch_size=32,
    shuffle=True,
    return_dataset=False,
):

    _custom_collate_fn = my_custom_collate
    dataset = CustomDataset(mRNA, TSS_dna, TTS_dna)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_custom_collate_fn,
        drop_last=False,  # Kind of important!
    )

    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader
