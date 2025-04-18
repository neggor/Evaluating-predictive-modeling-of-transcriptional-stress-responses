import sys
sys.path.append(".")
from Code.Training.train_cnn import train_cnn_model_from_specs, test_cnn
from Code.dataset_utils.GenerateDataSplits import DataHandler
from Code.CNN_model.res_CNN import myCNN
from Code.Training.train_linear import (
    handle_data_train_linear_models,
    handle_data_test_linear_models,
    fit_regression,
    test_linear,
)
from Code.Training.finetune_agroNT import (
    handle_data_AgroNT,
    finetune_agroNT,
)
import argparse
import json
import numpy as np
import torch
import os
import shutil

torch.manual_seed(19998)
np.random.seed(19998)

mapping = {
    "B": "MeJA",
    "C": "SA",
    "D": "SA+MeJA",
    "G": "ABA",
    "H": "ABA+MeJA",
    "X": "3-OH10",
    "Y": "chitooct",
    "Z": "elf18",
    "W": "flg22",
    "V": "nlp20",
    "U": "OGs",
    "T": "Pep1",
}

def load_data(
    train_proportion,
    val_proportion,
    treatments,
    problem_type,
    dna_format,
    DNA_specs,
    gene_families_file="Data/Processed/gene_families.csv",
    data_path="Data/Processed",
    mask_exons=False,
    kmer_rc=True,
):
    # reset data calling if files exist
    if os.path.exists("Data/RAW/DNA/Ath/previous_run_DNA_format.txt"):
        os.remove("Data/RAW/DNA/Ath/previous_run_DNA_format.txt")
    # remove all contents of Processed/DNA folder
    for filename in os.listdir("Data/Processed/DNA"):
        file_path = os.path.join("Data/Processed/DNA", filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Remove subdirectories

    """
    Loads the DNA and mRNA outcome.
    """
    data_handler = DataHandler(
        DNA_specs,
        gene_families_file,
        data_path,
        train_proportion,
        val_proportion,
        random_state=19998,
        mask_exons=mask_exons,
        dna_format=dna_format,
        kmer_rc=kmer_rc,
    )

    mRNA_train, mRNA_validation, mRNA_test, TSS_sequences, TTS_sequences, metadata = (
        data_handler.get_data(treatments, problem_type)
    )

    return (
        mRNA_train,
        mRNA_validation,
        mRNA_test,
        TSS_sequences,
        TTS_sequences,
        metadata,
    )

def get_cnn_performance(n_rep, store_folder, train_proportion, val_proportion, cnn_config: dict):
    
    dna_specs = [ cnn_config["upstream_TSS"], cnn_config["downstream_TSS"],
                  cnn_config["downstream_TTS"], cnn_config["upstream_TTS"]]
    
    mRNA_train, mRNA_validation, mRNA_test, TSS_sequences, TTS_sequences, metadata = load_data(
            train_proportion,
            val_proportion,
            treatments=cnn_config["treatments"],
            problem_type=cnn_config["problem_type"],
            dna_format=cnn_config["dna_format"],
            DNA_specs= dna_specs,
            mask_exons=cnn_config["mask_exons"],
            kmer_rc= False, # it does not apply to the CNN
        )

    for i in range(n_rep):
        i_store_folder = f"{store_folder}/model_{i}"
        model = myCNN(
            n_labels= cnn_config["n_labels"],
            n_ressidual_blocks= cnn_config["n_ressidual_blocks"],
            in_channels = cnn_config["in_channels"],
            out_channels= cnn_config["out_channels"],
            kernel_size= cnn_config["kernel_size"],
            max_pooling_kernel_size= cnn_config["max_pooling_kernel_size"],
            dropout_rate= cnn_config["dropout_rate"],
            stride= cnn_config["stride"],
            RC_equivariant= cnn_config["equivariant"],
        )

        # Train the model with early stopping
        # returns the best model
        best_model, my_metrics, my_metrics_val = train_cnn_model_from_specs(
            model,
            cnn_config,
            TSS_sequences=TSS_sequences,
            TTS_sequences=TTS_sequences,
            mRNA_train=mRNA_train,
            mRNA_validation=mRNA_validation,
            device=cnn_config["device"],
            store_folder=i_store_folder,
            name =cnn_config["model_name"],
        )

        best_model.load_state_dict(torch.load(f"{i_store_folder}/best_model.pth"))
        
        Y_hat, Y, m = test_cnn( best_model,
                                training_specs=cnn_config,
                                TSS_sequences=TSS_sequences,
                                TTS_sequences=TTS_sequences,
                                mRNA_test=mRNA_test,
                                treatments=cnn_config["treatments"],
                                device=cnn_config["device"],
                                store_folder=i_store_folder,
                                return_metrics=True)

def get_linear_performance(store_folder, train_proportion, val_proportion, linear_config: dict):
        dna_specs = [ linear_config["upstream_TSS"], linear_config["downstream_TSS"],
                  linear_config["downstream_TTS"], linear_config["upstream_TTS"]]
        (
            mRNA_train,
            mRNA_validation,
            mRNA_test,
            TSS_sequences,
            TTS_sequences,
            metadata,
        ) = load_data(
            train_proportion=train_proportion,
            val_proportion=val_proportion,
            DNA_specs=dna_specs,
            treatments=linear_config["treatments"],
            problem_type=linear_config["problem_type"],
            mask_exons=linear_config["mask_exons"], 
            dna_format=linear_config["dna_format"],
        )
        X, Y, mean, std, family = handle_data_train_linear_models(
            TSS_sequences,
            TTS_sequences,
            mRNA_train,
            mRNA_validation,
            DNA_format=linear_config["dna_format"],
            separated_segments=linear_config["separated_segments"],
        )
        X_test, Y_test = handle_data_test_linear_models(
            TSS_sequences, TTS_sequences, mRNA_test, DNA_format= linear_config["dna_format"], means=mean, std=std, separated_segments=linear_config["separated_segments"]
        )
        for i, tr in enumerate(linear_config["treatments"]):
            y = Y[:, i : i + 1]
            y_test = Y_test[:, i : i + 1]

            reg = fit_regression(
                X,
                y,
                family,
                linear_config["linear_model_kind"],
                store_folder,
                mapping[tr],
                metadata,
                DNA_format=linear_config["dna_format"],
                separated_segments=linear_config["separated_segments"],
            )

            test_linear(
                X_test,
                y_test,
                reg,
                linear_config["linear_model_kind"],
                name=mapping[tr],
                folder=store_folder,
                metadata=metadata,

        )

def get_AgroNT_performance(store_folder,  train_proportion, val_proportion, agroNT_config: dict):
        dna_specs = [ agroNT_config["upstream_TSS"], agroNT_config["downstream_TSS"],
                  agroNT_config["downstream_TTS"], agroNT_config["upstream_TTS"]]
        (
            mRNA_train,
            mRNA_validation,
            mRNA_test,
            TSS_sequences,
            TTS_sequences,
            metadata,
        ) = load_data(
            train_proportion=train_proportion,
            val_proportion=val_proportion,
            DNA_specs=dna_specs,
            treatments=agroNT_config["treatments"],
            problem_type=agroNT_config["problem_type"],
            mask_exons=agroNT_config["mask_exons"], 
            dna_format=agroNT_config["dna_format"],
        )
        n_tokens = (np.array(dna_specs[:2]).sum() // 6)
        tokenized_datasets_train_promoter, tokenized_datasets_validation_promoter, tokenized_datasets_test_promoter, tokenizer = handle_data_AgroNT(
            TSS_sequences,
            mRNA_train,
            mRNA_validation,
            mRNA_test,
            store_folder= store_folder,
            n_tokens= n_tokens
        )

        weights = None
        if agroNT_config['problem_type'] == "DE_per_treatment" or agroNT_config['problem_type'] == "quantiles_per_treatment":
            # get the proportions
            weights = []
            for i in mRNA_train.columns[1:-1]:
                mask = mRNA_train[i] != 3
                ps_clas = np.sum(mRNA_train[i][mask].values)
                print(f"Class {i} - Positive instances: {ps_clas}")
                ns_clas = mRNA_train[mask].shape[0] - ps_clas
                print(f"Class {i} - Negative instances: {ns_clas}")
                weight_i = ns_clas / ps_clas
                weights.append(weight_i)
            # concat in a vector
            weights = torch.tensor(weights).float()
            print("Weights:")
            print(weights)

        test_results = finetune_agroNT(
            tokenized_datasets_train_promoter,
            tokenized_datasets_validation_promoter,
            tokenized_datasets_test_promoter,
            tokenizer,
            config=agroNT_config,
            output_dir=store_folder,
            weights=weights,
        )
        

if __name__ == "__main__":
   # prepare an argparse totake model type to run and the config file. Also defaults for train proportion, test proportion, and number of replicates for the CNN

    parser = argparse.ArgumentParser(description="Evaluate performance of models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Type of model to run. Options are: CNN, L6mer, LDapseq, AgroNT",
    )

    parser.add_argument(
        "--train_proportion",
        type=float,
        required=True,
        help="Proportion of data to use for training (rest goes to test)",
    )

    parser.add_argument(
        "--val_proportion",
        type=float,
        required=True,
        help="Proportion of data to use for validation (as part of the training set)",
    )

    parser.add_argument(
        "--n_rep",
        type=int,
        default=1,
        help="Number of replicates to run for the CNN model",
    )

    parser.add_argument(
        "--store_folder",
        type=str,
        required=True,
        help="Folder to store the results",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file for the model",
    )

    args = parser.parse_args()
    # call the function to get the performance of the CNN
    # load the config file
    with open(args.config_file, "r") as f:
        config = json.load(f)
    
    #get_cnn_performance(args.n_rep, args.store_folder, args.train_proportion, args.val_proportion, cnn_config["problem_type"], cnn_config)
    if args.model == "CNN":
        get_cnn_performance(
            args.n_rep,
            args.store_folder,
            args.train_proportion,
            args.val_proportion,
            config,
        )

    elif args.model == "linear":
        get_linear_performance(
            args.store_folder,
            args.train_proportion,
            args.val_proportion,
            config,
        )
    elif args.model == "AgroNT":
        get_AgroNT_performance(
            args.store_folder,
            args.train_proportion,
            args.val_proportion,
            config,
        )
    else:
        raise ValueError("Model not recognized")
    
    # save the config file in the store folder
    with open(f"{args.store_folder}/config.json", "w") as f:
        json.dump(config, f)