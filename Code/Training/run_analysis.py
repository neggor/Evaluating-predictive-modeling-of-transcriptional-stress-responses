import sys
import os
sys.path.append(".")
import subprocess
import json
import pandas as pd

#outcome_types = ["log2FC", "amplitude", "quantiles_per_treatment", "DE_per_treatment", "TPM_cuartiles"]
outcome_types = ["TPM_cuartiles"]

# 1 Run linear models
def linear_models():

    # load config file
    config_file = "Configs/linear_config.json"
    # load
    with open(config_file, "r") as f:
        config = json.load(f)

    for problem_type in outcome_types:
        for dna_format in ["6-mer", "DAPseq"]:
            config["problem_type"] = problem_type
            config["dna_format"] = dna_format
            config["model_name"] = f"linear_{problem_type}_{dna_format}"
            config["linear_model_kind"] = (
                "lasso" if problem_type in ["log2FC", "amplitude", "TPM"] else "logistic_l1"
            )
            if "TPM" in problem_type:
                config["treatments"] = ["up_down_q_TPM"] if problem_type == "TPM_cuartiles" else ["mean"]
            else:
                config["treatments"] =  ["B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"]
            config["n_labels"] = len(config["treatments"])
            store_folder = f"Results/linear_models/{problem_type}/{dna_format}"
            # create the store folder if it does not exist
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            # save the config file
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)

            command = [
                "python",
                "Code/Training/get_performance.py",
                "--model",
                "linear",
                "--train_proportion",
                "0.8",
                "--val_proportion",
                "0.1",
                "--store_folder",
                store_folder,
                "--config_file",
                config_file,
            ]

            # run the command
            process = subprocess.run(command)


# 2 Run CNNs
# 3 Run CNNs with different length
# 4 Run CNNs with masked exons


def run_cnn():
    # load config file
    config_file = "Configs/cnn_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    for problem_type in outcome_types:
        if "TPM" in problem_type:
                config["treatments"] = ["up_down_q_TPM"] if problem_type == "TPM_cuartiles" else ["mean"]
        else:
            config["treatments"] =  [ "B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"]
        config["n_labels"] = len(config["treatments"])
        for dna_length in [2048, 4096, 8192]:
            for exons_masked in [True, False]:
                if exons_masked and dna_length != 2048:
                    continue
                print(
                    f"Running CNN for {problem_type} with dna_length {dna_length} and exons_masked {exons_masked}"
                )
                if dna_length == 2048:
                    config["upstream_TSS"] = 814
                    config["upstream_TTS"] = 200
                    config["downstream_TSS"] = 200
                    config["downstream_TTS"] = 814

                elif dna_length == 4096:
                    config["upstream_TSS"] = 1019
                    config["upstream_TTS"] = 1019
                    config["downstream_TSS"] = 1019
                    config["downstream_TTS"] = 1019

                elif dna_length == 8192:
                    config["upstream_TSS"] = 1024*2
                    config["upstream_TTS"] = 1024*2
                    config["downstream_TSS"] = 1019*2
                    config["downstream_TTS"] = 1019*2
                else:
                    raise ValueError(f"Unsupported dna_length: {dna_length}")
                config["mask_exons"] = exons_masked
                config["problem_type"] = problem_type

                store_folder = f"Results/CNN/{problem_type}/{dna_length}/exons_masked_{exons_masked}"
                # create the store folder if it does not exist
                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)
                # save the config file
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=4)

                command = [
                    "python",
                    "Code/Training/get_performance.py",
                    "--n_rep",
                    "5",
                    "--model",
                    "CNN",
                    "--train_proportion",
                    "0.8",
                    "--val_proportion",
                    "0.1",
                    "--store_folder",
                    store_folder,
                    "--config_file",
                    config_file,
                ]
                # run the command
                process = subprocess.run(command)


# 5 Run AgroNT


def run_agroNT():
    # load config file
    config_file = "Configs/agroNT_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    for problem_type in outcome_types:
        if "TPM" in problem_type:
            continue
        config["problem_type"] = problem_type
        store_folder = f"Results/agroNT/{problem_type}"
        # create the store folder if it does not exist
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        # save the config file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

        command = [
            "python",
            "Code/Training/get_performance.py",
            "--model",
            "AgroNT",
            "--train_proportion",
            "0.8",
            "--val_proportion",
            "0.1",
            "--store_folder",
            store_folder,
            "--config_file",
            config_file,
        ]
        # run the command
        process = subprocess.run(command)


def run_RF():
    # load config file
    config_file = "Configs/rf_config.json"
    # load
    with open(config_file, "r") as f:
        config = json.load(f)

    for problem_type in outcome_types:
        for dna_format in ["6-mer", "DAPseq"]:
            config["problem_type"] = problem_type
            config["dna_format"] = dna_format
            config["model_name"] = f"linear_{problem_type}_{dna_format}"
            config["linear_model_kind"] = (
                "regression" if problem_type in ["log2FC", "amplitude"] else "classification"
            )
            store_folder = f"Results/RF/{problem_type}/{dna_format}"
            # create the store folder if it does not exist
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            # save the config file
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)

            command = [
                "python",
                "Code/Training/get_performance.py",
                "--model",
                "RF",
                "--train_proportion",
                "0.8",
                "--val_proportion",
                "0.1",
                "--store_folder",
                store_folder,
                "--config_file",
                config_file,
            ]

            # run the command
            process = subprocess.run(command) 
# Gather all the data and construct plots


if __name__ == "__main__":
    linear_models()
    #run_cnn()
    #run_agroNT()
    #run_RF()