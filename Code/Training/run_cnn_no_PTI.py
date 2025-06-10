import sys
import os
sys.path.append(".")
import subprocess


def run_cnn_no_PTI():
    # load config file
    config_file = "Configs/cnn_no_PTI_config.json"
    store_folder = f"Results/CNN_no_PTI/"
    # create the store folder if it does not exist
    if not os.path.exists(store_folder):
        os.makedirs(store_folder)
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


if __name__ == "__main__":
    run_cnn_no_PTI()
    