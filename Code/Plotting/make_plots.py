import sys
sys.path.append(".")
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from statannotations.Annotator import Annotator
import numpy as np
import os
import matplotlib.patches as mpatches
from Code.Training.get_performance import load_data
from Code.Training.train_cnn import test_cnn
import torch
import statsmodels as sm
from scipy.stats import gaussian_kde
from scipy import stats
from Code.CNN_model.res_CNN import myCNN
import json
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
from tqdm import tqdm
from matplotlib.lines import Line2D
import requests

mapping = {
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

magentaorgange_palette = ["#D35FB7", "#FF8C00"]
coraldarkteal_palette = ["#FF7F50", "#2CA58D"]

def get_tf_consensus_from_jaspar(tf_name, species="Arabidopsis thaliana"):
    """
    Get the consensus motif for a given TF name using JASPAR API.
    
    Parameters:
        tf_name (str): Name of the transcription factor (e.g., "WRKY70").
        species (str): Optional, species name to filter (default is Arabidopsis).
    
    Returns:
        str: Consensus motif (A/C/G/T string).
    """
    # Step 1: Search JASPAR for TF
    if tf_name == "GT3a":
        tf_name = "GT-3a" # Why dude?
    if tf_name == "ATAF1":
        tf_name = "NAC002"
    if tf_name == "AREB3":
        tf_name = "DPBF3"

    # the issue with the aliases makes that this function is not tottally general.
    # should be enough for replication, however.
    query_url = "https://jaspar.elixir.no/api/v1/matrix/"
    params = {
        "search": tf_name,
        "tax_group": "plants",
        "page_size": 1,
        "species": species,
    }
    print(f"Querying JASPAR for TF: {tf_name} in species: {species}")
    response = requests.get(query_url, params=params)
    response.raise_for_status()
    results = response.json()["results"]
    
    if not results:
        print (f"No results found for TF: {tf_name} this is likely due to a missing alias in JASPAR.")
        return None
    else:
        matrix_id = results[0]["matrix_id"]

    # Step 2: Get the full matrix including PFM
    matrix_url = f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}/"
    r = requests.get(matrix_url)
    r.raise_for_status()
    pfm = r.json()["pfm"]

    # Step 3: Convert PFM to consensus using argmax
    bases = ['A', 'C', 'G', 'T']
    matrix = np.array([pfm[base] for base in bases])
    # remove columns if entropy is too high
    entropy = stats.entropy(matrix, base=2, axis=0) # automatically normalizes 
    # get first and last columns with entropy < 1.5
    low_entropy_indices = np.where(entropy < 1.5)[0]
    # ge the between the first and last low entropy indices
    if len(low_entropy_indices) == 0:
        print(f"No low entropy indices found for TF: {tf_name}")
        return None
    first_low_entropy = low_entropy_indices[0]
    last_low_entropy = low_entropy_indices[-1]
    matrix = matrix[:, first_low_entropy:last_low_entropy + 1]
    consensus_indices = np.argmax(matrix, axis=0)
    consensus = ''.join([bases[i] for i in consensus_indices])
    # return up to 6 bases
    consensus = consensus[:8]  # Limit to first 6 bases cuz of the 6-mer

    return consensus

def set_plot_style():
    # Set Seaborn style + Matplotlib overrides
    sns.set_style("whitegrid")  # Background style (choose one)
    sns.set_context(
        "paper"
    )  # Scale fonts for paper ("paper", "notebook", "talk", "poster")

    # Override Matplotlib defaults for consistency
    mpl.rcParams.update(
        {
            #'font.family': 'serif',          # Use serif (e.g., Times New Roman)
            #'font.serif': ['Times New Roman'],
            "font.size": 18,  # Base font
            "axes.titlesize": 14,  # Title size
            "axes.labelsize": 14,  # Axis labels
            "xtick.labelsize": 14,  # Tick labels
            "ytick.labelsize": 14,
            "xtick.color": "#555555",  # X-tick color
            "ytick.color": "#555555",  # Y-tick color
            "legend.fontsize": 14,  # Legend font size
            "legend.title_fontsize": 16,  # Legend title font size
            "figure.dpi": 300,  # High-res figures
            "savefig.dpi": 300,
            "savefig.format": "pdf",  # Vector format
            "savefig.bbox": "tight",
            "axes.spines.top": False,  # Hide top spine
            "axes.spines.right": False,  # Hide right spine
            "axes.spines.left": True,  # Show left spine
            "axes.spines.bottom": True,  # Show bottom spine
            "axes.linewidth": 1.5,  # Spine width
            "axes.edgecolor": "black",  # Spine color
        }
    )

    sns.set_palette("deep")  # Set default color palette
    sns.color_palette("viridis", as_cmap=True)

def figure_1a(figsize=(10, 7)):
    res = pd.read_csv("Data/Processed/mRNA/DESeq2_padj_results_ALL.csv")
    res = res[res["treatment"] == "T"]
    # drop if NA in pvalue
    res = res.dropna(subset=["padj"])
    # Now calculate the upper and lower quartiles for the stat column
    res["stat"] = res["stat"].astype(float)
    # make them stop at 100
    res["stat"] = res["stat"].clip(upper=51)
    quantiles = res["stat"].quantile([0.25, 0.75])
    # get the stat value corresponding to padj of 0.01
    stat_001 = res[res["padj"] < 0.01]["stat"].min()
    # Add the class color to specify the class of each bin, 1 if below 0.25q and 2 if above 0.75q, 3 if between 0.25q and 0.75q
    res["class"] = "Neutral"
    res.loc[res["stat"] < quantiles[0.25], "class"] = "Below Lower Quartile"
    res.loc[res["stat"] > quantiles[0.75], "class"] = "Above Upper Quartile"
    # Make histogram with the hlines for the quartiles and the stat_001, put colors on the bins of each class
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # Define custom colors for the classes with stronger shades
    colors = {
        "Neutral": "lightgrey",
        "Below Lower Quartile": "dimgray",
        "Above Upper Quartile": "dimgray",
    }
    # Plot the histogram with custom colors
    sns.histplot(res, x="stat", bins=200, hue="class", palette=colors, legend=False)
    # sns.move_legend(ax, "upper right", bbox_to_anchor=(1, 1), fontsize=12)
    ax.axvline(quantiles[0.25], color="r", linestyle="dashed", linewidth=2)
    # add label for the ax line
    ax.text(quantiles[0.25] + 1, 300, "0.25 quantile", rotation=90)
    ax.axvline(quantiles[0.75], color="r", linestyle="dashed", linewidth=2)
    ax.text(quantiles[0.75] + 1, 300, "0.75 quantile", rotation=90)
    ax.axvline(stat_001, color="g", linestyle="dashed", linewidth=2)
    ax.text(stat_001 + 1, 300, "padj = 0.01", rotation=90)
    # Now, in the 100 in the x-axis put >100
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(False)  # Disable grid

    # Make axis lines more visible
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", which="major", width=1.5, length=6)
    ax.tick_params(axis="both", which="minor", width=1.5, length=4)

    # remove the grid
    plt.xticks([0, 10, 20, 30, 40, 50], ["0", "10", "20", "30", "40", "50"])
    # limit x axis to LR <= 100
    plt.ylim(0, 500)
    plt.xlim(0, 50)
    plt.xlabel("Likelihood ratio")
    plt.ylabel("Frequency")
    # put legent on the right upper
    # plt.title("Likelihood ratio between including or excluding the treatment effect \n Pep1 treatment")
    # save
    plt.savefig("Images/figure_1a.pdf", bbox_inches="tight")

def figure_2a(figsize=(10, 7), pvals=True, metric="AUC"):
    """
    Plot the figure 2a
    """
    res = pd.read_csv("Results/Results_table.csv")
    res["in_type"] = res["in_type"].replace(
        {
            "One-Hot": "CNN",
            "DAPseq": "L. (DAPseq)",
            "String": "AgroNT",
            "6-mer": "L. (6mer)",
            "embeddings": "L. (AgroNT emb.)",
        }
    )
    res = res[(res["length"] == "2048") | (res["length"] == "not apply")]
    res = res[res["exons"] != "masked"]
    res = res[res["rc"] != "False"]
    res = res[
        (
            (res["outcome_type"] == "quantiles_per_treatment")
            | (res["outcome_type"] == "DE_per_treatment")
        )
        & (res["metric"] == metric)
    ]
    res["outcome_type"] = res["outcome_type"].replace(
        {
            "DE_per_treatment": "S.DE",
            "quantiles_per_treatment": "S.Q",
        }
    )
    print(res)
    # get the pvalues for quantiles and DE, comparing AgroNT to CNN and Linear
    plt.figure(figsize=figsize, dpi=300)
    print(res)
    # check if NAN
    print(res.dropna(subset=["value"]))
    # Reorder the outcome_type to make "S.DE" appear first and "S.Q" second
    res["outcome_type"] = pd.Categorical(
        res["outcome_type"], categories=["S.DE", "S.Q"], ordered=True
    )
    # print the maximum MCC for each in_type
    print(res.groupby(["in_type", "outcome_type"])["value"].max())
    ax = sns.boxplot(x="outcome_type", y="value", data=res, hue="in_type")
    # print the average of the values per in_type
    print(res.groupby(["in_type", "outcome_type"])["value"].mean())
    sns.swarmplot(
        x="outcome_type",
        y="value",
        data=res,
        hue="in_type",
        dodge=True,
        marker=".",
        color=".25",
        legend=False,
    )
    # pairs: For plots grouped by hue: `[
    #           ((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))
    #        ]`
    if pvals:
        pairs = [
            (
                (res["outcome_type"].unique()[0], "CNN"),
                (res["outcome_type"].unique()[0], "L. (DAPseq)"),
            ),
            (
                (res["outcome_type"].unique()[1], "CNN"),
                (res["outcome_type"].unique()[1], "L. (DAPseq)"),
            ),
            (
                (res["outcome_type"].unique()[0], "CNN"),
                (res["outcome_type"].unique()[0], "AgroNT"),
            ),
            (
                (res["outcome_type"].unique()[1], "CNN"),
                (res["outcome_type"].unique()[1], "AgroNT"),
            ),
            (
                (res["outcome_type"].unique()[0], "CNN"),
                (res["outcome_type"].unique()[0], "L. (6mer)"),
            ),
            (
                (res["outcome_type"].unique()[1], "CNN"),
                (res["outcome_type"].unique()[1], "L. (6mer)"),
            ),
            # (
            #    (res["outcome_type"].unique()[1], "CNN"),
            #    (res["outcome_type"].unique()[1], "L. (AgroNT emb.)"),
            # )
        ]

        annotator = Annotator(
            ax, data=res, x="outcome_type", y="value", hue="in_type", pairs=pairs
        )
        annotator.configure(
            test="Mann-Whitney", text_format="star", loc="inside", fontsize=10
        )
        annotator.apply_and_annotate()
    plt.legend(loc="upper left", frameon=False, ncol=1)
    plt.ylabel(f"{metric}-ROC" if metric == "AUC" else metric)
    plt.xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    
    if metric == "AUC":
        plt.savefig("Images/figure_2a.pdf", bbox_inches="tight")
    elif metric == "MCC":
        plt.savefig("Images/SUP_figure_1.pdf", bbox_inches="tight")

def figure_1b(
    figsize=(10, 7),
    fitted_values_file="Data/Processed/mRNA/fitted_values_Hormone_C.csv",
    gene_selected="AT2G14610",
):
    X = pd.read_csv(fitted_values_file)
    # set Unnamed: 0 name as "gene"
    X = X.rename(columns={"Unnamed: 0": "gene"})
    X = X[X["gene"] == gene_selected]
    # select columns that have control
    X_control = X.filter(like="control")
    # select columns that have treatment
    treatment = fitted_values_file.split("/")[-1].split("_")[-1].split(".")[0]
    X_treatment = X.filter(like=treatment)

    # put in long format, with the columns as features.
    # They ae separated by _, being treatment, time, replication
    X_control = pd.melt(X_control, var_name="features", value_name="fitted_values")
    X_control["treatment"] = "control"
    X_treatment = pd.melt(X_treatment, var_name="features", value_name="fitted_values")
    X_treatment["treatment"] = mapping[treatment]
    # merge the two dataframes
    X = pd.concat([X_control, X_treatment])
    # split the features column into three columns
    X[["treatment", "time", "replication"]] = X["features"].str.split("_", expand=True)
    # if treatment =! "control" then set treatment to the value in the mapping
    X["treatment"] = X["treatment"].replace(mapping)
    # drop the features column
    X = X.drop(columns=["features"])
    # goup by by replication and take the average (the are actually the same value!)
    X = X.groupby(["treatment", "time"]).mean().reset_index()
    # set the time as int
    X["time"] = X["time"].astype(int)

    # order the dataframe by time
    X = X.sort_values(by=["treatment", "time"])

    # now make a lineplot for treatment and control

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # make the lineplot
    #  add a column for the ration
    nom = X[X["treatment"] != "control"]["fitted_values"].values
    denom = X[X["treatment"] == "control"]["fitted_values"].values

    ratio = nom / denom
    max_ratio = np.abs((ratio - 1)).max()
    ratio_df = pd.DataFrame(
        {
            "time": X[X["treatment"] != "control"]["time"].values,
            "ratio": ratio,
            "treatment": mapping[treatment],
        }
    )
    sns.lineplot(
        data=X,
        x="time",
        y="fitted_values",
        hue="treatment",
        ax=ax,
        palette={"control": "green", mapping[treatment]: "dimgray"},
    )
    # plot the ratio only for the treatment
    ax2 = ax.twinx()  # Create a second y-axis
    ax2.spines["right"].set_visible(True)
    # and the value color
    sns.lineplot(
        data=ratio_df, x="time", y="ratio", linestyle="--", ax=ax2, color="black"
    )

    # add shaded area below the lines
    # add the shaded area for the control
    ax.fill_between(
        X[X["treatment"] == "control"]["time"],
        X[X["treatment"] == "control"]["fitted_values"],
        color="green",
        alpha=0.2,
    )
    # add the shaded area for the treatment
    ax.fill_between(
        X[X["treatment"] == "control"]["time"],
        X[X["treatment"] != "control"]["fitted_values"],
        color="dimgray",
        alpha=0.2,
    )

    # add the points
    sns.scatterplot(
        data=X,
        x="time",
        y="fitted_values",
        hue="treatment",
        ax=ax,
        legend=False,
        palette={"control": "green", mapping[treatment]: "dimgray"},
    )
    # make a red circle on top of the maximum absolute (ratio -1)
    # get the index of the maximum absolute value
    max_index = np.abs((ratio - 1)).argmax()
    # get the time of the maximum absolute value
    max_time = X[X["treatment"] != "control"]["time"].values[max_index]
    # get the value of the maximum absolute value
    max_value = ratio[max_index]
    # plot the circle
    ax2.scatter(
        max_time,
        max_value,
        facecolors="none",
        edgecolors="red",
        s=200,
        linewidth=2,
        zorder=10,
    )
    # add the legend
    ax2.set_ylabel("Ratio")
    # ax2.tick_params(axis="y", labelcolor="#555555")
    # x title
    ax.set_xlabel("Time (min.)")
    # y title
    ax.set_ylabel("RNA counts")
    # remove grid
    ax.grid(False)
    ax2.grid(False)
    # remove the top line
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # remove the buffer between the lines and the axes
    ax.set_ylim(-0.01, 1.2 * np.max(X["fitted_values"]))
    ax.set_xlim(np.min(X["time"]) * 0.97, 1.001 * np.max(X["time"]))

    for spine in ["left", "bottom", "right"]:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color("black")

    for spine in ["right"]:
        ax2.spines[spine].set_visible(True)
        ax2.spines[spine].set_linewidth(1.5)
        ax2.spines[spine].set_color("black")

    # fill the color of the axes

    ax.tick_params(axis="both", which="major", width=1.5, length=6)
    ax.tick_params(axis="both", which="minor", width=1.5, length=4)

    # Define custom legend handles
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="control"),
        Patch(facecolor="dimgray", edgecolor="black", label=mapping[treatment]),
    ]

    # Add the custom legend
    ax.legend(
        handles=legend_elements,
        title="Condition",
        loc="upper right",
        frameon=False,
        ncol=1,
    )
    plt.savefig("Images/figure_1b.pdf", bbox_inches="tight")

def figure_2b(figsize=(10, 7), pvals=True, metric="Spearman"):
    res = pd.read_csv("Results/Results_table.csv")
    res["in_type"] = res["in_type"].replace(
        {
            "One-Hot": "CNN",
            "DAPseq": "L. (DAPseq)",
            "String": "AgroNT",
            "6-mer": "L. (6mer)",
            "embeddings": "L. (AgroNT emb.)",
        }
    )

    res = res[(res["length"] == "2048") | (res["length"] == "not apply")]
    res = res[res["exons"] != "masked"]
    res = res[res["rc"] != "False"]
    res = res[
        ((res["outcome_type"] == "amplitude") | (res["outcome_type"] == "log2FC"))
        & (res["metric"] == metric)
    ]

    # rename log2FC to LFC.T and amplitude to LFC.A
    res["outcome_type"] = res["outcome_type"].replace(
        {
            "amplitude": "LFC.A",
            "log2FC": "LFC.T",
        }
    )
    print(res)

    # get the pvalues for quantiles and DE, comparing AgroNT to CNN and Linear
    plt.figure(figsize=figsize, dpi=300)
    print(res)
    # print the res just for the CNN
    print(res[res["in_type"] == "CNN"])
    # and just for AgroNT
    print(res[res["in_type"] == "AgroNT"])
    # just for DAPseq
    print(res[res["in_type"] == "L. (DAPseq)"])
    # just for 6mer
    print(res[res["in_type"] == "L. (6mer)"])

    ax = sns.boxplot(x="outcome_type", y="value", data=res, hue="in_type")
    sns.swarmplot(
        x="outcome_type",
        y="value",
        data=res,
        hue="in_type",
        dodge=True,
        marker=".",
        color=".25",
        legend=False,
    )
    # pairs: For plots grouped by hue: `[
    #           ((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))
    #        ]`
    if pvals:
        pairs = [
            (
                (res["outcome_type"].unique()[0], "CNN"),
                (res["outcome_type"].unique()[0], "L. (DAPseq)"),
            ),
            (
                (res["outcome_type"].unique()[1], "CNN"),
                (res["outcome_type"].unique()[1], "L. (DAPseq)"),
            ),
            (
                (res["outcome_type"].unique()[0], "CNN"),
                (res["outcome_type"].unique()[0], "AgroNT"),
            ),
            (
                (res["outcome_type"].unique()[1], "CNN"),
                (res["outcome_type"].unique()[1], "AgroNT"),
            ),
            (
                (res["outcome_type"].unique()[0], "CNN"),
                (res["outcome_type"].unique()[0], "L. (6mer)"),
            ),
            (
                (res["outcome_type"].unique()[1], "CNN"),
                (res["outcome_type"].unique()[1], "L. (6mer)"),
            ),
            # (
            #    (res["outcome_type"].unique()[1], "CNN"),
            #    (res["outcome_type"].unique()[1], "L. (AgroNT emb.)"),
            # )
        ]

        annotator = Annotator(
            ax, data=res, x="outcome_type", y="value", hue="in_type", pairs=pairs
        )
        annotator.configure(
            test="Mann-Whitney", text_format="star", loc="inside", fontsize=10
        )
        annotator.apply_and_annotate()
    # remove the legend altogether
    plt.legend().remove()
    # plt.legend(loc="upper center", fontsize=10, frameon=False, ncol=1)# TMP
    if metric == "Spearman":
        plt.ylabel("Spearman Correlation")
    else:
        plt.ylabel("Pearson Correlation")
    # add grid
    plt.xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    plt.savefig("Images/figure_2b.pdf", bbox_inches="tight")

def figure_3a(outcome="log2FC"):
    if outcome not in ["log2FC", "amplitude"]:
        raise ValueError("outcome must be either log2FC or amplitude")

    DNA_specs = [814, 200, 200, 814]
    treatments = ["B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"]
    Y_hats = []
    training_specs = {
        "lr": 7e-5,
        "weight_decay": 0.00001,
        "n_epochs": 1500,
        "patience": 25,
        "problem_type": outcome,
        "equivariant": True,
        "n_labels": len(treatments),
        "batch_size": 64,
    }
    (
        mRNA_train,
        mRNA_validation,
        mRNA_test_CNN,
        TSS_sequences,
        TTS_sequences,
        metadata,
    ) = load_data(
        train_proportion=0.80,
        val_proportion=0.1,
        DNA_specs=DNA_specs,
        treatments=treatments,
        problem_type=outcome,
        mask_exons=False,
        dna_format="One_Hot_Encoding",
    )

    cnn_config_url = f"Results/CNN/{outcome}/2048/exons_masked_False/config.json"
    with open(cnn_config_url, "r") as f:
        cnn_config = json.load(f)

    for i in range(0, 5):
        model_weights = (
            f"Results/CNN/{outcome}/2048/exons_masked_False/model_{i}/best_model.pth"
        )
        # load the model
        model = myCNN(
            n_labels=cnn_config["n_labels"],
            n_ressidual_blocks=cnn_config["n_ressidual_blocks"],
            in_channels=cnn_config["in_channels"],
            out_channels=cnn_config["out_channels"],
            kernel_size=cnn_config["kernel_size"],
            max_pooling_kernel_size=cnn_config["max_pooling_kernel_size"],
            dropout_rate=cnn_config["dropout_rate"],
            stride=cnn_config["stride"],
            RC_equivariant=cnn_config["equivariant"],
        )
        # load the weights
        model.load_state_dict(torch.load(model_weights))

        # pass through the test functionality
        Y_hat, Y = test_cnn(
            model=model,
            training_specs=training_specs,
            TSS_sequences=TSS_sequences,
            TTS_sequences=TTS_sequences,
            mRNA_test=pd.concat([mRNA_test_CNN]),
            device=torch.device("cuda"),
            treatments=treatments,
            store_folder=f"Results/CNN/{outcome}/2048/exons_masked_False/model_{i}",
            save_results=False,
        )

        print(Y_hat.shape)

        Y_hats.append(Y_hat[np.newaxis, :])

    Y_hat = np.concatenate(Y_hats, axis=0)
    Y_hat = np.mean(Y_hat, axis=0)
    Y_hat = pd.DataFrame(Y_hat, columns=[mapping[t] for t in treatments])

    Y = pd.DataFrame(Y, columns=[mapping[t] for t in treatments])

    # Now for each treatment, get the R^2 and pvalue of each model using statsmodels
    # and plot the regression line in the scatterplot

    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 8), dpi=300)

    # fig.suptitle("Regression of True vs Predicted Values", fontsize=14)

    for i, treatment in enumerate([mapping[t] for t in treatments]):
        # Get the R^2 and p-value
        X = Y_hat[treatment].values
        X = sm.tools.tools.add_constant(X)  # Add intercept
        model = sm.regression.linear_model.OLS(Y[treatment], X)
        results = model.fit()
        # calculate the spearman correlation
        spearman = stats.spearmanr(Y_hat[treatment], Y[treatment]).statistic
        r2 = results.rsquared
        pvalue = results.pvalues[1]
        print(f"{treatment}: R² = {r2:.2f}, p-value = {pvalue:.4e}")

        # Plot scatterplot
        ax = axes[i // 6, i % 6]
        # ax.scatter(Y_hat[treatment], Y[treatment], s=1, alpha=0.5)
        ax.set_title(
            f"{treatment}\n" + rf"$R^2$: {r2:.2f}" + rf" $\rho$: {spearman:.2f}",
            fontsize=10,
        )
        # Plot scatterplot with density
        xy = np.vstack([Y_hat[treatment], Y[treatment]])
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(
            Y_hat[treatment], Y[treatment], c=z, s=1, cmap="rocket", alpha=0.8
        )
        # fig.colorbar(scatter, ax=ax, label='Density')

        # Plot regression line
        x = np.linspace(Y_hat[treatment].min(), Y_hat[treatment].max(), 100)
        y = results.params[0] + results.params[1] * x
        ax.plot(x, y, color="green", linestyle="--", linewidth=1.5)
        # reduce the fontsize of the ticks
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=9)
        # make the spines softer
        ax.spines["left"].set_linewidth(1)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["bottom"].set_color("black")
    axes = axes.flatten()
    fig.delaxes(axes[len(treatments)])
    # Set overall x and y labels
    fig.text(0.5, 0.02, r"Predicted LFC.T", ha="center", fontsize=12)
    fig.text(0.02, 0.5, r"True LFC.T", va="center", rotation="vertical", fontsize=12)

    plt.tight_layout(
        rect=[0.03, 0.03, 1, 0.95], h_pad=1.6, w_pad=1.6
    )  # Adjust layout to fit suptitle and increase horizontal separation
    # Save the figure
    plt.savefig(f"Images/figure_3a.pdf", bbox_inches="tight")
    # also as a png
    plt.savefig(f"Images/figure_3a.png", bbox_inches="tight")

def figure_3b(figsize=(10, 7), outcome="log2FC"):
    if outcome not in ["log2FC", "amplitude"]:
        raise ValueError("outcome must be either log2FC or amplitude")

    # Initialize figure
    plt.figure(figsize=figsize, dpi=300)

    # Read and concatenate data
    my_concat = pd.DataFrame()

    # CNN Model
    for i in range(0, 5):
        metrics = pd.read_csv(
            f"Results/CNN/{outcome}/2048/exons_masked_False/model_{i}/test_metrics.csv",
            index_col=0,
        )
        metric_m = pd.DataFrame(metrics.T)
        metric_m["replicate"] = i
        metric_m["model"] = "CNN"
        my_concat = pd.concat([my_concat, metric_m])

    # AgroNT Model
    metrics = pd.read_csv(f"Results/agroNT/{outcome}/test_metrics.csv", index_col=0)
    metric_m = pd.DataFrame(metrics.T)
    metric_m["replicate"] = 0
    metric_m["model"] = "AgroNT"
    my_concat = pd.concat([my_concat, metric_m])

    # 6-mer
    for file in os.listdir(f"Results/linear_models/{outcome}/6-mer"):
        if "metrics" not in file:
            continue
        treatment_name = file.split("_")[0]
        metrics = pd.read_csv(
            f"Results/linear_models/{outcome}/6-mer/{file}", index_col=0
        )
        metric_m = pd.DataFrame(metrics)
        metric_m["replicate"] = 0
        metric_m["model"] = "6-mer"
        metric_m["treatment"] = treatment_name
        # reset index
        metric_m.reset_index(inplace=True)
        # set index treatment
        metric_m.set_index("treatment", inplace=True)
        my_concat = pd.concat([my_concat, metric_m])

    # DAPseq
    for file in os.listdir(f"Results/linear_models/{outcome}/DAPseq/"):
        if "metrics" not in file:
            continue
        treatment_name = file.split("_")[0]
        metrics = pd.read_csv(
            f"Results/linear_models/{outcome}/DAPseq/{file}", index_col=0
        )
        metric_m = pd.DataFrame(metrics)
        metric_m["replicate"] = 0
        metric_m["model"] = "DAPseq"
        metric_m["treatment"] = treatment_name
        # reset index
        metric_m.reset_index(inplace=True)
        # set index treatment
        metric_m.set_index("treatment", inplace=True)
        my_concat = pd.concat([my_concat, metric_m])

    # Reset index
    my_concat.reset_index(inplace=True)
    my_concat.rename(columns={"index": "Metric"}, inplace=True)
    # apply mapping to the treatment
    my_concat["treatment"] = my_concat["Metric"].replace(mapping)

    # Boxplot for CNN
    sns.boxplot(
        data=my_concat[my_concat["model"] == "CNN"],
        x="treatment",
        y="sign_accuracy",
        color="#2ca02c",
        width=0.6,
        showfliers=False,
    )

    # Stripplot (dots) for all other models
    strip = sns.stripplot(
        data=my_concat[
            (my_concat["model"] != "CNN") & (my_concat["model"] != "Random")
        ],
        x="treatment",
        y="sign_accuracy",
        hue="model",
        dodge=True,
        jitter=True,
        alpha=0.8,
        linewidth=0.5,
        size=10,
        palette={
            "6-mer": "#1f77b4",
            "DAPseq": "#ff7f0e",
            "AgroNT": "#d62728",
            "Random": "#999999",
        },
    )


    plt.xticks(rotation=45, ha="right")
    if outcome == "log2FC":
        plt.ylabel("LFC.T direction correctly predicted (proportion)")
    else:
        plt.ylabel("LFC.A direction correctly predicted (proportion)")
    plt.xlabel("Treatment")
    strip.spines["top"].set_visible(False)
    strip.spines["right"].set_visible(False)
   
    # Manually create a legend including CNN
    handles, labels = strip.get_legend_handles_labels()
    cnn_patch = mpatches.Patch(color="#2ca02c", label="CNN")
    handles.insert(0, cnn_patch)
    labels.insert(0, "CNN")
    # Add vertical grid lines between treatments
    # Add customized vertical grid lines between treatments
    ax = plt.gca()
    treatments = my_concat["treatment"].unique()
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    
    # Calculate positions for vertical gridlines (offset by 0.5)
    for i in range(len(treatments)+1):
        ax.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    plt.legend(
        handles=handles,
        labels=labels,
        title="Model",
        loc="upper right",
        bbox_to_anchor=(0.19, 1) if outcome == "log2FC" else (1., 0.45),
        frameon=False if outcome == "log2FC" else True,)

    # plt.title("Comparison of Test Metrics Across Models")
    plt.tight_layout()
    if outcome == "log2FC":
        plt.savefig(f"Images/figure_3b.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"Images/SUP_figure_5.pdf", bbox_inches="tight") 

def figure_3c(figsize=(10, 7), outcome="log2FC"):
    DNA_specs = [814, 200, 200, 814]
    treatments = ["B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"]


    training_specs = {
        "lr": 7e-5,
        "weight_decay": 0.00001,
        "n_epochs": 1500,
        "patience": 25,
        "problem_type": outcome,
        "equivariant": True,
        "n_labels": len(treatments),
        "batch_size": 64,
    }
    (
        mRNA_train,
        mRNA_validation,
        mRNA_test_CNN,
        TSS_sequences,
        TTS_sequences,
        metadata,
    ) = load_data(
        train_proportion=0.85,
        val_proportion=0.05,
        DNA_specs=DNA_specs,
        treatments=treatments,
        problem_type=outcome,
        mask_exons=False,
        dna_format="One_Hot_Encoding",
    )

    Y_hats = []
    cnn_config_url = f"Results/CNN/{outcome}/2048/exons_masked_False/config.json"
    with open(cnn_config_url, "r") as f:
        cnn_config = json.load(f)

    for i in range(0, 5):
        model_weights = (
            f"Results/CNN/{outcome}/2048/exons_masked_False/model_{i}/best_model.pth"
        )
        # load the model
        model = myCNN(
            n_labels=cnn_config["n_labels"],
            n_ressidual_blocks=cnn_config["n_ressidual_blocks"],
            in_channels=cnn_config["in_channels"],
            out_channels=cnn_config["out_channels"],
            kernel_size=cnn_config["kernel_size"],
            max_pooling_kernel_size=cnn_config["max_pooling_kernel_size"],
            dropout_rate=cnn_config["dropout_rate"],
            stride=cnn_config["stride"],
            RC_equivariant=cnn_config["equivariant"],
        )
        # load the weights
        model.load_state_dict(torch.load(model_weights))

        Y_hat, Y = test_cnn(
            model=model,
            training_specs=training_specs,
            TSS_sequences=TSS_sequences,
            TTS_sequences=TTS_sequences,
            mRNA_test=pd.concat([mRNA_test_CNN]),
            device=torch.device("cuda"),
            treatments=treatments,
            store_folder=f"Results/CNN/{outcome}/2048/exons_masked_False/model_{i}",  # does not matter
            save_results=False,
        )

        print(Y_hat.shape)

        Y_hats.append(Y_hat[np.newaxis, :])
        # break

    Y_hat = np.concatenate(Y_hats, axis=0)
    Y_hat = np.mean(Y_hat, axis=0)
    Y_hat = pd.DataFrame(Y_hat, columns=[mapping[t] for t in treatments])

    Y = pd.DataFrame(Y, columns=[mapping[t] for t in treatments])

    # get the correlation matrix
    # correlation_matrix = np.corrcoef(Y_hat.T)
    correlation_matrix = stats.spearmanr(Y_hat).statistic
    # plot the correlation matrix
    plt.figure(figsize=figsize, dpi=300)
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        annot=True,
        cbar=False,
        annot_kws={"size": 8, "color": "white", "weight": "bold"},
        fmt=".2f",
        linewidths=0.5,
        linecolor="black",
        vmin=-1,
        vmax=1,  # Enforce colormap range from -1 to 1
    )
    # reduce the fontsize of the numbers inside the heatmap

    # add the names of the treatments
    plt.xticks(
        np.arange(len(treatments)) + 0.5,
        labels=[mapping[t] for t in treatments],
        rotation=40,
        fontsize=10,
    )
    plt.yticks(
        np.arange(len(treatments)) + 0.5,
        labels=[mapping[t] for t in treatments],
        rotation=40,
        fontsize=10,
    )
    # remove colormap
    # plt.colorbar().remove()

    # save the figure
    if outcome == "log2FC":
        plt.savefig(f"Images/figure_3c.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"Images/figure_SUP3_amplitude.pdf", bbox_inches="tight") # TODO

def figure_4a(figsize=(10, 7), metric="AUC"):
    res = pd.read_csv("Results/Results_table.csv")
    # Change the "in_type" column name to "Model type"
    res = res.rename(columns={"in_type": "Model type"})
    # Now change One-hot encoding to CNN, 6mer to Linear (6mer) and DAPseq to Linear (DAPseq)
    res["Model type"] = res["Model type"].replace(
        {"One-Hot": "CNN", "6-mer": "L. (6mer)", "DAPseq": "L. (DAPseq)"}
    )
    # remove the 6mer, DAPseq and AgroNT
    res = res[res["Model type"] == "CNN"]
    res = res[res["rc"] != "False"]
    # more name changes
    res["outcome_type"] = res["outcome_type"].replace(
        {
            "quantiles_per_treatment": "S.Q",
            "DE_per_treatment": "S.DE",
            "amplitude": "LFC.A",
            "log2FC": "LFC.T",
        }
    )
    # select only log2FC and sensitivity to treatment
    # the idea is to make a 2x2 plot, where the first row is for length
    # and the second row is for exons.
    # the first column is for sensitivity and the second column is for log2FC

    plt.figure(figsize=figsize, dpi=300)
    res_length = res[res["exons"] != "masked"]

    # remove all not in quantiles_per_treatment or DE_per_treatment
    res_length = res_length[
        (res_length["outcome_type"] == "S.DE") | (res_length["outcome_type"] == "S.Q")
    ]

    res_length = res_length[res_length["metric"] == metric]
    ax = sns.boxplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="length",
        palette=magentaorgange_palette,
    )
    sns.swarmplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="length",
        dodge=True,
        color=".25",
        legend=False,
        marker=".",
    )
    pairs = [
        (
            (res_length["outcome_type"].unique()[0], "4096"),
            (res_length["outcome_type"].unique()[0], "2048"),
        ),
        (
            (res_length["outcome_type"].unique()[1], "4096"),
            (res_length["outcome_type"].unique()[1], "2048"),
        ),
    ]

    annotator = Annotator(
        ax, data=res_length, x="outcome_type", y="value", hue="length", pairs=pairs
    )
    annotator.configure(
        test="Mann-Whitney", text_format="star", loc="inside", fontsize=10
    )
    annotator.apply_and_annotate()
    plt.ylabel(f"{metric}-ROC" if metric == "AUC" else f"{metric}")
    plt.xlabel("")
    plt.legend(loc="upper center", frameon=False, ncol=1)
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    # remove x tick labels
    plt.xticks([])
    # save the figure
    plt.savefig("Images/figure_4a.pdf", bbox_inches="tight")

def figure_4b(figsize=(10, 7), metric="Spearman"):
    res = pd.read_csv("Results/Results_table.csv")
    # Change the "in_type" column name to "Model type"
    res = res.rename(columns={"in_type": "Model type"})
    # Now change One-hot encoding to CNN, 6mer to Linear (6mer) and DAPseq to Linear (DAPseq)
    res["Model type"] = res["Model type"].replace(
        {"One-Hot": "CNN", "6-mer": "L. (6mer)", "DAPseq": "L. (DAPseq)"}
    )
    # remove the 6mer, DAPseq and AgroNT
    res = res[res["Model type"] == "CNN"]
    res = res[res["rc"] != "False"]
    # more name changes
    res["outcome_type"] = res["outcome_type"].replace(
        {
            "quantiles_per_treatment": "S.Q",
            "DE_per_treatment": "S.DE",
            "amplitude": "LFC.A",
            "log2FC": "LFC.T",
        }
    )
    # select only log2FC and sensitivity to treatment
    # the idea is to make a 2x2 plot, where the first row is for length
    # and the second row is for exons.
    # the first column is for sensitivity and the second column is for log2FC

    plt.figure(figsize=figsize, dpi=300)
    res_length = res[res["exons"] != "masked"]
    # remove all not in quantiles_per_treatment or DE_per_treatment
    res_length = res_length[
        (res_length["outcome_type"] == "LFC.T")
        | (res_length["outcome_type"] == "LFC.A")
    ]

    res_length = res_length[res_length["metric"] == metric]

    ax = sns.boxplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="length",
        palette=magentaorgange_palette,
    )
    sns.swarmplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="length",
        dodge=True,
        color=".25",
        legend=False,
        marker=".",
    )
    pairs = [
        (
            (res_length["outcome_type"].unique()[0], "4096"),
            (res_length["outcome_type"].unique()[0], "2048"),
        ),
        (
            (res_length["outcome_type"].unique()[1], "4096"),
            (res_length["outcome_type"].unique()[1], "2048"),
        ),
    ]

    annotator = Annotator(
        ax, data=res_length, x="outcome_type", y="value", hue="length", pairs=pairs
    )
    annotator.configure(
        test="Mann-Whitney", text_format="star", loc="inside", fontsize=10
    )
    annotator.apply_and_annotate()
    if metric == "Spearman":
        plt.ylabel("Spearman Correlation")
    elif metric == "Pearson":
        plt.ylabel("Pearson Correlation")

    plt.xlabel("")
    #plt.legend(loc="upper center", frameon=False, ncol=1)
    plt.legend().remove()  # Removes the current legend
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    # save the figure
    # remove x tick labels
    plt.xticks([])
    plt.savefig("Images/figure_4b.pdf", bbox_inches="tight")

def figure_4c(figsize=(10, 7), metric="AUC"):
    res = pd.read_csv("Results/Results_table.csv")
    # Change the "in_type" column name to "Model type"
    res = res.rename(columns={"in_type": "Model type"})
    # Now change One-hot encoding to CNN, 6mer to Linear (6mer) and DAPseq to Linear (DAPseq)
    res["Model type"] = res["Model type"].replace(
        {"One-Hot": "CNN", "6-mer": "L. (6mer)", "DAPseq": "L. (DAPseq)"}
    )
    # remove the 6mer, DAPseq and AgroNT
    res = res[res["Model type"] == "CNN"]
    res = res[res["rc"] != "False"]
    # more name changes
    res["outcome_type"] = res["outcome_type"].replace(
        {
            "quantiles_per_treatment": "S.Q",
            "DE_per_treatment": "S.DE",
            "amplitude": "LFC.A",
            "log2FC": "LFC.T",
        }
    )
    # select only log2FC and sensitivity to treatment
    # the idea is to make a 2x2 plot, where the first row is for length
    # and the second row is for exons.
    # the first column is for sensitivity and the second column is for log2FC

    plt.figure(figsize=figsize, dpi=300)
    res_length = res[res["length"] != "4096"]
    # remove all not in quantiles_per_treatment or DE_per_treatment
    res_length = res_length[
        (res_length["outcome_type"] == "S.DE") | (res_length["outcome_type"] == "S.Q")
    ]

    res_length = res_length[res_length["metric"] == metric]

    ax = sns.boxplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="exons",
        palette=coraldarkteal_palette,
    )
    sns.swarmplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="exons",
        dodge=True,
        color=".25",
        legend=False,
        marker=".",
    )
    pairs = [
        (
            (res_length["outcome_type"].unique()[0], "masked"),
            (res_length["outcome_type"].unique()[0], "all"),
        ),
        (
            (res_length["outcome_type"].unique()[1], "masked"),
            (res_length["outcome_type"].unique()[1], "all"),
        ),
    ]

    annotator = Annotator(
        ax, data=res_length, x="outcome_type", y="value", hue="exons", pairs=pairs
    )
    annotator.configure(
        test="Mann-Whitney", text_format="star", loc="inside", fontsize=10
    )
    annotator.apply_and_annotate()
    plt.ylabel(f"{metric}-ROC" if metric == "AUC" else f"{metric}")

    plt.xlabel("")
    plt.legend(loc="upper center", frameon=False, ncol=1)
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    # save the figure
    plt.savefig("Images/figure_4c.pdf", bbox_inches="tight")

def figure_4d(figsize=(10, 7), metric="Spearman"):
    res = pd.read_csv("Results/Results_table.csv")
    # Change the "in_type" column name to "Model type"
    res = res.rename(columns={"in_type": "Model type"})
    # Now change One-hot encoding to CNN, 6mer to Linear (6mer) and DAPseq to Linear (DAPseq)
    res["Model type"] = res["Model type"].replace(
        {"One-Hot": "CNN", "6-mer": "L. (6mer)", "DAPseq": "L. (DAPseq)"}
    )
    # remove the 6mer, DAPseq and AgroNT
    res = res[res["Model type"] == "CNN"]
    res = res[res["rc"] != "False"]
    # more name changes
    res["outcome_type"] = res["outcome_type"].replace(
        {
            "quantiles_per_treatment": "S.Q",
            "DE_per_treatment": "S.DE",
            "amplitude": "LFC.A",
            "log2FC": "LFC.T",
        }
    )
    # select only log2FC and sensitivity to treatment
    # the idea is to make a 2x2 plot, where the first row is for length
    # and the second row is for exons.
    # the first column is for sensitivity and the second column is for log2FC

    plt.figure(figsize=figsize, dpi=300)
    res_length = res[res["length"] != "4096"]
    # remove all not in quantiles_per_treatment or DE_per_treatment
    res_length = res_length[
        (res_length["outcome_type"] == "LFC.T")
        | (res_length["outcome_type"] == "LFC.A")
    ]

    res_length = res_length[res_length["metric"] == metric]

    ax = sns.boxplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="exons",
        palette=coraldarkteal_palette,
    )
    sns.swarmplot(
        x="outcome_type",
        y="value",
        data=res_length,
        hue="exons",
        dodge=True,
        color=".25",
        legend=False,
        marker=".",
    )
    pairs = [
        (
            (res_length["outcome_type"].unique()[0], "masked"),
            (res_length["outcome_type"].unique()[0], "all"),
        ),
        (
            (res_length["outcome_type"].unique()[1], "masked"),
            (res_length["outcome_type"].unique()[1], "all"),
        ),
    ]

    annotator = Annotator(
        ax, data=res_length, x="outcome_type", y="value", hue="exons", pairs=pairs
    )
    annotator.configure(
        test="Mann-Whitney", text_format="star", loc="inside", fontsize=10
    )
    annotator.apply_and_annotate()
    if metric == "Spearman":
        plt.ylabel("Spearman Correlation")
    elif metric == "Pearson":
        plt.ylabel("Pearson Correlation")

    plt.xlabel("")
    #plt.legend(loc="upper center", frameon=False, ncol=1)
    #remove the legend 
    plt.legend().remove()  # Removes the current legend
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    # save the figure
    plt.savefig("Images/figure_4d.pdf", bbox_inches="tight")

def figure_5a(figsize=(10, 7)):
    '''
    PCA coefficients 6-mer log2fc
    '''

    coefs = pd.DataFrame()
    for treatment in ["3-OH10","chitooct","elf18","flg22","nlp20","OGs","Pep1"]:
        coef = pd.read_csv(f"Results/linear_models/log2FC/6-mer/{treatment}_coefficients.csv")
        coef = coef.rename(columns={"Coefficient": treatment})
        if coefs.empty:
            coefs = coef
        else:
            coefs = pd.merge(coefs, coef, on="TF", how="outer").fillna(0)
    
    coefs = coefs.set_index("TF")
    coefs =  coefs.mean(axis=1)
    
    # Now rename column as PTI
    coefs = coefs.rename("PTI")

    # add the rest

    for hormone in ["MeJA", "SA", "SA+MeJA", "ABA"]:
        coef = pd.read_csv(f"Results/linear_models/log2FC/6-mer/{hormone}_coefficients.csv")
        coef = coef.rename(columns={"Coefficient": hormone})
        coefs = pd.merge(coefs, coef, on="TF", how="outer").fillna(0)
    
    # Set TF as index and fill NaN values with 0
    coefs = coefs.set_index("TF").fillna(0)
    # preprocess by standardizing the data
    coefs.loc[:, :] = StandardScaler().fit_transform(coefs.values)
    # Perform PCA
    pca = PCA(n_components=2)
    coefs_pca = pca.fit_transform(coefs)  # Keep TFs as rows

    # Create DataFrame for plotting
    pca_df = pd.DataFrame(coefs_pca, columns=["PC1", "PC2"], index=coefs.index)

    # Select top 100 most variable treatments (highest absolute variance in PC1 or PC2)
    top_n = 25
    important_points = pca_df.apply(np.linalg.norm, axis=1).nlargest(top_n).index.tolist()

    # Create the biplot
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Plot samples (treatments)
    sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], s=8, color="black", ax=ax, alpha=0.5)

    # Store text labels for adjustment
    texts = []
    for i, txt in enumerate(pca_df.index):
        if txt in important_points:
            texts.append(ax.text(pca_df["PC1"][i], pca_df["PC2"][i], txt, fontsize=13, alpha=1))
    loadings = pca.components_  # Get PC1 and PC2 loadings
    scaling_factor = 14  # Adjust arrow length
    for i, treatment in enumerate(coefs.columns):
        texts.append(ax.text(loadings[0, i] * scaling_factor, loadings[1, i] * scaling_factor, treatment, fontsize=18, alpha=1, color='red'))
    # Adjust text positions to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='blue', alpha=1))
    
    # Plot loadings (treatment contributions)
    for i, treatment in enumerate(coefs.columns):
        ax.arrow(0, 0, loadings[0, i] * scaling_factor, loadings[1, i] * scaling_factor,
                color="green", alpha=1, head_width=0.5, head_length=0.5, linewidth=2)
    # Labels and title
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)", fontsize=16)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)", fontsize=16)
    #ax.set_title("PCA Biplot of DAP-seq Coefficients")
    # increase font x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=14)

    # add the spine right and top
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    # remove the grid
    ax.grid(False)
    # add a x and y lines at 0
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.axvline(0, color='black', lw=1, ls='--')
    # Save plot
    plt.savefig("Images/figure_5a.pdf", bbox_inches="tight")
    plt.close('all')
    print(f"Explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

def figure_5b(figsize=(10, 7)):
        '''
        PCA coefficients 6-mer log2fc
        '''

        coefs = pd.DataFrame()
        for treatment in ["3-OH10","chitooct","elf18","flg22","nlp20","OGs","Pep1"]:
            coef = pd.read_csv(f"Results/linear_models/log2FC/DAPseq/{treatment}_coefficients.csv")
            coef = coef.rename(columns={"Coefficient": treatment})
            if coefs.empty:
                coefs = coef
            else:
                coefs = pd.merge(coefs, coef, on="TF", how="outer").fillna(0)
        
        coefs = coefs.set_index("TF")
        coefs =  coefs.mean(axis=1)
        
        # Now rename column as PTI
        coefs = coefs.rename("PTI")

        # add the rest

        for hormone in ["MeJA", "SA", "SA+MeJA", "ABA"]:
            coef = pd.read_csv(f"Results/linear_models/log2FC/DAPseq/{hormone}_coefficients.csv")
            coef = coef.rename(columns={"Coefficient": hormone})
            coefs = pd.merge(coefs, coef, on="TF", how="outer").fillna(0)
        
        # Set TF as index and fill NaN values with 0
        coefs = coefs.set_index("TF").fillna(0)
        # preprocess by standardizing the data
        coefs.loc[:, :] = StandardScaler().fit_transform(coefs.values)
        # Perform PCA
        pca = PCA(n_components=2)
        coefs_pca = pca.fit_transform(coefs)  # Keep TFs as rows
        coefs_pca[:, 0] = coefs_pca[:, 0] #* -1
        # Create DataFrame for plotting
        pca_df = pd.DataFrame(coefs_pca, columns=["PC1", "PC2"], index=coefs.index)

        # Select top 100 most variable treatments (highest absolute variance in PC1 or PC2)
        top_n = 15
        important_points = pca_df.apply(np.linalg.norm, axis=1).nlargest(top_n).index.tolist()

        # get the motifs of the important points
        consensus_motifs = {tf:get_tf_consensus_from_jaspar(tf, "Arabidopsis thaliana") for tf in important_points}
        #import pdb; pdb.set_trace()
        # Create the biplot
        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        # Plot samples (treatments)
        sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], s=8, color="black", ax=ax, alpha=0.5)

        # Store text labels for adjustment
        texts = []
        for i, txt in enumerate(pca_df.index):
            if txt in important_points:
                texts.append(ax.text(
                    pca_df["PC1"][i], 
                    pca_df["PC2"][i], 
                    f"{txt}\n({consensus_motifs[txt]})" if txt in consensus_motifs else txt, 
                    fontsize=11, 
                    alpha=1
                ))
        loadings = pca.components_  # Get PC1 and PC2 loadings
        loadings[0, :] = loadings[0, :]# * -1
        scaling_factor = 14  # Adjust arrow length
        for i, treatment in enumerate(coefs.columns):
            texts.append(ax.text(loadings[0, i] * scaling_factor, loadings[1, i] * scaling_factor, treatment, fontsize=18, alpha=1, color='red'))
        # Adjust text positions to avoid overlaps
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='blue', alpha=1))
        
        # Plot loadings (treatment contributions)
        for i, treatment in enumerate(coefs.columns):
            ax.arrow(0, 0, loadings[0, i] * scaling_factor, loadings[1, i] * scaling_factor,
                    color="green", alpha=1, head_width=0.5, head_length=0.5, linewidth=2)
        # Labels and title
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)", fontsize=16)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)", fontsize=16)
        #ax.set_title("PCA Biplot of DAP-seq Coefficients")
        # increase font x and y ticks
        ax.tick_params(axis='both', which='major', labelsize=14)

        # add the spine right and top
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        # remove the grid
        ax.grid(False)
        # add a x and y lines at 0
        ax.axhline(0, color='black', lw=1, ls='--')
        ax.axvline(0, color='black', lw=1, ls='--')
        
        # Save plot
        plt.savefig("Images/figure_5b.pdf", bbox_inches="tight")
        plt.close('all')
        print(f"Explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

def figure_5c(figsize=(10, 7), outcome = "log2FC"):
    # Define DNA regions
    DNA_specs = [814, 200, 200, 814]  # Promoter, UTR, UTR, Terminator

    # Load treatment folders
    treatments_folders = os.listdir(f"Results/Interpretation/{outcome}")

    contrib_scores = {}
    for treatment in treatments_folders:
        if treatment == "queries":
            continue
        gene_files = os.listdir(f"Results/Interpretation/{outcome}/{treatment}/hypothetical_scores")
        hyp_scores = []
        
        for gene_file in tqdm(gene_files):
            gene = gene_file.split("_")[0]
            hyp_scores.append(np.load(f"Results/Interpretation/{outcome}/{treatment}/hypothetical_scores/{gene_file}"))
        
        hyp_scores = np.array(hyp_scores)
        # Store contribution scores
        contrib_scores[treatment] = hyp_scores
        
    # Compute the average contribution scores
    avg_contrib_scores = {}
    # Trimmed nucleotide positions
    terminator_start = DNA_specs[0] + DNA_specs[1] + 20

    # Plot settings
    colors = ['green', 'orange', 'blue', 'red']
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]}, dpi=300)
    axes[0].spines['right'].set_visible(True)
    axes[1].spines['right'].set_visible(True)
    axes[0].spines['top'].set_visible(True)
    axes[1].spines['top'].set_visible(True)
    max = -np.inf
    min = np.inf
    for treatment in contrib_scores:
        avg_contrib_scores[treatment] = np.mean(contrib_scores[treatment], axis=(0, 1))
        max = np.max([max, np.max(avg_contrib_scores[treatment][:, :(DNA_specs[0] + DNA_specs[1] - 5)])])
        min = np.min([min, np.min(avg_contrib_scores[treatment][:, :(DNA_specs[0] + DNA_specs[1] - 5)])])
        # Average over treatments
        #avg_contrib_scores = np.mean(np.array(list(avg_contrib_scores.values())), axis=0)
        # min-max normalization
        #avg_contrib_scores[treatment] = 2 * (avg_contrib_scores[treatment] - np.min(avg_contrib_scores[treatment])) / (np.max(avg_contrib_scores[treatment]) - np.min(avg_contrib_scores[treatment])) - 1

    for treatment in contrib_scores:
        # min-max normalization
        avg_contrib_scores[treatment] = 2 * (avg_contrib_scores[treatment] - min) / (max - min) - 1
        # Promoter + TSS subplot
        axes[0].set_title("Promoter + 5' UTR")
        for nb, base in enumerate('ACGT'):
            axes[0].plot(np.arange(DNA_specs[0] + DNA_specs[1] - 5), avg_contrib_scores[treatment][nb, :(DNA_specs[0] + DNA_specs[1] - 5)], linewidth=1, color=colors[nb], alpha=0.1)

        # Terminator + TTS subplot
        axes[1].set_title("Terminator + 3' UTR")
        for nb, base in enumerate('ACGT'):
            #import pdb; pdb.set_trace()
            axes[1].plot(np.arange(DNA_specs[2] + DNA_specs[3] - 5), avg_contrib_scores[treatment][nb, (terminator_start + 5):], linewidth=1, color=colors[nb], alpha=0.1)
    
    avg_contrib_scores = np.mean(np.array(list(avg_contrib_scores.values())), axis=0)

    axes[0].set_title("Promoter + 5' UTR")
    for nb, base in enumerate('ACGT'):
        axes[0].plot(np.arange(DNA_specs[0] + DNA_specs[1] - 5), avg_contrib_scores[nb, :(DNA_specs[0] + DNA_specs[1] - 5)], linewidth=1, color=colors[nb])
    axes[0].axvline(DNA_specs[0], color='black', linestyle='dashed', linewidth=1, label="TSS")

    # Terminator + TTS subplot
    axes[1].set_title("Terminator + 3' UTR")
    for nb, base in enumerate('ACGT'):
        #import pdb; pdb.set_trace()
        axes[1].plot(np.arange(DNA_specs[2] + DNA_specs[3] - 5), avg_contrib_scores[nb, (terminator_start + 5):], linewidth=1, color=colors[nb])
    axes[1].axvline(DNA_specs[2], color='black', linestyle='dashed', linewidth=1, label="TTS")
    
    # Common formatting
    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.5)

        # Set a single y-axis label in the middle
    #fig.text(0.005, 0.5, "Normalized Hypothetical Contribution Scores", va='center', rotation='vertical', fontsize=15)

    # X-axis settings
    axes[0].set_xticks([
        0, DNA_specs[0], DNA_specs[0] + DNA_specs[1]
    ])
    axes[0].set_xticklabels([
        "0", f"{DNA_specs[0]} (TSS)", f"{DNA_specs[0] + DNA_specs[1]}"
    ])

    axes[1].set_xticks([
        0, DNA_specs[2], DNA_specs[2] + DNA_specs[3]
    ])
    axes[1].set_xticklabels([
        f"{DNA_specs[0] + DNA_specs[1] + 20}", f"{DNA_specs[2] + DNA_specs[0] + DNA_specs[1] + 20} (TTS)", f"{DNA_specs[2] + DNA_specs[3] + DNA_specs[0] + DNA_specs[1] + 20}"
    ])

    axes[1].set_xlabel("Nucleotide position")
    
    # Legend
    custom_lines = [
        Line2D([0], [0], color='green', lw=2, label='A'),
        Line2D([0], [0], color='orange', lw=2, label='C'),
        Line2D([0], [0], color='blue', lw=2, label='G'),
        Line2D([0], [0], color='red', lw=2, label='T')
    ]
    axes[0].legend(handles=custom_lines, loc='upper left', frameon=False)

    # set the ylim to -1, 1 in both
    axes[0].set_ylim(-1, 1)
    axes[1].set_ylim(-1, 1)
    # increase fontsize in ticks for both subplots
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)
    # plt.tight_layout()
    if outcome == "log2FC":
        plt.savefig("Images/figure_5c.pdf", bbox_inches="tight")
    else:
        plt.savefig("Images/figure_2_1.pdf", bbox_inches="tight")

def figure_5d(figsize=(10, 7)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # read the cluster table
    cluster_table = pd.read_csv("Results/Interpretation/cluster_patterns/cluster_table.csv", index_col=0)
    heatmap_data = cluster_table.iloc[
        :, :-2
    ]  # Exclude best_match and query_consensus columns
    # Define the maximum absolute value for symmetric scaling
    vmax = heatmap_data.abs().max().max()

    # Create a custom colormap with white at the center
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_RdWGn", ["red", "white", "green"], N=256
    )

    # Create a normalization that sets 0 to white
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Create the heatmap
    heatmap = ax.imshow(heatmap_data, cmap=custom_cmap, aspect="auto", norm=norm)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, heatmap_data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heatmap_data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove the y label and the x label
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Add text labels for each cell
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[i, j]
            # if value is -0. ... do not put a negative sign
            if abs(value) < 1: 
                value = abs(value)

            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=16,
                color="black",
            )

    # Add text labels for best_match and query_consensus
    for i, row in cluster_table.iterrows():
        ax.text(
            heatmap_data.shape[1] - 0.4,
            i,
            row["best_match"],
            fontsize=12,
            va="center",
            ha="left",
            color="black",
        )
        ax.text(
            heatmap_data.shape[1] - 0.4,
            i + 0.2,
            row["query_consensus"][:10],
            fontsize=8,
            va="center",
            ha="left",
            color="black",
        )

    # Set axis labels and ticks
    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticklabels(heatmap_data.columns, ha="center", fontsize=16)
    ax.set_yticks(range(heatmap_data.shape[0]))
    ax.set_yticklabels(range(1, heatmap_data.shape[0] + 1), fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Cluster", fontsize=16)

    plt.savefig("Images/figure_5d.pdf", bbox_inches="tight")

def figure_S2(figsize=(10, 7)):
    '''
    Compare performance between hormones and PTI for AUC and MCC
    '''
    res = pd.read_csv("Results/Results_table.csv")
    # Change the "in_type" column name to "Model type"
    df = res.rename(columns={"in_type": "Model type"})
    
    
    hormone_treatments = ["MeJA", "SA", "SA+MeJA", "ABA"]
    df["treatment_group"] = df["treatment"].apply(lambda x: "Hormone" if x in hormone_treatments else "PTI")

    # Filter the DataFrame to the relevant subset for plotting
    filtered_df = df[
        (df["outcome_type"].isin(["DE_per_treatment", "quantiles_per_treatment"])) &
        (df["metric"].isin(["AUC", "MCC"]))
    ].copy()

    # Create treatment groups
    hormones = ["MeJA", "SA", "SA+MeJA", "ABA"]
    filtered_df["treatment_group"] = filtered_df["treatment"].apply(lambda x: "Hormone" if x in hormones else "PTI")

    # Plot using seaborn
    plt.figure(figsize=figsize, dpi=300)
    g = sns.catplot(
        data=filtered_df,
        x="metric",
        y="value",
        hue="treatment_group",
        col="outcome_type",
        kind="box",
        palette="Set2",
        height=5,
        aspect=1
    )

    g.set_axis_labels("Metric", "Value")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig("Images/SUP_figure_2.pdf", bbox_inches="tight")

def figure_S3(figsize = (10, 7)):
    '''
    Heatmaps of overlaps of sensitive genes for different treatments
    '''
    DNA_specification = [814, 200, 200, 814]
    treatments = ["B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"]
    

    for outcome in ["DE_per_treatment", "quantiles_per_treatment"]:
        (mRNA_train,
        mRNA_validation,
        mRNA_test,
        _,
        _,
        _) = load_data(train_proportion=0.85,
                            val_proportion=0.05,
                            DNA_specs=DNA_specification,
                            treatments=treatments,
                            problem_type=outcome,
                            mask_exons=False,
                            dna_format="String")
        # Concatenate al mRNA data
        mRNA = pd.concat([mRNA_train, mRNA_validation, mRNA_test])

        print(mRNA)
        # Make a heatmap of the intersection of sensitive genes divided by the union of sensitive genes
        # for the different treatments
        overlaps = np.zeros((len(treatments), len(treatments)))
        for t1 in treatments:
            for t2 in treatments:
                mRNA_t1 = mRNA[t1]
                mRNA_t2 = mRNA[t2]
                mask = (mRNA_t1 != 3) & (mRNA_t2 != 3)
                intersection = np.sum(((mRNA_t1 == 1) & (mRNA_t2 == 1))[mask])
                union = np.sum((((mRNA_t1 == 1) | (mRNA_t2 == 1)))[mask])
                overlaps[treatments.index(t1), treatments.index(t2)] = intersection / union

        plt.figure(figsize=figsize)
        sns.heatmap(
            overlaps,
            cmap="coolwarm",
            annot=True,
            cbar=False,
            annot_kws={"size": 8, "color": "white", "weight": "bold"},
            fmt=".2f",
            linewidths=0.5,
            linecolor="black",
            vmax=1,
            vmin=-1,
        )
        # reduce the fontsize of the numbers inside the heatmap

        # add the names of the treatments
        plt.xticks(
            np.arange(len(treatments)) + 0.5,
            labels=[mapping[t] for t in treatments],
            rotation=40,
            fontsize=10,
        )
        plt.yticks(
            np.arange(len(treatments)) + 0.5,
            labels=[mapping[t] for t in treatments],
            rotation=40,
            fontsize=10,
    )
        # remove colormap
        plt.savefig(f"Images/SUP_figure_3_{outcome}.pdf", bbox_inches="tight")
        
def figure_S4(figsize = (10, 7)):
    '''
    Heatmaps of relative correlations for LFCT and LFCA
    '''
    DNA_specs = [814, 200, 200, 814]
    treatments = ["B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"]

    for outcome in ["log2FC", "amplitude" ]:
        (mRNA_train,
        mRNA_validation,
        mRNA_test,
        _,
        _,
        _) = load_data(train_proportion=0.85,
                            val_proportion=0.05,
                            DNA_specs=DNA_specs,
                            treatments=treatments,
                            problem_type=outcome,
                            mask_exons=False,
                            dna_format="One_Hot_Encoding")
    
        mRNA = pd.concat([mRNA_train, mRNA_validation, mRNA_test])
        
        Y_hat = mRNA.values[:, 1:-1]
        correlation_matrix = stats.spearmanr(Y_hat).statistic
        # plot the correlation matrix
        plt.figure(figsize=figsize, dpi=300)
        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            annot=True,
            cbar=False,
            annot_kws={"size": 8, "color": "white", "weight": "bold"},
            fmt=".2f",
            linewidths=0.5,
            linecolor="black",
            vmax=1,
            vmin=-1,
        )
        # reduce the fontsize of the numbers inside the heatmap

        # add the names of the treatments
        plt.xticks(
            np.arange(len(treatments)) + 0.5,
            labels=[mapping[t] for t in treatments],
            rotation=40,
            fontsize=10,
        )
        plt.yticks(
            np.arange(len(treatments)) + 0.5,
            labels=[mapping[t] for t in treatments],
            rotation=40,
            fontsize=10,
    )
        # remove colormap
        plt.savefig(f"Images/SUP_figure_4_{outcome}.pdf", bbox_inches="tight")
        
def figure_S6(figsize=(10, 7)):
    for outcome in ["log2FC", "amplitude"]:
        # Initialize figure
        plt.figure(figsize=figsize, dpi=300)

        # Read and concatenate data
        my_concat = pd.DataFrame()

        # CNN Model
        for i in range(0, 5):
            metrics = pd.read_csv(
                f"Results/CNN/{outcome}/2048/exons_masked_False/model_{i}/test_metrics.csv",
                index_col=0,
            )
            metric_m = pd.DataFrame(metrics.T)
            metric_m["replicate"] = i
            metric_m["model"] = "CNN"
            my_concat = pd.concat([my_concat, metric_m])

        # AgroNT Model
        metrics = pd.read_csv(f"Results/agroNT/{outcome}/test_metrics.csv", index_col=0)
        metric_m = pd.DataFrame(metrics.T)
        metric_m["replicate"] = 0
        metric_m["model"] = "AgroNT"
        my_concat = pd.concat([my_concat, metric_m])

        # 6-mer
        for file in os.listdir(f"Results/linear_models/{outcome}/6-mer"):
            if "metrics" not in file:
                continue
            treatment_name = file.split("_")[0]
            metrics = pd.read_csv(
                f"Results/linear_models/{outcome}/6-mer/{file}", index_col=0
            )
            metric_m = pd.DataFrame(metrics)
            metric_m["replicate"] = 0
            metric_m["model"] = "6-mer"
            metric_m["treatment"] = treatment_name
            # reset index
            metric_m.reset_index(inplace=True)
            # set index treatment
            metric_m.set_index("treatment", inplace=True)
            my_concat = pd.concat([my_concat, metric_m])

        # DAPseq
        for file in os.listdir(f"Results/linear_models/{outcome}/DAPseq/"):
            if "metrics" not in file:
                continue
            treatment_name = file.split("_")[0]
            metrics = pd.read_csv(
                f"Results/linear_models/{outcome}/DAPseq/{file}", index_col=0
            )
            metric_m = pd.DataFrame(metrics)
            metric_m["replicate"] = 0
            metric_m["model"] = "DAPseq"
            metric_m["treatment"] = treatment_name
            # reset index
            metric_m.reset_index(inplace=True)
            # set index treatment
            metric_m.set_index("treatment", inplace=True)
            my_concat = pd.concat([my_concat, metric_m])

        # Reset index
        my_concat.reset_index(inplace=True)
        my_concat.rename(columns={"index": "Metric"}, inplace=True)
        # apply mapping to the treatment
        my_concat["treatment"] = my_concat["Metric"].replace(mapping)

        # Boxplot for CNN
        sns.boxplot(
            data=my_concat[my_concat["model"] == "CNN"],
            x="treatment",
            y="Spearman",
            color="#2ca02c",
            width=0.6,
            showfliers=False,
        )

        # Stripplot (dots) for all other models
        strip = sns.stripplot(
            data=my_concat[
                (my_concat["model"] != "CNN") & (my_concat["model"] != "Random")
            ],
            x="treatment",
            y="Spearman",
            hue="model",
            dodge=True,
            jitter=True,
            alpha=0.8,
            linewidth=0.5,
            size=10,
            palette={
                "6-mer": "#1f77b4",
                "DAPseq": "#ff7f0e",
                "AgroNT": "#d62728",
                "Random": "#999999",
            },
        )

        plt.xticks(rotation=45, ha="right")
    
        plt.ylabel("Spearman Correlation")
        
        plt.xlabel("Treatment")
        strip.spines["top"].set_visible(False)
        strip.spines["right"].set_visible(False)
    
        # Manually create a legend including CNN
        handles, labels = strip.get_legend_handles_labels()
        cnn_patch = mpatches.Patch(color="#2ca02c", label="CNN")
        handles.insert(0, cnn_patch)
        labels.insert(0, "CNN")
        # Add vertical grid lines between treatments
        # Add customized vertical grid lines between treatments
        ax = plt.gca()
        treatments = my_concat["treatment"].unique()
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        
        # Calculate positions for vertical gridlines (offset by 0.5)
        for i in range(len(treatments)+1):
            ax.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

        plt.legend(
            handles=handles,
            labels=labels,
            title="Model",
            loc="upper right",
            bbox_to_anchor=(0.19, 1) if outcome == "log2FC" else (1., 0.45),
            frameon=False if outcome == "log2FC" else True,)

        # plt.title("Comparison of Test Metrics Across Models")
        plt.tight_layout()
        plt.savefig(f"Images/SUP_figure_6_{outcome}.pdf", bbox_inches="tight")

def figure_S7(figsize=(10, 7)):
    '''
    boxplot performance per treatment no PTI
    '''
    # Initialize figure
    plt.figure(figsize=figsize, dpi=300)

    # Read and concatenate data
    my_concat = pd.DataFrame()

    # CNN Model
    for i in range(0, 5):
        metrics = pd.read_csv(
            f"Results/CNN_no_PTI/model_{i}/test_metrics.csv",
            index_col=0,
        )
        metric_m = pd.DataFrame(metrics.T)
        metric_m["replicate"] = i
        metric_m["model"] = "CNN"
        my_concat = pd.concat([my_concat, metric_m])

 
    # Reset index
    my_concat.reset_index(inplace=True)
    my_concat.rename(columns={"index": "Metric"}, inplace=True)
    # apply mapping to the treatment
    my_concat["treatment"] = my_concat["Metric"].replace(mapping)

    # Boxplot for CNN
    sns.boxplot(
        data=my_concat[my_concat["model"] == "CNN"],
        x="treatment",
        y="Spearman",
        color="#2ca02c",
        width=0.6,
        showfliers=False,
    )

    # Stripplot (dots) for all other models
    strip = sns.stripplot(
        data=my_concat[
            (my_concat["model"] != "CNN") & (my_concat["model"] != "Random")
        ],
        x="treatment",
        y="Spearman",
        hue="model",
        dodge=True,
        jitter=True,
        alpha=0.8,
        linewidth=0.5,
        size=10,
        palette={
            "6-mer": "#1f77b4",
            "DAPseq": "#ff7f0e",
            "AgroNT": "#d62728",
            "Random": "#999999",
        },
    )

    plt.xticks(rotation=45, ha="right")

    plt.ylabel("Spearman Correlation")

    plt.xlabel("Treatment")
    strip.spines["top"].set_visible(False)
    strip.spines["right"].set_visible(False)

    # Manually create a legend including CNN
    handles, labels = strip.get_legend_handles_labels()
    cnn_patch = mpatches.Patch(color="#2ca02c", label="CNN")
    handles.insert(0, cnn_patch)
    labels.insert(0, "CNN")
    # Add vertical grid lines between treatments
    # Add customized vertical grid lines between treatments
    ax = plt.gca()
    treatments = my_concat["treatment"].unique()
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Calculate positions for vertical gridlines (offset by 0.5)
    for i in range(len(treatments)+1):
        ax.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    plt.legend(
        handles=handles,
        labels=labels,
        title="Model",
        loc="upper right")

    # plt.title("Comparison of Test Metrics Across Models")
    plt.tight_layout()
    plt.savefig(f"Images/SUP_figure_7.pdf", bbox_inches="tight")

if __name__ == "__main__":
    set_plot_style()
    figure_1a()
    ##figure_1b() Data for this is not made publicly available (is figure 1c in the paper)
    
    figure_2a()
    figure_2b()
    
    figure_5c(outcome = "quantiles_per_treatment") # 2.1
    
    figure_3c()
    figure_3a()
    figure_3b()
    
    figure_4a()
    figure_4b()
    figure_4c()
    figure_4d()
    
    figure_5a()
    figure_5b()
    figure_5c()

    ## Reset for last figure
    sns.reset_defaults()
    sns.set_theme()
    mpl.rcParams.update(mpl.rcParamsDefault)
    figure_5d()
    #SUP figures
    set_plot_style()
    figure_2a(metric="MCC") # SUP 1
    figure_S2()
    figure_S3()
    figure_S4(figsize=(10, 7))
    figure_3b(outcome="amplitude") # SUP 5
    figure_S6(figsize=(10, 7))
    figure_S7(figsize=(10, 7))
    