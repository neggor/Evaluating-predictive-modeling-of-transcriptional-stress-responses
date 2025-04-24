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

magentaorgange_palette = ["#D35FB7", "#FF8C00"]
coraldarkteal_palette = ["#FF7F50", "#2CA58D"]


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


def _figure_1(
    figsize=(10, 7), fitted_values_file="Data/Processed/mRNA/fitted_values_PTI_X.csv"
):

    X = pd.read_csv(fitted_values_file)
    # select a subset of genes and make a nice scatterplot on time
    # set Unnamed: 0 name as "gene"
    X = X.rename(columns={"Unnamed: 0": "gene"})
    X.set_index("gene", inplace=True)
    X = X.sample(n=100, random_state=42)
    X_control = X.filter(like="control")
    # select columns that have treatment
    treatment = fitted_values_file.split("/")[-1].split("_")[-1].split(".")[0]
    X_treatment = X.filter(like=treatment)
    # Random gene selection
    treatment = fitted_values_file.split("/")[-1].split("_")[-1].split(".")[0]
    X_treatment = X.filter(like=treatment)
    # put in long format, with the columns as features.
    # They ae separated by _, being treatment, time, replication
    X_control = pd.melt(
        X_control.reset_index(),
        id_vars=["gene"],
        var_name="features",
        value_name="fitted_values",
    )
    X_control["treatment"] = "control"
    X_treatment = pd.melt(
        X_treatment.reset_index(),
        id_vars=["gene"],
        var_name="features",
        value_name="fitted_values",
    )
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
    X = X.groupby(["treatment", "time", "gene"]).mean().reset_index()
    # set the time as int
    X["time"] = X["time"].astype(int)
    # now make a lineplot for treatment and control
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for gene in X["gene"].unique():
        X_gene_treatment = X[(X["gene"] == gene) & ((X["treatment"] != "control"))]
        X_gene_control = X[(X["gene"] == gene) & ((X["treatment"] == "control"))]
        fold_change = (
            X_gene_treatment["fitted_values"].values
            / X_gene_control["fitted_values"].values
        )
        # make a new dataframe with the time and the fold change
        X_gene_treatment = X_gene_treatment.copy()
        X_gene_treatment["fitted_values"] = np.log2(fold_change)
        # make the lineplot
        #  add a column for the ration
        sns.lineplot(data=X_gene_treatment, x="time", y="fitted_values", ax=ax)

    plt.show()


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
    plt.savefig("Images/figure_2a.pdf", bbox_inches="tight")


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
    treatments = ["B", "C", "D", "G", "H", "X", "Y", "Z", "W", "V", "U", "T"]
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

    # Set overall x and y labels
    fig.text(0.5, 0.02, r"Predicted FC.T", ha="center", fontsize=12)
    fig.text(0.02, 0.5, r"True FC.T", va="center", rotation="vertical", fontsize=12)

    plt.tight_layout(
        rect=[0.03, 0.03, 1, 0.95], h_pad=1.6, w_pad=1.6
    )  # Adjust layout to fit suptitle and increase horizontal separation
    # Save the figure
    plt.savefig(f"Images/figure_3a.pdf", bbox_inches="tight")


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

    # Simulate random model
    # DNA_specs = [814, 200, 200, 814]
    # treatments = ["B", "C", "D", "G", "H", "X", "Y", "Z", "W", "V", "U", "T"]
    # (
    #    mRNA_train,
    #    mRNA_validation,
    #    mRNA_test,
    #    TSS_sequences,
    #    TTS_sequences,
    #    metadata,
    # ) = load_data(
    #    train_proportion=0.80,
    #    val_proportion=0.1,
    #    DNA_specs=DNA_specs,
    #    treatments=treatments,
    #    problem_type=outcome,
    #    mask_exons=False,
    #    dna_format="One_Hot_Encoding",
    # )
    #
    #
    # random_results = []
    # np.random.seed(42)  # for reproducibility
    # for treatment in treatments:
    #    Y = mRNA_test[treatment]
    #    # get the sign
    #    Y_sign = Y.apply(lambda x: 1 if x > 0 else 0)
    #    for replicate in range(5):
    #        correct = np.random.binomial(
    #            n=Y.shape[0],
    #            p=0.5,
    #        )
    #        # calculate the accuracy
    #        acc = correct / Y.shape[0]
    #        random_results.append({
    #            "sign_accuracy": acc,
    #            "replicate": replicate,
    #            "model": "Random",
    #            "treatment": mapping[treatment],
    #    })
    #
    # random_df = pd.DataFrame(random_results)
    # my_concat = pd.concat([my_concat, random_df], ignore_index=True)

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

    # sns.boxplot(
    #    data=my_concat[my_concat["model"] == "Random"],
    #    x="treatment",
    #    y="sign_accuracy",
    #    color="#999999",
    #    width=0.2,
    #    showfliers=False,
    # )

    plt.xticks(rotation=45, ha="right")
    if outcome == "log2FC":
        plt.ylabel("LFC.T direction correctly predicted (proportion)")
    else:
        plt.ylabel("LFC.A direction correctly predicted (proportion)")
    plt.xlabel("Treatment")
    strip.spines["top"].set_visible(False)
    strip.spines["right"].set_visible(False)
    # remove the grid
    plt.grid(color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    # Manually create a legend including CNN
    handles, labels = strip.get_legend_handles_labels()
    cnn_patch = mpatches.Patch(color="#2ca02c", label="CNN")
    handles.insert(0, cnn_patch)
    labels.insert(0, "CNN")
    # plt.ylim(0, 1)
    plt.legend(
        handles=handles,
        labels=labels,
        title="Model",
        loc="upper right",
        bbox_to_anchor=(1, 0.55),
    )

    # plt.title("Comparison of Test Metrics Across Models")
    plt.tight_layout()
    plt.savefig(f"Images/figure_3b.pdf", bbox_inches="tight")


def figure_3c(figsize=(10, 7), outcome="log2FC"):
    DNA_specs = [814, 200, 200, 814]
    treatments = ["B", "C", "D", "G", "H", "X", "Y", "Z", "W", "V", "U", "T"]
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
    plt.savefig(f"Images/figure_3c.pdf", bbox_inches="tight")


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
    plt.ylabel(f"{metric}")
    plt.xlabel("")
    plt.legend(loc="upper center", frameon=False, ncol=1)
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
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
    plt.legend(loc="upper center", frameon=False, ncol=1)
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    # save the figure
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
    plt.ylabel(f"{metric}")

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
    plt.legend(loc="upper center", frameon=False, ncol=1)
    plt.grid(axis="y", color="black", alpha=0.3, linestyle="--", linewidth=0.5)
    # save the figure
    plt.savefig("Images/figure_4d.pdf", bbox_inches="tight")

def figure_5a(figsize=(10, 7)):
    '''
    PCA coefficients 6-mer log2fc
    '''

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

    for hormone in ["MeJA", "SA", "SA+MeJA", "ABA", "ABA+MeJA"]:
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
    top_n = 12
    important_points = pca_df.abs().nlargest(top_n, "PC1").index.tolist() + pca_df.abs().nlargest(top_n, "PC2").index.tolist()

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

        for hormone in ["MeJA", "SA", "SA+MeJA", "ABA", "ABA+MeJA"]:
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
        # multiply x axis by -1
        coefs_pca[:, 0] = coefs_pca[:, 0] * -1
        # Create DataFrame for plotting
        pca_df = pd.DataFrame(coefs_pca, columns=["PC1", "PC2"], index=coefs.index)

        # Select top 100 most variable treatments (highest absolute variance in PC1 or PC2)
        top_n = 12
        important_points = pca_df.abs().nlargest(top_n, "PC1").index.tolist() + pca_df.abs().nlargest(top_n, "PC2").index.tolist()

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
        # multiply x axis by -1
        loadings[0, :] = loadings[0, :] * -1
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

def figure_5c(figsize=(10, 7)):
    pass

def figure_5d(figsize=(14, 10)):
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

    # Adjust layout and save the figure
    #plt.tight_layout()
    # Add a colorbar to indicate the relationship between color and values
    #cbar = fig.colorbar(heatmap, ax=ax, orientation="vertical", pad=0.1)
    #cbar.set_label("Number of Positive/Negative Seqlets", fontsize=14)
    #cbar.ax.tick_params(labelsize=12)
    plt.savefig("Images/figure_5d.pdf", bbox_inches="tight")

if __name__ == "__main__":
    set_plot_style()
    #figure_1a()
    #figure_1a()
    #figure_1b()
    #figure_2a()
    #figure_2b()
    #figure_3c()
    #figure_3a()
    #figure_3b()
    #figure_4a()
    #figure_4b()
    #figure_4c()
    #figure_4d()
    figure_5a()
    figure_5b()
    # Reset for last figure
    #sns.reset_defaults()
    #sns.set_theme()
    #mpl.rcParams.update(mpl.rcParamsDefault)
    #figure_5d()