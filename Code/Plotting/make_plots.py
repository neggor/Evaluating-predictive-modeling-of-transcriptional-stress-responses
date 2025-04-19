import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from statannotations.Annotator import Annotator
import numpy as np

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

def set_plot_style():
    # Set Seaborn style + Matplotlib overrides
    sns.set_style("whitegrid")  # Background style (choose one)
    sns.set_context("paper")   # Scale fonts for paper ("paper", "notebook", "talk", "poster")

    # Override Matplotlib defaults for consistency
    mpl.rcParams.update({
        #'font.family': 'serif',          # Use serif (e.g., Times New Roman)
        #'font.serif': ['Times New Roman'],
        'font.size': 10,                 # Base font
        'axes.titlesize': 10,            # Title size
        'axes.labelsize': 10,            # Axis labels
        'xtick.labelsize': 8,            # Tick labels
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,               # High-res figures
        'savefig.dpi': 300,
        'savefig.format': 'pdf',         # Vector format
        'savefig.bbox': 'tight',
    })

    sns.set_palette("deep")  # Set default color palette
    sns.color_palette("viridis", as_cmap=True)

def figure_1a(figsize = (10, 7),fitted_values_file = "Data/Processed/mRNA/fitted_values_PTI_X.csv"):
    
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
    X_control = pd.melt(X_control.reset_index(), id_vars=["gene"], var_name="features", value_name="fitted_values")
    X_control["treatment"] = "control"
    X_treatment = pd.melt(X_treatment.reset_index(), id_vars=["gene"], var_name="features", value_name="fitted_values")
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
        #import pdb; pdb.set_trace()
        X_gene_treatment = X[(X["gene"] == gene) & ((X["treatment"] != "control"))]
        X_gene_control = X[(X["gene"] == gene) & ((X["treatment"] == "control"))]
        fold_change = X_gene_treatment["fitted_values"].values / X_gene_control["fitted_values"].values
        # make a new dataframe with the time and the fold change
        X_gene_treatment = X_gene_treatment.copy()
        X_gene_treatment["fitted_values"] = np.log2(fold_change)
        # make the lineplot
        #  add a column for the ration
        sns.lineplot(data=X_gene_treatment, x="time", y="fitted_values", ax=ax)
    
    plt.show()
        
def figure_1b(figsize = (10, 7)):
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
    res.loc[res["stat"] < quantiles[0.25], "class"] = 'Below Lower Quartile'
    res.loc[res["stat"] > quantiles[0.75], "class"] = 'Above Upper Quartile'
    # Make histogram with the hlines for the quartiles and the stat_001, put colors on the bins of each class
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # Define custom colors for the classes with stronger shades
    colors = {
        "Neutral": "lightgrey",
        "Below Lower Quartile": "dimgray",
        "Above Upper Quartile": "dimgray"
    }
    # Plot the histogram with custom colors
    sns.histplot(res, x="stat", bins=200, hue="class", palette=colors, legend=False)
    #sns.move_legend(ax, "upper right", bbox_to_anchor=(1, 1), fontsize=12)
    ax.axvline(quantiles[0.25], color='r', linestyle='dashed', linewidth=2)
    # add label for the ax line
    ax.text(quantiles[0.25] + 1, 300, "0.25 quantile", rotation=90)
    ax.axvline(quantiles[0.75], color='r', linestyle='dashed', linewidth=2)
    ax.text(quantiles[0.75] + 1, 300, "0.75 quantile", rotation=90)
    ax.axvline(stat_001, color='g', linestyle='dashed', linewidth=2)
    ax.text(stat_001 + 1, 300, "padj = 0.01", rotation=90)
    # Now, in the 100 in the x-axis put >100
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)  # Disable grid
    # remove the grid
    plt.xticks([0, 10, 20, 30, 40, 50], ["0", "10", "20", "30", "40", ">50"])
    # limit x axis to LR <= 100
    plt.ylim(0, 500)
    plt.xlim(0, 50)
    plt.xlabel("Likelihood ratio")
    plt.ylabel("Frequency")
    # put legent on the right upper
    #plt.title("Likelihood ratio between including or excluding the treatment effect \n Pep1 treatment")
    # save
    plt.savefig("Images/figure_1b.pdf", bbox_inches='tight')

def figure_2a(figsize = (10, 7), pvals = False, metric = "AUC"):
    """
    Plot the figure 2a
    """
    res = pd.read_csv("Results/Results_table.csv")
    res["in_type"] = res["in_type"].replace(
        {"One-Hot": "CNN", "DAPseq": "L. (DAPseq)", "String": "AgroNT", "6-mer": "L. (6mer)", "embeddings": "L. (AgroNT emb.)"}
    )
    res = res[(res["length"] == 2048) | (res["length"] == "not apply")]
    res = res[res["exons"] != "masked"]
    res = res[res['rc'] != 'False']
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
    res["outcome_type"] = pd.Categorical(res["outcome_type"], categories=["S.DE", "S.Q"], ordered=True)
    
    ax = sns.boxplot(x="outcome_type", y="value", data=res, hue="in_type")
    # print the average of the values per in_type
    print(res.groupby(["in_type", "outcome_type"])["value"].mean())
    sns.swarmplot(x="outcome_type", y="value", data=res, hue="in_type", dodge=True, marker=".", color=".25", legend=False)
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
            #(
            #    (res["outcome_type"].unique()[1], "CNN"),
            #    (res["outcome_type"].unique()[1], "L. (AgroNT emb.)"),
            #)

        ]

        annotator = Annotator(
            ax, data=res, x="outcome_type", y="value", hue="in_type", pairs=pairs
        )
        annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
        annotator.apply_and_annotate()
    plt.legend(loc="upper center", fontsize=10, frameon=False, ncol=1)
    plt.ylabel(f"{metric}-ROC" if metric == "AUC" else metric)
    plt.xlabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.savefig("Images/figure_2a.pdf", bbox_inches='tight')

def figure_1c(figsize = (10, 7),
               fitted_values_file = "Data/Processed/mRNA/fitted_values_PTI_X.csv",
               gene_selected = "AT3G56400"):
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
    # now make a lineplot for treatment and control
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # make the lineplot
    #  add a column for the ration
    nom = X[X["treatment"] != "control"]["fitted_values"].values
    denom = X[X["treatment"] == "control"]["fitted_values"].values
    ratio = nom / denom
    max_ratio = np.abs((ratio -1)).max()
    ratio_df = pd.DataFrame({
        "time": X[X["treatment"] != "control"]["time"].values,
        "ratio": ratio,
        "treatment": mapping[treatment],
    })
    sns.lineplot(data=X, x="time", y="fitted_values", hue="treatment", ax=ax, palette={"control": "blue", mapping[treatment]: "orange"})
    # plot the ratio only for the treatment
    ax2 = ax.twinx()  # Create a second y-axis
    sns.lineplot(data=ratio_df, x="time", y="ratio", linestyle="--", ax=ax2, color="green")



    # add shaded area below the lines
    # add the shaded area for the control
    ax.fill_between(
        X[X["treatment"] == "control"]["time"],
        X[X["treatment"] == "control"]["fitted_values"],
        color="blue",
        alpha=0.2
    )
    # add the shaded area for the treatment
    ax.fill_between(
        X[X["treatment"] == "control"]["time"],
        X[X["treatment"] != "control"]["fitted_values"],
        color="orange",
        alpha=0.2
    )

    # add the points
    sns.scatterplot(data=X, x="time", y="fitted_values", hue="treatment", ax=ax, legend=False, palette={"control": "blue", mapping[treatment]: "orange"})
    # make a red circle on top of the maximum absolute (ratio -1)
    # get the index of the maximum absolute value
    max_index = np.abs((ratio -1)).argmax()
    # get the time of the maximum absolute value
    max_time = X[X["treatment"] != "control"]["time"].values[max_index]
    # get the value of the maximum absolute value
    max_value = ratio[max_index]
    # plot the circle
    ax2.scatter(max_time, max_value, facecolors='none', edgecolors="red", s=200, linewidth=2, zorder=10)
    # add the legend
    plt.legend(loc="upper left", fontsize=10, frameon=False, ncol=1)
    ax2.set_ylabel("Ratio (Treatment/Control)", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    # x title
    ax.set_xlabel("Time (min.)")
    # y title
    ax.set_ylabel("RNA counts")
    # remove grid
    ax.grid(False)
    ax2.grid(False)
    # remove the top line
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # remove the buffer between the lines and the axes
    ax.set_ylim(-0.01, 1.2 * np.max(X["fitted_values"]))
    ax.set_xlim(np.min(X["time"]) * 0.97, 1.001 * np.max(X["time"]))


    plt.savefig("Images/figure_1c.pdf", bbox_inches='tight')

def figure_2b(figsize = (10, 7), pvals = False, metric = "Spearman"):
    res = pd.read_csv("Results/Results_table.csv")
    res["in_type"] = res["in_type"].replace(
        {"One-Hot": "CNN", "DAPseq": "L. (DAPseq)", "String": "AgroNT", "6-mer": "L. (6mer)",  "embeddings": "L. (AgroNT emb.)"}
    )

    res = res[(res["length"] == 2048) | (res["length"] == "not apply")]
    res = res[res["exons"] != "masked"]
    res = res[res['rc'] != "False"]
    res = res[
        ((res["outcome_type"] == "amplitude") | (res["outcome_type"] == "log2FC"))
        & (res["metric"] == metric)
    ]
     
    #rename log2FC to LFC.T and amplitude to LFC.A
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
    #sns.swarmplot(x="outcome_type", y="value", data=res, hue="in_type", dodge=True, color=".25", legend=False)
    sns.swarmplot(x="outcome_type", y="value", data=res, hue="in_type", dodge=True, marker=".", color=".25", legend=False)
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
            #(
            #    (res["outcome_type"].unique()[1], "CNN"),
            #    (res["outcome_type"].unique()[1], "L. (AgroNT emb.)"),
            #)

        ]

        annotator = Annotator(
            ax, data=res, x="outcome_type", y="value", hue="in_type", pairs=pairs
        )
        annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
        annotator.apply_and_annotate()
    # remove the legend altogether
    plt.legend().remove()
    #plt.legend(loc="upper center", fontsize=10, frameon=False, ncol=1)# TMP
    if metric == "Spearman":
        plt.ylabel("Spearman Correlation")
    else:
        plt.ylabel("Pearson Correlation")
    # add grid
    plt.xlabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', color='black', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.savefig("Images/figure_2b.pdf", bbox_inches='tight')

if __name__ == "__main__":
    set_plot_style()
    figure_2a()
    plt.show()