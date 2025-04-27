import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

mRNA_data_matrix = pd.read_csv("Data/Processed/mRNA/DESeq2_padj_results_ALL.csv")
mRNA_data_matrix = mRNA_data_matrix.dropna(subset=["padj"])
my_values = {}

for problem_type in [
    "log2FC",
    "amplitude",
    "quantiles_per_treatment",
    "DE_per_treatment",
]:
    
    if problem_type in ['amplitude', 'log2FC']:
        # Linear regression
        X = sm.add_constant(mRNA_data_matrix["average"].values)
        y = mRNA_data_matrix["Log2_Fold_Change"].values if problem_type == 'log2FC' else mRNA_data_matrix["amplitude"].values
        model = sm.OLS(y, X).fit()
        # Print the summary
        print(model.summary())
        # extract F-statistic and p-value
        f_statistic = model.fvalue
        p_value = model.f_pvalue
        # insert into dictionary
        my_values[problem_type] = {}
        my_values[problem_type]["Coefficient"] = model.params[1]  # Coefficient of the independent variable
        my_values[problem_type]["p-value"] = model.pvalues[1]  # p-value for the coefficient
        # calculate the Spearman corelation coefficient between the predicted and actual values
        spearman_corr = stats.spearmanr(model.predict(X), y)
        my_values[problem_type]["Spearman correlation"] = spearman_corr.correlation
    elif problem_type == 'DE_per_treatment':
        fdr = 0.01
        print("FDA threshold: ", fdr)
        mRNA_data_matrix["padj"] = mRNA_data_matrix["padj"].fillna(1)
        mRNA_data_matrix["class"] = 3
        for h in mRNA_data_matrix["treatment"].unique():
            mRNA_data_matrix.loc[mRNA_data_matrix["treatment"] == h, "class"] = 0
            mRNA_data_matrix.loc[
                (mRNA_data_matrix["treatment"] == h)
                & (mRNA_data_matrix["padj"] < fdr),
                "class",
            ] = 1

        # Logistic regression
        X = sm.add_constant(mRNA_data_matrix["average"].values)
        y = mRNA_data_matrix["class"].values
        model = sm.Logit(y, X).fit()
        # Print the summary
        print(model.summary())
        # insert into dictionary
        my_values[problem_type] = {}
        my_values[problem_type]["Coefficient"] = model.params[1]  # Coefficient of the independent variable
        my_values[problem_type]["p-value"] = model.pvalues[1]  # p-value for the coefficient
        # calculate the AUC
        auc = roc_auc_score(y, model.predict(X))
        my_values[problem_type]["AUC"] = auc

    elif problem_type == 'quantiles_per_treatment':
        fdr = 0.01
        for h in mRNA_data_matrix["treatment"].unique():
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
        mask = mRNA_data_matrix["class"] != 3
        X = sm.add_constant(mRNA_data_matrix["average"].values[mask])
        y = mRNA_data_matrix["class"].values[mask]
        model = sm.Logit(y, X).fit()
        # Print the summary
        print(model.summary())
        # insert into dictionary
        my_values[problem_type] = {}
        my_values[problem_type]["Coefficient"] = model.params[1]  # Coefficient of the independent variable
        my_values[problem_type]["p-value"] = model.pvalues[1]  # p-value for the coefficient
        # calculate the AUC
        auc = roc_auc_score(y, model.predict(X))
        my_values[problem_type]["AUC"] = auc
        

# Now generate a dataframe (if no value, just NA)
# make a nice latex table
df = pd.DataFrame(my_values).T
df = df.fillna("NA")
print(df.to_latex())
# save to results 
df.to_csv("Results/Average_BIAS_tests.csv")


    

#print(mRNA_train)
#print(mRNA_train.shape, mRNA_validation.shape, mRNA_test.shape)