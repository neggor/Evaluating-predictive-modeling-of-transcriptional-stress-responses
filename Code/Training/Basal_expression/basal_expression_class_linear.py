import sys
sys.path.append(".")
import pandas as pd
from Code.Training.get_performance import load_data
from Code.DNA_processing.get_representation import load_promoters, generate_kmer_count_vector
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from Code.DNA_processing.get_gene_families import add_all_genes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve, 
    average_precision_score, ConfusionMatrixDisplay
)
from Code.dataset_utils.GenerateDataSplits import DataHandler
from Code.Training.train_linear import handle_data_train_linear_models, handle_data_test_linear_models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
if __name__ == "__main__":
    DNA_specification = [500, 100, 50, 15]
    gene_families_file = "Data/Processed/gene_families.csv"
    data_path = "Data/Processed"
    train_proportion = 0.85
    validation_proportion = 0
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
            treatments=["B"], # does not matter, just put one
            problem_type="TPM_cuartiles",
        )
    )
    X, Y, mean, std, family = handle_data_train_linear_models(
        TSS_sequences,
        TTS_sequences,
        mRNA_train,
        mRNA_validation,
        DNA_format= dna_format,
        separated_segments=True,
    )
    
    X_test, Y_test = handle_data_test_linear_models(
        TSS_sequences,
        TTS_sequences,
        mRNA_test,
        DNA_format=dna_format,
        means=mean,
        std=std,
        separated_segments=True,
    )

    # Define logistic regression with L1 penalty
    logreg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, verbose=1)

    # Grid of C values to search (inverse of regularization strength)
    param_grid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]}

    # 5-fold cross-validation for hyperparameter tuning
    grid = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose = 2)
    grid.fit(X, Y.ravel())

    # Best model
    best_model = grid.best_estimator_
    print("Best C:", grid.best_params_)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    print(confusion_matrix(Y_test, y_pred))
    


    accuracy = accuracy_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")


    metrics_path = "Basal/metrics_class.csv"
    results_row = pd.DataFrame([{
        "Model": "Linear",
        "Accuracy": accuracy,
        "AUC": auc,
        "F1": f1
    }])

    try:
        # Append if file exists
        existing = pd.read_csv(metrics_path)
        updated = pd.concat([existing, results_row], ignore_index=True)
    except FileNotFoundError:
        # Create new if doesn't exist
        updated = results_row

    updated.to_csv(metrics_path, index=False)
    print(f"Logged metrics to {metrics_path}")


    # Save .csv of the coefficients
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_model.coef_.ravel()
    }).sort_values(by='Coefficient', ascending=False)
    coef_df.to_csv("Basal/logreg_coefficients.csv", index=False)