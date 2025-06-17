import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from scipy import stats
import numpy as np
import pickle
import pandas as pd
import os
import sys
sys.path.append(".")
from Code.Training.train_linear import handle_data_test_linear_models, handle_data_train_linear_models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier





def cross_validate(X, y, family, task="regression", cv=3, scoring=None, n_jobs=12):
    """
    Perform cross-validation for RandomForestRegressor or RandomForestClassifier,
    tuning n_estimators, max_depth, and min_samples_split.

    Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        family (array-like): Group labels for GroupKFold splitting.
        task (str): "regression" or "classification".
        cv (int): Number of cross-validation folds.
        scoring (str or None): Scoring metric to optimize.
        n_jobs (int): Number of parallel jobs (-1 uses all CPUs).

    Returns:
        best_model (sklearn estimator): Best estimator after grid search.
        best_params (dict): Best hyperparameters.
        grid_search (GridSearchCV object): Full grid search results.
    """

    if task == "regression":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 14, 20],
            "min_samples_split": [2, 5, 10]
        }
        if scoring is None:
            scoring = "r2"
    elif task == "classification":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 14, 20],
            "min_samples_split": [2, 5, 10]
        }
        if scoring is None:
            scoring = "roc_auc"
    else:
        raise ValueError("Invalid task. Choose 'regression' or 'classification'.")

    gcv = GroupKFold(n_splits=cv)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=gcv.split(X, y, groups=family),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search


def fit_rf(
    X, Y, family, model, folder, name, metadata, DNA_format, separated_segments=True
):
    """
    Performs cross-validation to optimize sparsity penalty for Lasso regression
    or lasso logistic regression.

    Returns the best model and prints the results of the best model. Stores the best model as a pickle file.
    """
    assert Y.shape[1] == 1, "Y must have only one column"  # TODO
    print(f"Fitting {model} regression")
    print("Y has shape", Y.shape)
    print("X has shape", X.shape)

    print("=============================================")
    print("TRAIN")
    print("=============================================")
    if model == "regression":
        reg, best_params, grid_search = cross_validate(
            X, Y.ravel(), family, task="regression", cv=5, n_jobs=20
        )
        print(best_params)
        Y_pred = reg.predict(X)
        print(f"R2 score: {r2_score(Y, Y_pred)}")

    elif model == "classification":
        mask = Y.ravel() == 3
        Y = Y[~mask]
        X = X[~mask]
        family = np.array(family)[~mask]
        reg, best_params, grid_search = cross_validate(
            X, Y.ravel(), family, task="classification", cv=5, n_jobs=20
        )
        Y_pred = reg.predict(X)
        print(best_params)
        print(f"Accuracy: {accuracy_score(Y, Y_pred)}")
        print("MCC:", matthews_corrcoef(Y, Y_pred))
        print("F1:", f1_score(Y, Y_pred, average="weighted"))
    else:
        raise ValueError("Model must be either 'regression' or 'classification'")

    print("=============================================")

    # Save the model
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/{name}_linear_model.pkl", "wb") as f:
        pickle.dump(reg, f)

    return reg


def test_rf(X_test, Y_test, reg, model, folder, name, metadata=None):
    """
    Tests the RandomForest model on the test set and prints the results.

    Parameters:
        X_test: Test features.
        Y_test: Test target.
        reg: Trained RandomForestRegressor or RandomForestClassifier.
        model (str): "regression" or "classification".
        folder (str): Path to save predictions and metrics.
        name (str): File prefix for saved files.
        metadata: (Optional) Extra info.
    """
    print("=============================================")
    print("TEST")
    print("=============================================")

    os.makedirs(folder, exist_ok=True)
    metrics = {}

    if model == "regression":
        Y_pred = reg.predict(X_test)
        metrics["R2"] = r2_score(Y_test, Y_pred)
        metrics["Pearson"] = np.corrcoef(Y_test.ravel(), Y_pred.ravel())[0, 1]
        metrics["Spearman"] = stats.spearmanr(Y_test.ravel(), Y_pred.ravel()).statistic
        metrics["sign_accuracy"] = np.mean(np.sign(Y_test.ravel()) == np.sign(Y_pred.ravel()))

        print(f"R2: {metrics['R2']:.3f}")
        print(f"Pearson: {metrics['Pearson']:.3f}")
        print(f"Spearman: {metrics['Spearman']:.3f}")
        print(f"Sign accuracy: {metrics['sign_accuracy']:.3f}")

    elif model == "classification":
        mask = Y_test.ravel() == 3  # if applicable
        Y_test = Y_test[~mask]
        X_test = X_test[~mask]
        Y_pred_prob = reg.predict_proba(X_test)[:, 1]
        Y_pred_bin = Y_pred_prob > 0.5

        metrics["Accuracy"] = accuracy_score(Y_test, Y_pred_bin)
        metrics["MCC"] = matthews_corrcoef(Y_test, Y_pred_bin)
        metrics["F1"] = f1_score(Y_test, Y_pred_bin)
        metrics["AUC"] = roc_auc_score(Y_test, Y_pred_prob)

        print(f"Accuracy: {metrics['Accuracy']:.3f}")
        print(f"MCC: {metrics['MCC']:.3f}")
        print(f"F1: {metrics['F1']:.3f}")
        print(f"AUC: {metrics['AUC']:.3f}")

        # Save probabilities instead of binary preds
        np.save(f"{folder}/{name}_predictions.npy", Y_pred_prob)

    else:
        raise ValueError("Model must be either 'regression' or 'classification'")

    # Save metrics
    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(f"{folder}/{name}_metrics.csv", index=False)