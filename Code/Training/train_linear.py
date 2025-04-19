import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from scipy import stats
import numpy as np
import pickle
import pandas as pd
import os

def handle_data_train_linear_models(TSS_sequences, TTS_sequences, mRNA_train, mRNA_validation, DNA_format, separated_segments = True):

    # of the mRNA DataFrames.

    if DNA_format == "6-mer":
        X_train_TSS = []
        X_train_TTS = []
        y_train = []
        family_id = []
        for gene in mRNA_train["Gene"]:
            try: # Try-excepts for the AgroNT embeddings
                X_train_TSS.append(TSS_sequences[gene])
                X_train_TTS.append(TTS_sequences[gene])
                y_train.append(mRNA_train[mRNA_train["Gene"] == gene].drop(columns=["Gene", "family_id"]).values[0])
                family_id.append(mRNA_train[mRNA_train["Gene"] == gene]["family_id"].values[0])
            except KeyError:
                print(f"Gene {gene} not found in the sequences")
        for gene in mRNA_validation["Gene"]:
            try:
                X_train_TSS.append(TSS_sequences[gene])
                X_train_TTS.append(TTS_sequences[gene])
                y_train.append(mRNA_validation[mRNA_validation["Gene"] == gene].drop(columns=["Gene", "family_id"]).values[0])
                family_id.append(mRNA_validation[mRNA_validation["Gene"] == gene]["family_id"].values[0])
            except KeyError:
                print(f"Gene {gene} not found in the sequences")

        X_train_TSS = np.array(X_train_TSS)
        X_train_TTS = np.array(X_train_TTS)
        
        if not separated_segments:
            X_train = np.sum([X_train_TSS, X_train_TTS], axis=0)
            X_train = X_train / np.linalg.norm(X_train, axis=1)[:, None]
        else:
            X_train = np.concatenate((X_train_TSS, X_train_TTS), axis=1)
        # Center at 0
        # remove columns where std is 0
        # (do not worry on overflows on normalization, these values will be filtered out)
        
        #old_std = X_train.std(axis=0)
        #X_train = X_train[:, old_std != 0]
        means_train = X_train.mean(axis=0)
        std_train = np.nan_to_num(X_train.std(axis=0))
        X_train = X_train - means_train
        X_train = X_train / (std_train + 1e-5) # just for the case of columns that are all 0
        assert not np.any(np.isnan(X_train)), "There are NaNs in the training data"
        y_train = np.array(y_train)


        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        return X_train, y_train, means_train, std_train, family_id

    elif DNA_format == "DAPseq":
        X_train_TSS = []
        X_train_TTS = []
        y_train = []
        family_id = []
        for gene in mRNA_train["Gene"]:
            X_train_TSS.append(TSS_sequences[gene])
            X_train_TTS.append(TTS_sequences[gene])
            y_train.append(mRNA_train[mRNA_train["Gene"] == gene].drop(columns=["Gene", "family_id"]).values[0])
            family_id.append(mRNA_train[mRNA_train["Gene"] == gene]["family_id"].values[0])
        for gene in mRNA_validation["Gene"]:
            X_train_TSS.append(TSS_sequences[gene])
            X_train_TTS.append(TTS_sequences[gene])
            y_train.append(mRNA_validation[mRNA_validation["Gene"] == gene].drop(columns=["Gene", "family_id"]).values[0])
            family_id.append(mRNA_validation[mRNA_validation["Gene"] == gene]["family_id"].values[0])
        
        # now either do an OR or just concatenate
        X_train_TSS = np.array(X_train_TSS)
        X_train_TTS = np.array(X_train_TTS)
        X_train = np.maximum(X_train_TSS, X_train_TTS) if not separated_segments else np.concatenate((X_train_TSS, X_train_TTS), axis=1)
        y_train = np.array(y_train)

        # standardize the data
        means_train = X_train.mean(axis=0)
        std_train = np.nan_to_num(X_train.std(axis=0))
        X_train = X_train - means_train
        X_train = X_train / (std_train + 1e-5)

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")


        return X_train, y_train, means_train, std_train, family_id
    
def handle_data_test_linear_models(TSS_sequences, TTS_sequences, mRNA_test, DNA_format, means, std, separated_segments = True):
    # of the mRNA DataFrames.
    #assert separated_segments, "Only separated segments are supported for now"

    if DNA_format == "6-mer":
        #raise ValueError("6-mer is not supported for now")
        X_test_TSS = []
        X_test_TTS = []
        y_test = []

        for gene in mRNA_test["Gene"]:
            try:
                X_test_TSS.append(TSS_sequences[gene])
                X_test_TTS.append(TTS_sequences[gene])
                y_test.append(mRNA_test[mRNA_test["Gene"] == gene].drop(columns=["Gene", "family_id"]).values[0])
            except KeyError:
                print(f"Gene {gene} not found in the sequences")
        
        X_test_TSS = np.array(X_test_TSS)
        X_test_TTS = np.array(X_test_TTS)
        if not separated_segments:
            X_test = np.sum([X_test_TSS, X_test_TTS], axis=0)
            X_test = X_test / np.linalg.norm(X_test, axis=1)[:, None]
        else:
            X_test = np.concatenate((X_test_TSS, X_test_TTS), axis=1)
    
        old_std = np.nan_to_num(X_test.std(axis=0))
        if np.any(old_std[std == 0] != 0):
            print("There are test column with std != 0 and std == 0 in the training data")
            # now make those columns 0
            X_test[:, std == 0] = 0
        #X_test = X_test[:, old_std != 0]
        X_test = X_test - means
        X_test = X_test / (std + 1e-5) # just for the case of columns that are all 0
        # assert that std is not 0 where X_test column is not 0
        assert not np.any(np.isnan(X_test)), "There are NaNs in the test data"
        y_test = np.array(y_test)


        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        return X_test, y_test

    elif DNA_format == "DAPseq":
        X_test_TSS = []
        X_test_TTS = []
        y_test = []
        for gene in mRNA_test["Gene"]:
            X_test_TSS.append(TSS_sequences[gene])
            X_test_TTS.append(TTS_sequences[gene])
            y_test.append(mRNA_test[mRNA_test["Gene"] == gene].drop(columns=["Gene", "family_id"]).values[0])
        
        # now either do an OR or just concatenate
        X_test_TSS = np.array(X_test_TSS)
        X_test_TTS = np.array(X_test_TTS)
        X_test = np.maximum(X_test_TSS, X_test_TTS) if not separated_segments else np.concatenate((X_test_TSS, X_test_TTS), axis=1)
        y_test = np.array(y_test)
        
        old_std = np.nan_to_num(X_test.std(axis=0))
        if np.any(old_std[std == 0] != 0):
            print("There are test column with std != 0 and std == 0 in the training data")
            X_test[:, std == 0] = 0
        # standardize the data
        X_test = X_test - means
        X_test = X_test / (std + 1e-5)
        assert not np.any(np.isnan(X_test)), "There are NaNs in the test data"
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        return X_test, y_test

def cross_validate(X, y, family, task="regression", cv=3, scoring=None, n_jobs=12):
    """
    Perform cross-validation for SGDRegressor or SGDClassifier using elastic net penalty,
    tuning alpha (regularization strength) and l1_ratio.
    
    Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        task (str): "regression" for SGDRegressor, "classification" for SGDClassifier.
        cv (int): Number of cross-validation folds.
        scoring (str or None): Scoring metric to optimize.
        n_jobs (int): Number of parallel jobs (-1 uses all CPUs).
    
    Returns:
        best_model (sklearn estimator): Best estimator after grid search.
        best_params (dict): Best hyperparameters.
        grid_search (GridSearchCV object): Full grid search results.
    """
    
   
    if task == "regression":
        param_grid = {
            "alpha":[1e-2, 1e-1, 1, 10],  # Regularization strength
            }
    
        model = Lasso(max_iter=3000, random_state=123)
        if scoring is None:
            scoring = "r2"
    elif task == "classification":
        param_grid = {
            "C": [0.05, 0.01, 0.001, 0.0001],  # Inverse regularization strength
            }
        model = LogisticRegression(penalty="l1", max_iter=3000, random_state=123, class_weight="balanced", solver="saga")
        if scoring is None:
            scoring = "roc_auc"
    else:
        raise ValueError("Invalid task. Choose 'regression' or 'classification'.")

    gcv = GroupKFold(n_splits=cv)

    grid_search = GridSearchCV(model, param_grid, cv=gcv.split(X, y, family),
                                scoring=scoring, n_jobs=n_jobs, verbose=2)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search

def fit_regression(X, Y, family, model, folder, name, metadata, DNA_format, separated_segments = True):
    '''
    Performs cross-validation to optimize sparsity penalty for Lasso regression
    or lasso logistic regression.

    Returns the best model and prints the results of the best model. Stores the best model as a pickle file.
    '''
    assert Y.shape[1] == 1, "Y must have only one column" # TODO
    print(f"Fitting {model} regression")
    print("Y has shape", Y.shape)
    print("X has shape", X.shape)

    print("=============================================")
    print("TRAIN")
    print("=============================================")
    if model == "lasso":
        reg, best_params, grid_search = cross_validate(X, Y.ravel(), family, task="regression", cv=5, n_jobs=20)
        print(best_params)
        print(f"R2 score: {reg.score(X, Y)}")
        print(f"Sparsity: {(1 - np.sum(reg.coef_ != 0)/reg.coef_.shape[0]):.2%}")
        print(f"Number of non-zero coefficients: {np.sum(reg.coef_ != 0)}")

    elif model == "logistic_l1":
        mask = Y.ravel() == 3
        Y = Y[~mask]
        X = X[~mask]
        family = np.array(family)[~mask]
        reg, best_params, grid_search = cross_validate(X, Y.ravel(), family,  task="classification", cv=5, n_jobs=20)
        print(best_params)
        print(f"Accuracy: {reg.score(X, Y)}")
        print("MCC:", matthews_corrcoef(Y, reg.predict(X)))
        print("F1:", f1_score(Y, reg.predict(X), average="weighted"))
        print(f"Sparsity: {(1 - np.sum(reg.coef_ != 0)/reg.coef_.shape[1]):.2%}")
        print(f"Number of non-zero coefficients: {np.sum(reg.coef_ != 0)}")
    else:
        raise ValueError("Model must be either 'lasso' or 'logistic'")

    print("=============================================")
    
    # Save the model
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/{name}_linear_model.pkl", "wb") as f:
        pickle.dump(reg, f)

    if metadata is not None:
        # invert mapping dictionary. Each number corresponds to a column in the matrix
        # and key 0 does not exist
        if DNA_format == "6-mer" and 0 not in metadata['kmer_dict_TSS']:
            metadata['kmer_dict_TSS'] = {v: k for k, v in metadata['kmer_dict_TSS'].items()}
            metadata['kmer_dict_TTS'] = {v: k for k, v in metadata['kmer_dict_TTS'].items()}
            # assert that these two are the same actually
            assert metadata['kmer_dict_TSS'] == metadata['kmer_dict_TTS']
        # display the coefficients in a barplot by the order of the absolute value, removing the zero coefficients

        if separated_segments:
                coeff_names = [x for x in metadata["TFs_TSS"]] + [x for x in metadata["TFs_TTS"]] if DNA_format == "DAPseq" else [metadata['kmer_dict_TSS'][x] + "_TSS" for x in metadata['kmer_dict_TSS']] + [metadata['kmer_dict_TTS'][x] + "_TTS" for x in metadata['kmer_dict_TTS']]
        else:
            coeff_names = [x for x in metadata["TFs_TSS"]] if DNA_format == "DAPseq" else [metadata['kmer_dict_TSS'][x] for x in metadata['kmer_dict_TSS']]
            # remove the _TSS
            coeff_names = [x.split("_")[0] for x in coeff_names]

        if model == "lasso":
            plt.figure(figsize=(10, 10))
            coeff = reg.coef_
            coeff_names = np.array(coeff_names)[coeff != 0]
            coeff = coeff[coeff != 0]
            
            # now order the coefficients by the absolute value
            idx = np.argsort(-np.abs(coeff)) # because the default is ascending
            coeff = coeff[idx]
            coeff_names = coeff_names[idx]
            # Only show the first 30 coefficients
            plt.barh(coeff_names[:30][::-1], coeff[:30][::-1])
            plt.xlabel("Coefficient value")
            plt.ylabel("TF")
            plt.title(f"30 most important coefficients for {name} \n total number of non-zero coefficients: " + str(len(coeff)))
            plt.savefig(f"{folder}/{name}_coefficients.png")
            # save the coefficients and names in a df
            df = pd.DataFrame({"TF": coeff_names, "Coefficient": coeff})
            df.to_csv(f"{folder}/{name}_coefficients.csv", index=False)
        
        elif model == "logistic_l1":
            plt.figure(figsize=(10, 10))
            coeff = reg.coef_.ravel()
            coeff_names = np.array(coeff_names)[coeff != 0]
            coeff = coeff[coeff != 0]
            
            # now order the coefficients by the absolute value
            idx = np.argsort(-np.abs(coeff)) # because the default is ascending
            coeff = coeff[idx]
            coeff_names = coeff_names[idx]
            # Only show the first 30 coefficients
            plt.barh(coeff_names[:30][::-1], coeff[:30][::-1])
            plt.xlabel("Coefficient value")
            plt.ylabel("TF")
            plt.title(f"30 most important coefficients for {name} \n total number of non-zero coefficients: " + str(len(coeff)))
            plt.savefig(f"{folder}/{name}_coefficients.png")
            # save the coefficients and names in a df
            df = pd.DataFrame({"TF": coeff_names, "Coefficient": coeff})
            df.to_csv(f"{folder}/{name}_coefficients.csv", index=False)

    return reg

def test_linear(X_test, Y_test, reg, model, folder, name, metadata):
    '''
    Tests the linear model on the test set and prints the results.
    '''
    print("=============================================")
    print("TEST")
    print("=============================================")
    if model == "lasso":
        Y_pred = reg.predict(X_test)
        # gather the metrics
        metrics = {}
        metrics["R2"] = reg.score(X_test, Y_test)
        metrics["Pearson"] = np.corrcoef(Y_test.ravel(), Y_pred.ravel())[0, 1]
        metrics["Spearman"] = stats.spearmanr(Y_test.ravel(), Y_pred.ravel()).statistic
        metrics['sign_accuracy'] = np.mean(np.sign(Y_test) == np.sign(Y_pred))
        # save predictions
        np.save(f"{folder}/{name}_predictions.npy", Y_pred)
        # save metrics
        df = pd.DataFrame(metrics, index=[0])
        df.to_csv(f"{folder}/{name}_metrics.csv", index=False)
        
        print(f"R2: {metrics['R2']}")
        print(f"Pearson: {metrics['Pearson']}")
        print(f"Spearman: {metrics['Spearman']}")

    elif model == "logistic_l1":
        mask = Y_test.ravel() == 3
        Y_test = Y_test[~mask]
        X_test = X_test[~mask]
        Y_pred = reg.predict_proba(X_test)[:, 1]
        # gather the metrics
        metrics = {}
        metrics["Accuracy"] = accuracy_score(Y_test, Y_pred > 0.5)
        metrics["MCC"] = matthews_corrcoef(Y_test, Y_pred > 0.5)
        metrics["F1"] = f1_score(Y_test, Y_pred > 0.5)
        metrics["AUC"] = roc_auc_score(Y_test, Y_pred)
        
        print(f"Accuracy: {metrics['Accuracy']}")
        print(f"MCC: {metrics['MCC']}")
        print(f"F1: {metrics['F1']}")
        print(f"AUC: {metrics['AUC']}")

        # save predictions
        np.save(f"{folder}/{name}_predictions.npy", Y_pred)
        # save metrics
        df = pd.DataFrame(metrics, index=[0])
        df.to_csv(f"{folder}/{name}_metrics.csv", index=False)

    else:
        raise ValueError("Model must be either 'lasso' or 'logistic'")

