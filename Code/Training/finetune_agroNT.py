import sys

sys.path.append(".")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    r2_score,
)
from peft import get_peft_model, PeftModel, PeftConfig
from peft import LoraConfig, TaskType, IA3Config
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset
from sklearn.metrics import roc_curve
from Training.train_model.train_cnn import get_loader
from torch.utils.data import DataLoader
import json
import pandas as pd
from scipy import stats


def handle_data_AgroNT(
    TSS_sequences, mRNA_train, mRNA_validation, mRNA_test, store_folder, n_tokens
):

    # construct train, test and validation loaders
    os.makedirs(store_folder, exist_ok=True)
    training_loader = get_loader(
        mRNA_train,
        TSS_sequences,  # it does not matter, I will take just the first half!
        TSS_sequences,
        32,  # does not matter!
        True,
    )
    validation_loader = get_loader(
        mRNA_validation,
        TSS_sequences,
        TSS_sequences,
        32,
        False,
    )

    test_loader = get_loader(
        mRNA_test,
        TSS_sequences,
        TSS_sequences,
        32,
        False,
    )

    # Now I need to get the dataset from the loaders
    training_dataset = training_loader.dataset
    validation_dataset = validation_loader.dataset
    test_dataset = test_loader.dataset

    assert all(
        [
            i in training_dataset.TSS_dna.keys()
            for i in training_dataset.mRNA["Gene"].values
        ]
    )
    train_sequences_promoter = [
        training_dataset.TSS_dna[i] for i in training_dataset.mRNA["Gene"].values
    ]
    train_labels = training_dataset.mRNA.iloc[:, 1:-1].values.astype(np.float32)

    assert all(
        [
            i in validation_dataset.TSS_dna.keys()
            for i in validation_dataset.mRNA["Gene"].values
        ]
    )
    val_sequences_promoter = [
        validation_dataset.TSS_dna[i] for i in validation_dataset.mRNA["Gene"].values
    ]
    val_labels = validation_dataset.mRNA.iloc[:, 1:-1].values.astype(np.float32)

    assert all(
        [i in test_dataset.TSS_dna.keys() for i in test_dataset.mRNA["Gene"].values]
    )
    test_sequences_promoter = [
        test_dataset.TSS_dna[i] for i in test_dataset.mRNA["Gene"].values
    ]
    test_labels = test_dataset.mRNA.iloc[:, 1:-1].values.astype(np.float32)

    ds_train_promoter = Dataset.from_dict(
        {"data": train_sequences_promoter, "labels": train_labels}
    )
    ds_validation_promoter = Dataset.from_dict(
        {"data": val_sequences_promoter, "labels": val_labels}
    )
    ds_test_promoter = Dataset.from_dict(
        {"data": test_sequences_promoter, "labels": test_labels}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/agro-nucleotide-transformer-1b"
    )

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["data"], padding="max_length", truncation=True, max_length=n_tokens
        )
        return outputs

    # Creating tokenized promoter dataset
    tokenized_datasets_train_promoter = ds_train_promoter.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"],
    ).with_format("torch")
    tokenized_datasets_validation_promoter = ds_validation_promoter.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"],
    ).with_format("torch")
    tokenized_datasets_test_promoter = ds_test_promoter.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"],
    )
    tokenized_datasets_train_promoter.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    tokenized_datasets_validation_promoter.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    tokenized_datasets_test_promoter.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return (
        tokenized_datasets_train_promoter,
        tokenized_datasets_validation_promoter,
        tokenized_datasets_test_promoter,
        tokenizer,
    )
    # return None, None, tokenized_datasets_test_promoter, tokenizer


def compute_metrics_regression(eval_pred):
    logits, labels = eval_pred
    mse = ((logits - labels) ** 2).mean().item()
    r2 = r2_score(labels, logits, multioutput="raw_values")
    pearson_corr = [
        np.corrcoef(logits[:, i], labels[:, i])[0, 1] for i in range(logits.shape[1])
    ]
    # sign accuracy per head
    sign_accuracy = [
        np.mean(np.sign(logits[:, i]) == np.sign(labels[:, i]))
        for i in range(logits.shape[1])
    ]
    spearman_corr = [
        stats.spearmanr(logits[:, i], labels[:, i]).statistic
        for i in range(logits.shape[1])
    ]

    metrics = {
        "mse": mse,
    }
    for i, (r2_val, corr_val) in enumerate(zip(r2, pearson_corr)):
        metrics[f"R2_head_{i}"] = r2_val
        metrics[f"Pearson_head_{i}"] = corr_val
        metrics[f"sign-accuracy_head_{i}"] = sign_accuracy[i]
        metrics[f"Spearman_head_{i}"] = spearman_corr[i]

    return metrics


def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    logits = 1 / (1 + np.exp(-logits))
    metrics = {}
    # get loss BCEwithLogits

    loss = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss(torch.tensor(logits), torch.tensor(labels))
    loss = loss * (labels != 3)
    loss = loss.mean().item()
    metrics["loss"] = loss

    for i in range(labels.shape[1]):
        mask = labels[:, i] != 3
        auc = roc_auc_score(labels[mask, i], logits[mask, i])
        f1 = f1_score(labels[mask, i], logits[mask, i] > 0.5)
        mcc = matthews_corrcoef(labels[mask, i], logits[mask, i] > 0.5)
        acc = accuracy_score(labels[mask, i], logits[mask, i] > 0.5)

        metrics[f"AUC_head_{i}"] = auc
        metrics[f"F1_head_{i}"] = f1
        metrics[f"MCC_head_{i}"] = mcc
        metrics[f"Accuracy_head_{i}"] = acc

    return metrics


class MaskedBCELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")
        # Compute per-label loss

    def forward(self, output, target):
        mask = torch.ones(target.shape[0], output.shape[1]).to(target.device)
        mask[target == 3] = 0
        loss = self.loss_fn(output, target)
        return (loss * mask).sum() / mask.sum()


class CustomTrainerClassification(Trainer):
    def __init__(self, *args, loss_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = MaskedBCELoss(weights=loss_weights)  # Use the custom loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)
        logits = outputs.logits  # Get model predictions

        # Compute loss with the custom loss function
        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


class CustomTrainerRegression(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)
        logits = outputs.logits  # Get model predictions

        # Compute loss with the custom loss function
        loss = F.l1_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


def finetune_agroNT(
    ds_train,
    ds_val,
    ds_test,
    treatment_names,
    tokenizer,
    n_heads,
    batch_size,
    epochs,
    device,
    output_dir,
    problem_type="regression",
    weights=None,
    gradient_accumulation_steps=4,
    logging_steps=300,
    save_steps=300,
    early_stopping_patience=8,
):

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.15,
        target_modules=["query", "key", "value"],
        modules_to_save=[
            "classifier.out_proj.weight",
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.bias",
        ],
    )
    # This is to ALSO save the classification head, just to be sure
    # Indeed, with this configuration, the classification head is saved

    # Load the model
    if problem_type == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(
            "InstaDeepAI/agro-nucleotide-transformer-1b",
            problem_type="regression",
            num_labels=n_heads,
        )

        model = get_peft_model(
            model, peft_config
        )  # transform our classifier into a peft model
        model.print_trainable_parameters()
        model.to(device)

        args_promoter = TrainingArguments(
            output_dir,
            remove_unused_columns=False,
            evaluation_strategy="steps",
            save_strategy="steps",
            learning_rate=5e-5,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=1,  # This saves the BEST model and the last one
            load_best_model_at_end=True,  # Keep the best model according to the evaluation
            metric_for_best_model="mse",  # see above function
            # label_names=["labels"],
            dataloader_drop_last=True,
            fp16=False,  # Use mixed precision
            greater_is_better=False,  # This is for the metric if using mse, it should be False
        )

        trainer = CustomTrainerRegression(
            model,
            args_promoter,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            compute_metrics=compute_metrics_regression,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ],
        )

    elif problem_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            "InstaDeepAI/agro-nucleotide-transformer-1b",
            problem_type="multi_label_classification",
            num_labels=n_heads,
        )
        # Actually I do not need to change the problem type because I am making a custom loss function but ok

        model = get_peft_model(
            model, peft_config
        )  # transform our classifier into a peft model
        model.print_trainable_parameters()
        model.to(device)

        args_promoter = TrainingArguments(
            output_dir,
            remove_unused_columns=False,
            evaluation_strategy="steps",
            save_strategy="steps",
            learning_rate=5e-5,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=1,  # This saves the BEST model and the last one
            load_best_model_at_end=True,  # Keep the best model according to the evaluation
            metric_for_best_model="loss",  # see above function
            # label_names=["labels"],
            dataloader_drop_last=True,
            fp16=False,  # Use mixed precision
            greater_is_better=False,  # This is for the metric if using mse, it should be False
        )

        trainer = CustomTrainerClassification(
            model,
            args_promoter,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            compute_metrics=compute_metrics_classification,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ],
            loss_weights=weights.to(device) if weights is not None else None,
        )
    else:
        raise ValueError("Problem type must be either regression or classification")

    train_results = trainer.train()
    # save training results
    train_metrics = format_metrics(problem_type, train_results.metrics, treatment_names)
    # save
    train_metrics.to_csv(f"{output_dir}/train_metrics.csv")

    adapter_model = f"{output_dir}/adapter_model"
    model.save_pretrained(f"{adapter_model}")  # Save ONLY the LoRA adapter
    tokenizer.save_pretrained(
        f"{adapter_model}"
    )  # this actually does not matter but ok

    # now predict
    test_results = trainer.predict(ds_test)
    test_metrics = format_metrics(problem_type, test_results.metrics, treatment_names)
    # save
    test_metrics.to_csv(f"{output_dir}/test_metrics.csv")

    return test_results


def format_metrics(type_metrics, metrics, treatment_names):
    head_metrics = {k: v for k, v in metrics.items() if "head" in k}

    # Extract unique metric types and heads
    metric_types = set(key.split("_")[1] for key in head_metrics.keys())
    heads = set(int(key.split("_head_")[1]) for key in head_metrics.keys())

    # Create a dictionary to store the data
    data = {metric: [] for metric in metric_types}
    data["head"] = sorted(heads)

    # Populate the dictionary
    for head in sorted(heads):
        for metric in metric_types:
            key = f"test_{metric}_head_{head}"
            data[metric].append(head_metrics.get(key, None))

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set the "head" column as the index
    df.set_index("head", inplace=True)
    df = df.T
    df.columns = treatment_names
    # if sign-accuracy is present in index change it to sign_accuracy
    if "sign-accuracy" in df.index:
        df.rename(index={"sign-accuracy": "sign_accuracy"}, inplace=True)
    # Display the DataFrame
    print(df)
    # save the dataframe
    return df
