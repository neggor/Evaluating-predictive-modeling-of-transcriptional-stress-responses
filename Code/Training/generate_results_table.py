import pandas as pd
import os



def generate_table():
    outcome_types = ["quantiles_per_treatment", "amplitude", "log2FC", "DE_per_treatment"]
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
    metrics_df = pd.DataFrame()
    # initialize the columns, outcome_type, in_type, model_type, rc, treatment, metric, value
    columns = [
        "outcome_type",
        "in_type",
        "treatment",
        "metric",
        "value",
        "model_type",
        "exons",
        "length",
        'rc',
        'replicate'
    ]

    
    for outcome_type in outcome_types:
        # linear models
        for in_type in ["6-mer", "DAPseq"]: 
            for treatment in [mapping[t] for t in treatments]:
                file = f"{treatment}_metrics.csv"
                url_file = f"Results/linear_models/{outcome_type}/{in_type}/{file}"

                # if file does not exist, continue
                if not os.path.exists(url_file):
                    continue
                # load the file
                res_df = pd.read_csv(url_file)
                tmp_df = pd.DataFrame(
                    [
                        {
                            "outcome_type": outcome_type,
                            "in_type": in_type,
                            "treatment": treatment,
                            "metric": metric,
                            "value": res_df[metric].values[0],
                            "model_type": "linear",
                            "exons": "not apply",
                            "length": 2048,
                            "rc": True if in_type == "6-mer" else "not apply",
                            'replicate': "not apply"
                        }
                        for metric in res_df.columns
                    ],
                    columns=columns,
                                )
                metrics_df = pd.concat(
                                    [metrics_df, tmp_df], ignore_index=True
                                    )
        # CNN models
        for exons in [True, False]:
            for length in [5020, 4096, 8192]:
                for model_num in range(5):
                    file = f"Results/CNN/{outcome_type}/{length}/exons_masked_{exons}/model_{model_num}/test_metrics.csv"
                    # Results/CNN/amplitude/2048/exons_masked_False/model_0/test_metrics.csv
                    if not os.path.exists(file):
                        continue
                    res_df = pd.read_csv(file)
                    res_df.set_index(res_df.columns[0], inplace=True)
                    for tr in treatments:
                        # each column is a treatment
                        if tr not in res_df.columns:
                            continue

                        tr_metrics = res_df.loc[:, tr]
                        # create a temporary dataframe for each metric
                        tmp_df = pd.DataFrame(
                            [
                                {
                                    "outcome_type": outcome_type,
                                    "in_type": "One-Hot",
                                    "treatment": mapping[tr],
                                    "metric": metric,
                                    "value": tr_metrics[metric],
                                    "model_type": "CNN",
                                    "exons": f'{"masked" if exons else "all"}',
                                    "length": length,
                                    'rc': True,
                                    'replicate': model_num
                                }
                                for metric in tr_metrics.index
                            ],
                            columns=columns,
                        )
                        metrics_df = pd.concat([metrics_df, tmp_df], ignore_index=True)
        # AgroNT
        file = f"Results/agroNT/{outcome_type}/test_metrics.csv"
        if not os.path.exists(file):
            continue
        res_df = pd.read_csv(file)
        res_df.set_index(res_df.columns[0], inplace=True)
        for tr in treatments:
            # each column is a treatment
            if tr not in res_df.columns:
                continue
            tr_metrics = res_df.loc[:, tr]
            # create a temporary dataframe for each metric
            tmp_df = pd.DataFrame(
                [
                    {
                        "outcome_type": outcome_type,
                        "in_type": "String",
                        "treatment": mapping[tr],
                        "metric": metric,
                        "value": tr_metrics[metric],
                        "model_type": "AgroNT",
                        "exons": "not apply",
                        "length": "not apply",
                        'rc': "not apply",
                        'replicate': "not apply"
                    }
                    for metric in tr_metrics.index
                ],
                columns=columns,
            )
            metrics_df = pd.concat([metrics_df, tmp_df], ignore_index=True)

        print(metrics_df)

    # save the dataframe
    metrics_df.to_csv("Results/Results_table.csv", index=False)
    print("Results table saved!")
    print("DONE!")



if __name__ == "__main__":
    generate_table()