import pandas as pd
import numpy as np
import os


def run_modisco(DNA_specs, nb_num, offset, n_seqlets, treatments, mapping):

    outcome_types = ["log2FC"]#, "quantiles_per_treatment"]
    queries = {}
    # Assuming the following DNA specs:
    assert (sum(DNA_specs) - 2 * offset) % 2 == 0

    for outcome_type in outcome_types:
        for file in os.listdir(
                f"Results/Interpretation_{nb_num}/{outcome_type}/queries"
            ):
                if file.endswith(".npy"):
                    gene = file.split("_")[0]
                    queries[gene] = np.load(
                        f"Results/Interpretation_{nb_num}/{outcome_type}/queries/{file}"
                    )
        # replicates, treatments, nucleobases, positions
        for i, treatment in enumerate(treatments):
            #if the report folder exists, continue
            if os.path.exists(
               f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/report"
            ):
               print(
                   f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/report already exists. Skipping..."
               )
               continue
            # Load the genes hyp scores and queries
            # implies iterating over the genes
            
            hyp_scores = {}
            
            for file in os.listdir(
                f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/hypothetical_scores"
            ):
                if file.endswith(".npy"):
                    gene = file.split("_")[0]
                    hyp_scores[gene] = np.load(
                        f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/hypothetical_scores/{file}"
                    )


            # construct a 3D array of the hypothetical scores and another for the queries
            hyp_shap_values_array = np.zeros(
                (len(hyp_scores), 4, sum(DNA_specs) - 2 * offset)
            )  # because the padding region is removed
            print(hyp_shap_values_array.shape)
            queries_array = np.zeros((len(hyp_scores), 4, sum(DNA_specs) - 2 * offset))
            gene_list = list(hyp_scores.keys())
            for i, gene in enumerate(hyp_scores):

                processed_hyp_scores = hyp_scores[gene]
                processed_hyp_scores = np.concatenate(
                    (
                        processed_hyp_scores[
                            :, :, : (DNA_specs[0] + DNA_specs[1] - offset)
                        ],
                        processed_hyp_scores[
                            :, :, -(DNA_specs[2] + DNA_specs[3] - offset) :
                        ],
                    ),
                    axis=2,
                )

                processed_queries = queries[gene]
                processed_queries = np.concatenate(
                    (
                        processed_queries[
                            :, :, : (DNA_specs[0] + DNA_specs[1] - offset)
                        ],
                        processed_queries[
                            :, :, -(DNA_specs[2] + DNA_specs[3] - offset) :
                        ],
                    ),
                    axis=2,
                )

                # now, make sure that there is no all 0s case in the queries (there are ~8 instances in the database where this can happen)
                shape_missing = processed_queries[
                    processed_queries.sum(-1) == 0, :
                ].shape
                random_indices = np.random.randint(
                    0, 4, size=shape_missing[0]
                )  # Random indices for one-hot encoding
                one_hot_filled = np.zeros(shape_missing, dtype=int)
                one_hot_filled[np.arange(shape_missing[0]), random_indices] = (
                    1  # Fill with 1s
                )
                processed_queries[processed_queries.sum(-1) == 0, :] = one_hot_filled
                assert np.sum(processed_queries.sum(-1) == 0) == 0

                hyp_shap_values_array[i, :, :] = processed_hyp_scores
                queries_array[i, :, :] = processed_queries

            # store in modisco_run subfolder
            os.makedirs(
                f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run",
                exist_ok=True,
            )
            np.save(
                f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/hyp_shap_values.npy",
                hyp_shap_values_array,
            )
            np.save(
                f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/queries.npy",
                queries_array,
            )
            # write the csv file
            df = pd.DataFrame(data={"gene": gene_list})
            df.to_csv(
                f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/gene_list.csv",
                index=True,
            )
            # run modisco
            cmd = f"modisco motifs --sequences Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/queries.npy \
                --attributions Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/hyp_shap_values.npy \
                    -n {n_seqlets} -o Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/modisco_results.h5 -w {sum(DNA_specs) - 2 * offset}\
                        --verbose "  # Literally the window must be the size of the sequence
                
            print("Running:", cmd)
            os.system(cmd)

            # now remove the .npy
            os.remove(
                f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/hyp_shap_values.npy"
            )
            os.remove(
                f"Results/Interpretation_{nb_num}/{outcome_type}/{mapping[treatment]}/modisco_run/queries.npy"
            )

def run_report(treatments, bp): ## TODO fix this for the bp distinction kind of stuff
    outcome_types = ["log2FC"]#, "quantiles_per_treatment"]
    import pdb; pdb.set_trace()
    for outcome_type in outcome_types:
        # replicates, treatments, nucleobases, positions
        for i, treatment in enumerate(treatments):
            
            cmd_report = f"modisco report -i Results/Interpretation_{bp}/{outcome_type}/{mapping[treatment]}/modisco_run/modisco_results.h5\
                        -o Results/Interpretation_{bp}/{outcome_type}/{mapping[treatment]}/modisco_run/report/ -m Data/RAW/MOTIF/JASPAR_2024_PLANT_motifs.txt"
            print("Running:", cmd_report)
            os.system(cmd_report)


if __name__ == "__main__":
    DNA_specs = [814, 200, 200, 814]
    offset = 15
    n_seqlets = 50000
    treatments = ["B", "C", "D", "G", "X", "Y", "Z", "W", "V", "U", "T"]
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
    run_modisco(DNA_specs, 4096, offset, n_seqlets, treatments, mapping)
    run_report(treatments, 4096)
