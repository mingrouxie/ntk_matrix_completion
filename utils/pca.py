import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.package_matrix import get_ground_truth_energy_matrix
from features.prior import load_vector_priors
import pdb
import os
from utils.package_matrix import make_skinny
from features.precompute_osda_priors import smile_to_property
from utils.path_constants import (
    HYPOTHETICAL_OSDA_ENERGIES,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PREDICTED_ENERGIES,
)
from scipy.spatial import distance


def get_literature_osdas(finalDf, zeolite):
    rafa_data = "data/completeZeoliteData.pkl"
    if not os.path.exists(rafa_data):
        raise AssertionError(f"Path to matrix, {rafa_data} does not exist")
    osda_df = pd.read_pickle(rafa_data)
    zeolite_specific_osdas = osda_df.loc[osda_df["Zeolite"] == zeolite]
    literature_osdas = zeolite_specific_osdas.loc[
        zeolite_specific_osdas["In literature?"] == 1.0
    ]
    all_literature_smiles_in_index = set.intersection(
        set(finalDf.index), set(literature_osdas["SMILES"])
    )
    return finalDf.loc[all_literature_smiles_in_index]


def plot_double_histogram(ground_truth, predicted_energies, zeolite):
    if zeolite is not None:
        ground_truth_bin = ground_truth[zeolite]
        predicted_energies_bin = predicted_energies[zeolite]
    else:
        ground_truth_bin = make_skinny(ground_truth)
        predicted_energies_bin = make_skinny(
            predicted_energies, col_1="Zeolite", col_2="index"
        )
    plt.hist(ground_truth_bin, bins=100, alpha=0.5, label="ground_truth")
    plt.hist(predicted_energies_bin, bins=100, alpha=0.5, label="hypothetical_osdas")
    plt.title(zeolite + " Ground Truth & Hypothetical Templating Energies")
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.show()
    # plt.savefig(zeolite + "_ground_truth_histogram.png", dpi=100)


def distance_metric_from_literature(
    reasonable_hypothetical_osdas, finalDf, zeolite, sigma, lowest_ground_truth_energy
):
    literature_osdas_pca_components_df = get_literature_osdas(finalDf, zeolite)
    print(
        "how many literatrue osdas are there? ", len(literature_osdas_pca_components_df)
    )
    reasonable_osdas_pca_components_df = finalDf.loc[
        reasonable_hypothetical_osdas.index
    ]
    reasonable_osdas_pca_components_df = reasonable_osdas_pca_components_df.dropna()
    hypo_osdas_by_distance_df = pd.DataFrame()
    for index in range(len(reasonable_osdas_pca_components_df)):
        hypothetical_row = reasonable_osdas_pca_components_df.iloc[index]
        if len(literature_osdas_pca_components_df) >= 1:

            def calculate_distances(
                literature_osdas_pca_components_df, hypothetical_row
            ):
                distances = []
                for lit_row in literature_osdas_pca_components_df.iterrows():
                    keys = ["principal component 1", "principal component 2"]
                    distances.append(
                        distance.euclidean(
                            np.array([hypothetical_row[k] for k in keys]),
                            np.array([lit_row[1][k] for k in keys]),
                        )
                    )
                return distances

            distances = calculate_distances(
                literature_osdas_pca_components_df, hypothetical_row
            )
            # Get the minimum euclidean distance of hypothetical_row and all of the literature OSDAs.
            min_distance_from_literature = np.min(distances)
            closest_literature_index = np.where(
                distances == min_distance_from_literature
            )[0][0]
            closest_literature_osda = literature_osdas_pca_components_df.iloc[
                closest_literature_index
            ]
        else:
            min_distance_from_literature = np.inf
            closest_literature_osda = None
        predicted_energy = reasonable_hypothetical_osdas[hypothetical_row.name]
        series = pd.Series(
            data={
                "principal_component_1": hypothetical_row["principal component 1"],
                "principal_component_2": hypothetical_row["principal component 2"],
                "min_distance_from_literature": min_distance_from_literature,
                "closest_literature_osda_smile": closest_literature_osda.name
                if closest_literature_osda is not None
                else None,
                "closest_literature_osda_principal_component_1": closest_literature_osda[
                    "principal component 1"
                ]
                if closest_literature_osda is not None
                else None,
                "closest_literature_osda_principal_component_2": closest_literature_osda[
                    "principal component 2"
                ]
                if closest_literature_osda is not None
                else None,
                "predicted_templating_energy": predicted_energy,
                "zeolite": zeolite,
                # Aka how many sigma is this particular OSDA from the minimum energy
                "sigma_significance": (predicted_energy - lowest_ground_truth_energy)
                / sigma,
                "sigma": sigma,
                "lowest_ground_truth_energy": lowest_ground_truth_energy,
            },
            name=hypothetical_row.name,
        )
        hypo_osdas_by_distance_df = hypo_osdas_by_distance_df.append(series)
    return hypo_osdas_by_distance_df


def lets_look_at_interesting_osdas(finalDf, verbose=False):
    predicted_energies = pd.read_pickle(OSDA_HYPOTHETICAL_PREDICTED_ENERGIES)
    training_data, _binary_data = get_ground_truth_energy_matrix()
    # Let's plot all the energies first...
    # plot_double_histogram(training_data, predicted_energies, None)

    # Let's take it zeolite by zeolite.
    interesting_osdas = pd.DataFrame()
    for zeolite in predicted_energies.columns:
        # Now we should compute what is a reasonable hypothetical OSDA.
        sigma = np.std(training_data[zeolite])
        lowest_ground_truth_energy = np.min(training_data[zeolite])
        cutoff = lowest_ground_truth_energy + (sigma / 2)
        reasonable_hypothetical_osdas = predicted_energies.loc[
            predicted_energies[zeolite] <= cutoff
        ][zeolite]
        print(
            "for ",
            zeolite,
            " whats sigma? ",
            sigma,
            " and the cutoff? ",
            cutoff,
            " and how many osdas made it? ",
            len(reasonable_hypothetical_osdas),
        )
        if len(reasonable_hypothetical_osdas) == 0:
            continue
        if verbose:
            plot_double_histogram(training_data, predicted_energies, zeolite)
            pdb.set_trace()
        # Now we need to compute some distance measure between all reasonable hypothetical OSDAs & the literature.
        measured_by_distance_from_literature = distance_metric_from_literature(
            reasonable_hypothetical_osdas,
            finalDf,
            zeolite,
            sigma,
            lowest_ground_truth_energy,
        )
        interesting_osdas = interesting_osdas.append(
            measured_by_distance_from_literature
        )
    interesting_osdas.to_csv("interesting_osdas.csv")
    return interesting_osdas


ground_truth, binary_data = get_ground_truth_energy_matrix()
daniel_energies = pd.read_csv(HYPOTHETICAL_OSDA_ENERGIES)
precomputed_priors = pd.read_pickle(OSDA_HYPOTHETICAL_PRIOR_FILE)
daniel_energies = daniel_energies.reindex(precomputed_priors.index)
daniel_energies = daniel_energies.drop(
    set.intersection(set(ground_truth.index), set(daniel_energies.index))
)
all_data = pd.concat([ground_truth, daniel_energies])
# all_data = ground_truth
getaway_df = load_vector_priors(
    target_index=all_data.index,
    vector_feature="getaway",
    identity_weight=0,
    normalize=False,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
    replace_nan=None,
)
getaway_df = getaway_df.dropna()
# all_data = all_data.reindex(getaway_df.index)

x = np.nan_to_num(getaway_df.values)
# Oooo, it already normalizes down all of the columns... not just normalized over the whole matrix...
normalized_altogether = StandardScaler().fit_transform(x)


standardized_getaway_df = pd.DataFrame(
    data=normalized_altogether, index=getaway_df.index
)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(
    data=principalComponents, columns=["principal component 1", "principal component 2"]
)
print(principalDf.head(5))
print("what's the explained variance? ", pca.explained_variance_ratio_)


indicator = pd.DataFrame(
    data=np.vstack(
        (np.ones((ground_truth.shape[0], 1)), np.zeros((daniel_energies.shape[0], 1)))
    )
)
finalDf = pd.concat([principalDf, indicator], axis=1)
# TODO: this is for plot_zeo1_osdas_by_getaway
# finalDf = principalDf#pd.concat([principalDf, indicator], axis=1)
# pdb.set_trace()
finalDf.index = all_data.index

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1", fontsize=15)
ax.set_ylabel("Principal Component 2", fontsize=15)
ax.set_title("PCA of GETAWAY Descriptors", fontsize=20)


def plot_zeo1_osdas_by_getaway(finalDf, fig):
    # indicesToKeep = finalDf[0] == 1.0
    only_training_df = finalDf  # finalDf.loc[indicesToKeep]

    zeo_1_predicted_templating = pd.read_pickle("predicted_zeo1_osdas.pkl")
    zeo_1_predicted_templating = zeo_1_predicted_templating.T
    zeo_1_predicted_templating = zeo_1_predicted_templating.reindex(
        only_training_df.index
    )
    sc = ax.scatter(
        only_training_df["principal component 1"],
        only_training_df["principal component 2"],
        c=zeo_1_predicted_templating,
        s=70,
        alpha=0.5,
    )
    ax.legend(["OSDA Templating \nPredictions for ZEO-1"], loc="upper right")
    fig.colorbar(sc)
    ax.grid()
    plt.show()
    plt.savefig("pca.png")

    zeo_1_predicted_templating = pd.concat(
        [zeo_1_predicted_templating, finalDf], axis=1
    )
    # Okay let's investigate these lowest energy OSDAs...
    sorted_zeo_1 = zeo_1_predicted_templating.sort_values(by=0)
    pdb.set_trace()
    for row in sorted_zeo_1.iterrows():
        smile_to_property(
            row[0],
            save_file=str(round(row[1][0], 4)) + "zeo1" + "_" + row[0],
        )


interest_osdas = lets_look_at_interesting_osdas(finalDf, verbose=True)
# plot_zeo1_osdas_by_getaway(finalDf, fig)

targets = [0, 1]
colors = ["r", "b"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(
        finalDf.loc[indicesToKeep, "principal component 1"],
        finalDf.loc[indicesToKeep, "principal component 2"],
        c=color,
        s=70,
        alpha=0.05,
    )


zeolite_of_interest = "EWS"
literature_osdas_df = get_literature_osdas(finalDf=finalDf, zeolite=zeolite_of_interest)
ax.scatter(
    literature_osdas_df["principal component 1"],
    literature_osdas_df["principal component 2"],
    c="teal",
    s=70,
    alpha=1,
)
for row in literature_osdas_df.iterrows():
    smile_to_property(
        row[0],
        save_file=zeolite_of_interest + "_literature_" + row[0],
    )

# BEST OSDA...
osdas_by_zeolite = interest_osdas.loc[interest_osdas["zeolite"] == zeolite_of_interest]


for row in osdas_by_zeolite.iterrows():
    smile_to_property(
        row[0],
        save_file=zeolite_of_interest + "_" + row[0],
    )
ax.scatter(
    finalDf.loc[osdas_by_zeolite.index, "principal component 1"],
    finalDf.loc[osdas_by_zeolite.index, "principal component 2"],
    c="palegreen",
    s=70,
    alpha=1,
)
ax.legend(["Hypothetical", "Training", "In Literature", "Reasonable Hypothetical OSDA"])
ax.grid()
plt.show()
plt.savefig("pca.png")