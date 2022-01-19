import numpy as np
from numpy.linalg import norm
import pandas as pd
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
from functools import lru_cache
from tqdm import tqdm
import os
import pathlib
import math

from sklearn.preprocessing import normalize, OneHotEncoder

VALID_METHODS = {
    "identity",
    "CustomOSDA",
    "CustomZeolite",
    "random",
    "CustomOSDAandZeolite",
    "CustomOSDAandZeoliteAsRows",
    "CustomOSDAVector",
    "ManualZeolite",
}

ZEOLITE_PRIOR_LOOKUP = {
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
    # "alpha": 1.0,
    # "betta": 1.0,
    # "gamma": 1.0,
    "volume": 1.0,
    # "rdls": 1.0,
    # "framework_density": 1.0,
    # "td_10": 1.0,
    # "td": 1.0,
    "included_sphere_diameter": 1.0,
    # "diffused_sphere_diameter_a": 1.0,
    # "diffused_sphere_diameter_b": 1.0,
    # "diffused_sphere_diameter_c": 1.0,
    # "accessible_volume": 1.0,
}

OSDA_PRIOR_LOOKUP = {
    "mol_weights": 1.0,
    "volume": 1.0,
    "normalized_num_rotatable_bonds": 1.0,
    "formal_charge": 1.0,
    "asphericity": 1.0,
    "eccentricity": 1.0,
    "inertial_shape_factor": 1.0,
    "spherocity": 1.0,
    "gyration_radius": 1.0,
    "pmi1": 1.0,
    "pmi2": 1.0,
    "pmi3": 1.0,
    "npr1": 1.0,
    "npr2": 1.0,
    "free_sas": 1.0,
    "bertz_ct": 1.0,
}
ZEOLITE_PRIOR_FILE = "/Users/mr/Documents/Work/MIT/PhD/matrix_completion/ntk_matrix_completion/cmap_imputation/data/scraped_zeolite_data.pkl"
OSDA_PRIOR_FILE = "/Users/mr/Documents/Work/MIT/PhD/matrix_completion/ntk_matrix_completion/cmap_imputation/data/precomputed_OSDA_prior_10_with_whims.pkl"
OSDA_CONFORMER_PRIOR_FILE = "//Users/mr/Documents/Work/MIT/PhD/matrix_completion/ntk_matrix_completion/cmap_imputation/data/OSDA_priors_with_conjugates.pkl"


ZEOLITE_PRIOR_MAP = {
    "*CTH": "CTH",
    "*MRE": "MRE",
    "*PCS": "PCS",
    "*STO": "STO",
    "*UOE": "UOE",
    "*BEA": "BEA",
}


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    matrix.to_pickle(savepath)


# Turns out conformers are not very useful at all...
def load_conformer_priors(
    target_index,
    precomputed_file_name,
    identity_weight=0.01,
    normalize=True,
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    precomputed_prior = precomputed_prior.reindex(target_index)
    precomputed_prior = precomputed_prior.filter(items=["conformer"])
    exploded_prior = pd.DataFrame(
        columns=[
            "num_conformers",
            # "mean_volume",
            # "min_volume",
            "std_dev_volume",
            # "mean_energy",
            # "min_energy",
            "std_dev_energy",
        ]
    )
    for index, row in precomputed_prior.iterrows():
        if (
            row["conformer"] is np.NaN
            or isinstance(row["conformer"], float)
            or isinstance(row["conformer"].values[0], float)
            or len(row["conformer"].values[0]["volumes"]) == 0
        ):
            series = pd.Series(
                {
                    "num_conformers": 1.0,
                    # "mean_volume": 0.0,
                    # "min_volume": 0.0,
                    "std_dev_volume": 0.0,
                    # "mean_energy": 0.0,
                    # "min_energy": 0.0,
                    "std_dev_energy": 0.0,
                }
            )
        else:
            conformer_properties = row["conformer"].values[0]
            series = pd.Series(
                {
                    "num_conformers": len(conformer_properties["volumes"]),
                    # "mean_volume": np.mean(conformer_properties["volumes"]),
                    # "min_volume": min(conformer_properties["volumes"]),
                    "std_dev_volume": np.std(conformer_properties["volumes"]),
                    # "mean_energy": np.mean(conformer_properties["energies"]),
                    # "min_energy": min(conformer_properties["energies"]),
                    "std_dev_energy": np.std(conformer_properties["energies"]),
                }
            )
        series.name = index
        exploded_prior = exploded_prior.append(series)
    if normalize:
        exploded_prior = exploded_prior.apply(lambda x: x / x.max(), axis=0)

    conformer_prior = exploded_prior.to_numpy(dtype=float)
    normalized_conformer_prior = conformer_prior / (max(conformer_prior, key=sum).sum())
    return (1 - identity_weight) * normalized_conformer_prior


def load_vector_priors(
    target_index,
    vector_feature,
    precomputed_file_name,
    identity_weight=0.01,
    normalize=True,
    other_prior_to_concat="data/data_from_daniels_ml_models/precomputed_energies_78616by196WithWhims.pkl",
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    if other_prior_to_concat:
        big_precomputed_priors = pd.read_pickle(other_prior_to_concat)
        precomputed_prior = pd.concat([big_precomputed_priors, precomputed_prior])
        precomputed_prior = precomputed_prior[
            ~precomputed_prior.index.duplicated(keep="first")
        ]

    precomputed_prior = precomputed_prior.reindex(target_index)
    precomputed_prior = precomputed_prior.filter(items=[vector_feature])

    num_elements = len(precomputed_prior[vector_feature][0])

    def column_name(index, vector_feature=vector_feature):
        return vector_feature + "_" + str(index)

    column_names = [column_name(i) for i in range(num_elements)]
    exploded_prior = pd.DataFrame(columns=column_names)
    for index, row in precomputed_prior.iterrows():
        if row[vector_feature] is np.NaN:
            series = pd.Series({column_name(i): 0.0 for i in range(num_elements)})
        else:
            series = pd.Series(
                {
                    column_name(i): 0 if np.isnan(e) else e
                    for i, e in enumerate(row[vector_feature])
                }
            )
        series.name = index
        exploded_prior = exploded_prior.append(series)

    # Normalize across the whole thing...
    # Normalize to the biggest value & across all of the elements
    biggest_value = exploded_prior.max().max()
    normalization_factor = biggest_value * num_elements + identity_weight
    if normalize:
        exploded_prior = exploded_prior.apply(
            lambda x: x / normalization_factor, axis=0
        )
    return exploded_prior.to_numpy()


def load_prior(
    target_index,
    column_weights,
    precomputed_file_name,
    identity_weight=0.01,
    normalize=True,
    prior_index_map=None,
    other_prior_to_concat="data/data_from_daniels_ml_models/precomputed_energies_78616by196WithWhims.pkl",
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    if other_prior_to_concat:
        big_precomputed_priors = pd.read_pickle(other_prior_to_concat)
        precomputed_prior = pd.concat([big_precomputed_priors, precomputed_prior])
        precomputed_prior = precomputed_prior[
            ~precomputed_prior.index.duplicated(keep="first")
        ]

    # TODO(Mingrou): add new zeolite prior...
    if prior_index_map:  # zeolite prior lookup MR
        x = lambda i: prior_index_map[i] if i in prior_index_map else i
        precomputed_prior.index = precomputed_prior.index.map(x)
    precomputed_prior = precomputed_prior.reindex(target_index)
    precomputed_prior = precomputed_prior.filter(items=list(column_weights.keys()))
    precomputed_prior = precomputed_prior.apply(pd.to_numeric)
    precomputed_prior = precomputed_prior.fillna(0.0)

    # Normalize down each column to between 0 & 1
    if normalize:
        precomputed_prior = precomputed_prior.apply(lambda x: x / x.max(), axis=0)
    # Now time to weigh each column, taking into account identity_weight to make sure
    # later when we add the identity matrix we don't go over 1.0 total per row...
    normalization_factor = sum(column_weights.values()) + identity_weight
    results = precomputed_prior.apply(
        lambda x: x * column_weights[x.name] / normalization_factor, axis=0
    )

    return results


def osda_prior(
    all_data_df,
    identity_weight=0.01,
    normalize=True,
):
    return load_prior(
        all_data_df.index,
        OSDA_PRIOR_LOOKUP,
        OSDA_PRIOR_FILE,
        identity_weight,
        normalize,
    )


def osda_vector_prior(
    all_data_df,
    vector_feature="getaway",
    identity_weight=0.01,
    normalize=True,
):
    prior = osda_prior(all_data_df, identity_weight)
    getaway_prior = load_vector_priors(
        all_data_df.index,
        vector_feature,
        OSDA_PRIOR_FILE,
        identity_weight,
        normalize,
    )

    # Splitting the original prior and the vector prior 50-50
    normalized_getaway_prior = getaway_prior / (2 * max(getaway_prior, key=sum).sum())
    normalized_prior = prior / (2 * max(prior, key=sum).sum())
    stacked = np.hstack([normalized_prior, normalized_getaway_prior])

    # Make room for the identity weight
    stacked = (1 - identity_weight) * stacked
    return stacked


def zeolite_prior(
    all_data_df, feature_lookup, identity_weight=0.01, normalize=True, file_name=None
):
    return load_prior(
        all_data_df.index,
        ZEOLITE_PRIOR_LOOKUP if not feature_lookup else feature_lookup,
        ZEOLITE_PRIOR_FILE,
        identity_weight,
        normalize,
        ZEOLITE_PRIOR_MAP,
        other_prior_to_concat=file_name,
    )


def osda_zeolite_combined_prior(
    all_data_df,
    identity_weight=0.01,
    normalize=True,
):
    # Give identity weight more so we can normalize both to be less than 1
    identity_weight += max(
        np.array(list(OSDA_PRIOR_LOOKUP.values())).sum(),
        np.array(list(ZEOLITE_PRIOR_LOOKUP.values())).sum(),
    )
    osda_prior = load_prior(
        [i[0] for i in all_data_df.index],
        OSDA_PRIOR_LOOKUP,
        OSDA_PRIOR_FILE,
        identity_weight,
        normalize,
    )
    zeolite_prior = load_prior(
        [i[1] for i in all_data_df.index],
        ZEOLITE_PRIOR_LOOKUP,
        ZEOLITE_PRIOR_FILE,
        identity_weight,
        normalize,
        ZEOLITE_PRIOR_MAP,
    )
    return np.hstack([osda_prior, zeolite_prior])


def plot_matrix(M, file_name, mask=None, vmin=16, vmax=23):
    fig, ax = plt.subplots()
    cmap = mpl.cm.get_cmap()
    cmap.set_bad(color="white")
    if mask is not None:

        def invert_binary_mask(m):
            return np.logical_not(m).astype(int)

        inverted_mask = invert_binary_mask(mask)
        masked_M = np.ma.masked_where(inverted_mask, M)
    else:
        masked_M = M
    im = ax.imshow(masked_M, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    fig.savefig(file_name + ".png", dpi=150)


def make_prior(
    train,
    test,
    method="identity",
    normalization_factor=1.5,
    test_train_axis=0,
    feature=None,
    all_data=None,
    file_name=None,
):
    assert method in VALID_METHODS, f"Invalid method used, pick one of {VALID_METHODS}"
    if all_data is not None:
        all_data_df = all_data
    else:
        if test_train_axis == 0:
            all_data = np.vstack((train.to_numpy(), test.to_numpy()))
            all_data_df = pd.concat([train, test])
        elif test_train_axis == 1:
            all_data = np.hstack((train.to_numpy(), test.to_numpy()))
            all_data_df = pd.concat([train, test], 1)
        else:
            all_data = None
            all_data_df = pd.concat([train, test], test_train_axis)

    prior = None

    if method == "identity":
        prior = np.eye(all_data.shape[0])
        return prior

    elif method == "CustomOSDA":
        prior = osda_prior(all_data_df, normalization_factor)
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    elif method == "CustomOSDAVector":
        prior = osda_vector_prior(all_data_df, "getaway", normalization_factor)
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    elif method == "CustomZeolite":
        prior = zeolite_prior(all_data_df, feature, file_name=file_name)
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    # This one is for the failed experiment
    elif method == "CustomOSDAandZeolite":
        osda_axis1_lengths = osda_prior(
            all_data_df, column_name="Axis 1 (Angstrom)", normalize=False
        )
        zeolite_sphere_diameters = zeolite_prior(all_data_df)

        prior = np.zeros((len(osda_axis1_lengths), len(zeolite_sphere_diameters)))
        for i, osda_length in enumerate(osda_axis1_lengths):
            for j, zeolite_sphere_diameter in enumerate(zeolite_sphere_diameters):
                prior[i, j] = zeolite_sphere_diameter - osda_length

        if prior.min() < 0:
            prior = prior - prior.min()
        # TODO: is this necessary to normalize all of the values in prior to maximum?
        # This isn't working...
        # prior = prior / prior.max()

        # Normalize prior across its rows:
        max = np.reshape(np.repeat(prior.sum(axis=1), prior.shape[1]), prior.shape)
        prior = prior / max
        prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
        return prior
        # TODO: DO I ALSO NEED to normalize all of the rows to 1... DO I???
        # plot_matrix(prior, 'prior', vmin=0, vmax=1)
        # prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
        # return prior

    # This is the one for really skinny Matrices
    elif method == "CustomOSDAandZeoliteAsRows":
        prior = osda_zeolite_combined_prior(all_data_df, normalize=True)

    elif method == "random":
        dim = 100
        prior = np.random.rand(all_data.shape[0], dim)

    if test_train_axis == 0:
        prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
    elif test_train_axis == 1:
        prior = np.hstack(
            [
                prior,
                normalization_factor
                * np.eye(all_data.shape[1])[0 : all_data.shape[0], 1:],
            ]
        )
        # TODO: this is quite gross... is this the right way to be making this?
        # TODO: there is a better way to do this... add the bottom to the prior col first then just hstack the eye once.
        # prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[1])[0:all_data.shape[0]]])
        # lower_buffer = np.eye(all_data.shape[1])[all_data.shape[0]:]
        # lower_buffer = np.hstack([np.zeros((lower_buffer.shape[0], 1)), lower_buffer])
        # prior = np.vstack([prior, lower_buffer])
    # TODO: big debate... do I like this normalize?? probably not...
    normalize(prior, axis=1, copy=False)
    return prior
