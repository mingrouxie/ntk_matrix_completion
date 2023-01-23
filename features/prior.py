import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# from auto_tqdm import tqdm
import os
import pathlib
import math
import scipy as sp
import json

from scipy.sparse import csc_matrix
from functools import lru_cache
from numpy.linalg import norm
from sklearn.preprocessing import normalize, OneHotEncoder

from ntk_matrix_completion.utils.path_constants import (
    HYPOTHETICAL_OSDA_ENERGIES,
    ZEOLITE_PRIOR_FILE,
    HANDCRAFTED_ZEOLITE_PRIOR_FILE,
    OSDA_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_CONFORMER_PRIOR_FILE,
    ZEO_1_PRIOR,
    PERSISTENCE_ZEOLITE_PRIOR_FILE,
    ZEOLITE_GCNN_EMBEDDINGS_FILE,
    ZEOLITE_ALL_PRIOR_FILE,
    OSDA_CONFORMER_PRIOR_FILE_SIEVED,
    # TEMP_0D_PRIOR_FILE,
    OSDA_ZEO1_PRIOR_FILE,
)
from ntk_matrix_completion.utils.path_constants import (
    ZEOLITE_PRIOR_LOOKUP,
    OSDA_PRIOR_LOOKUP,
)

VALID_METHODS = {
    "identity",
    "CustomOSDA",
    "OldCustomOSDA",
    "CustomZeolite",
    "random",
    "CustomOSDAandZeolite",
    "CustomOSDAandZeoliteAsRows",
    "CustomOSDAVector",
    "ManualZeolite",
    "CustomOSDAandZeoliteAsSparseMatrix",
    "CustomZeoliteEmbeddings",
    # "CustomConformerOSDA",
}


ZEOLITE_PRIOR_MAP = {
    "*CTH": "CTH",
    "*MRE": "MRE",
    "*PCS": "PCS",
    "*STO": "STO",
    "*UOE": "UOE",
    "*BEA": "BEA",
    "_CON": "CON"
}


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    matrix.to_pickle(savepath)


# Turns out conformers are not very useful at all...
# So this function is not being used right now...
def load_conformer_priors(
    target_index,
    precomputed_file_name=OSDA_CONFORMER_PRIOR_FILE,
    identity_weight=0.01,
    normalize=True,
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    precomputed_prior = precomputed_prior.reindex(target_index)
    precomputed_prior = precomputed_prior.filter(items=["conformer"])
    exploded_prior = pd.DataFrame(
        columns=[
            "num_conformers",
            "mean_volume",
            "min_volume",
            "std_dev_volume",
            "mean_energy",
            "min_energy",
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
                    "mean_volume": 0.0,
                    "min_volume": 0.0,
                    "std_dev_volume": 0.0,
                    "mean_energy": 0.0,
                    "min_energy": 0.0,
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
                    # "std_dev_volume": np.std(conformer_properties["volumes"]),
                    # "mean_energy": np.mean(conformer_properties["energies"]),
                    # "min_energy": min(conformer_properties["energies"]),
                    # "std_dev_energy": np.std(conformer_properties["energies"]),
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
    precomputed_file_name=OSDA_PRIOR_FILE,
    identity_weight=0.01,
    normalize=True,
    other_prior_to_concat=None,  # OSDA_HYPOTHETICAL_PRIOR_FILE,  # OSDA_ZEO1_PRIOR_FILE
    replace_nan=0.0,
    already_exploded=False,
    clip_boundaries=(
        0.0,
        200,
    ),
):
    """
    Inputs:

        target index: List of substrates or ligands that we are interested in

        vector_feature: (str) name of the vector fingerprint we want. Currently 'getaway' or 'whims'

        precomputed_file_name: (str) Path of file containing the priors

        identity_weight: (float) priors are normalized to (1-identity_weight). See normalize

        normalize: If True, normalizes the priors across each row. See identity_weight

        other_prior_to_concat: For now, this looks like the prior file of new ligands/ substrates to predict for

        replace_nan: Replace NaN entries in the prior file with NaN. Stopgap, really shouldn't be used

        already_exploded: If False, grabs the vector fingerprint from the initial prior DataFrame

        clip_boundaries: Clip values in vector fingerprint. Currently manually set at (0.0, 200) for GETAWAY fingerprints,
            and needs to be corrected if we want to use WHIMS

    Returns:

        Normalized prior as a dataframe of series, each series containing the embeddings.
    """
    precomputed_prior = pd.read_pickle(precomputed_file_name)

    if other_prior_to_concat:
        big_precomputed_priors = pd.read_pickle(other_prior_to_concat)
        precomputed_prior = pd.concat([big_precomputed_priors, precomputed_prior])
        precomputed_prior = precomputed_prior[
            ~precomputed_prior.index.duplicated(keep="first")
        ]  # keeps entry from big_precomputed_priors if there are repeats

    if not already_exploded:  # grab just the embedding priors from the data

        def vector_explode(x):
            return pd.Series(x[vector_feature])

        precomputed_prior = precomputed_prior.apply(vector_explode, axis=1)
    exploded_prior = precomputed_prior.reindex(target_index)

    # TODO(Yitong): Exchanged .loc[] with .reindex() to fix a break. Was there a reason to use loc[]?
    # exploded_prior = precomputed_prior.loc[target_index] # target_index=all_data_df.index for gcnn embeddings

    if replace_nan is not None:
        print(
            f"[prior/load_vector_priors] WARNING: there are {exploded_prior.isna().sum().sum()} NaN entries in prior, filling with {replace_nan}"
        )
        exploded_prior = exploded_prior.fillna(replace_nan)

    exploded_prior = exploded_prior.clip(clip_boundaries[0], clip_boundaries[1])
    print(
        f"[prior/load_vector_priors] Sanity check on clipped values with boundaries {clip_boundaries}",
        exploded_prior.shape,
        exploded_prior.min().min(),
        exploded_prior.max().max(),
    )

    # Check & translate if anything is negative, as NTK prior only takes positive values
    lowest_value = exploded_prior.min().min()
    if lowest_value < 0:
        exploded_prior += -lowest_value

    # Normalize by the largest row sum
    if normalize:
        exploded_prior = exploded_prior / max(exploded_prior.sum(axis=1))
        exploded_prior = (1 - identity_weight) * exploded_prior

    return exploded_prior


# TODO(Yitong): Cache this. Mingrou add on: target_index is of type 'pandas.core.indexes.base.Index', not hashable
# snakeviz showing the reading of the prior file is taking the bulk of the time (and for some reason is alot slower on hartree than locally)
# TODO MINGROU need to find a way to cache this without breaking everything
# Index is not hashable (can use tuple), dict is not hashable (try frozendict)
# @lru_cache(maxsize=128)
def load_prior(
    target_index,
    column_weights,
    precomputed_file_name,
    identity_weight=0.01,
    normalize=True,
    prior_index_map=None,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
):
    """
    Inputs:

        target_index: List of substrates or ligands that we are interested in

        column_weights: (str) Path of json file. Gets parsed into a dictionary where keys are columns names and values are how much to weigh each feature

        precomputed_file_name: (str) Path of prior file to read from. See other_prior_to_concat

        identity_weight: (float) priors are normalized to (1-identity_weight). See normalize

        normalize: If True, normalizes the priors across each row. See identity_weight

        prior_index_map: (dict) Dictionary for renaming some entries in the index. See zeolite_prior method for example. Default is None

        other_prior_to_concat: (str) Path of second prior file to read from. In case of repeats, priors from the first file is kept

    Returns:

        A DataFrame with priors in the columns
    """
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    print(f"[prior/load_prior] Precomputed prior file {precomputed_file_name} read")

    if other_prior_to_concat:
        big_precomputed_priors = pd.read_pickle(other_prior_to_concat)
        precomputed_prior = pd.concat([big_precomputed_priors, precomputed_prior])
        precomputed_prior = precomputed_prior[
            ~precomputed_prior.index.duplicated(keep="first")
        ]
        print(f"Other prior to concat {other_prior_to_concat} read")

    if prior_index_map:  # rename some zeolites

        def x(i):
            return prior_index_map[i] if i in prior_index_map else i

        precomputed_prior.index = precomputed_prior.index.map(x)
    precomputed_prior = precomputed_prior.reindex(target_index)  
    # keeps rows with index in target_index, assigns NaN to other indices in target_index
    with open(column_weights, "r") as f:
        column_weights = json.load(f)
    precomputed_prior = precomputed_prior[list(column_weights.keys())]
    # precomputed_prior = precomputed_prior.filter(items=list(column_weights.keys()))
    precomputed_prior = precomputed_prior.apply(pd.to_numeric)

    print(
        f"[prior/load_prior] Precomputed prior has {precomputed_prior.isna().sum().sum()} NaN entries"
    )
    if precomputed_prior.isna().sum().sum() > 0:
        print("[prior/load_prior] WARNING: DATA POINTS WITH NAN ENTRIES. This will mess up your data, please go and check your prior source files. There should NOT be any NaN entries. Breakpoint now")
        breakpoint()
    # results = precomputed_prior.fillna(0.0)
    results = precomputed_prior.dropna()
    print(f"[prior/load_prior] Prior of shape {precomputed_prior.shape} now of size {results.shape}")

    if normalize:
        # Normalize down each column to between 0 & 1
        precomputed_prior = precomputed_prior.apply(lambda x: x / x.max())
        # Weigh each column, taking into account identity_weight to make sure
        # later when we add the identity matrix we don't go over 1.0 total per row...
        normalization_factor = sum(column_weights.values())
        results = precomputed_prior.apply(
            lambda x: x * column_weights[x.name] / normalization_factor
        )

    return (1 - identity_weight) * results


def osda_prior(
    all_data_df,
    identity_weight=0.01,
    osda_prior=OSDA_PRIOR_FILE,
    prior_map=OSDA_PRIOR_LOOKUP,
    normalize=True,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
):
    """
    Inputs:

        all_data_df: (Dataframe)

        identity_weight: (float) priors are normalized to (1-identity_weight). See normalize

        osda_prior: (str) Path of prior file to read from

        prior_map: For CustomZeolite, CustomOSDA, and CustomOSDAVector: how do you want to weight the
        individual descriptors? Default is to weight all descriptors equally (check out ZEOLITE_PRIOR_LOOKUP &
        OSDA_PRIOR_LOOKUP). This might be a good thing to tweak for calibrated ensemble uncertainty.

        normalize: If True, normalizes the priors across each row. See identity_weight

        other_prior_to_concat: (str) Path of second prior file to read from. In case of repeats, priors from the first file is kept

    Returns:

        Output from the load_prior method, which is a DataFrame
    """
    # some data files have SMILES and Zeolite as the index. TODO: hacky solution
    if type(all_data_df.index[0]) == tuple:
        target_index = all_data_df.reset_index()["SMILES"]
    else:
        target_index = tuple(all_data_df.index)

    return load_prior(
        target_index=target_index,
        column_weights=prior_map if prior_map is not None else OSDA_PRIOR_LOOKUP,
        precomputed_file_name=osda_prior,
        identity_weight=identity_weight,
        normalize=normalize,
        other_prior_to_concat=other_prior_to_concat,
    )


def osda_vector_prior(
    all_data_df,
    vector_feature="getaway",
    identity_weight=0.01,
    other_prior_to_concat=None,
    osda_prior_file=OSDA_PRIOR_FILE,
    second_vector_feature=None,
    osda_prior_map=OSDA_PRIOR_LOOKUP,
):
    """
    Inputs:

        all_data_df: (Dataframe)

        vector_feature: (str) name of the vector fingerprint we want. Currently 'getaway' or 'whims'

        identity_weight: (float) priors are normalized to (1-identity_weight). See normalize

        other_prior_to_concat: (str) Path of second prior file to read from. In case of repeats, priors from the first file is kept

        osda_prior_file: (str) Path of prior file to read from. See other_prior_to_concat

        second_vector_feature: hacky argument to incorporate a second vector fingerprint if desired

        osda_prior_map: (str) Path of json file that gets read into a dictionary where the keys are the OSDA
        features and the values their weights. For CustomZeolite, CustomOSDA, and CustomOSDAVector: how do you want
        to weight the individual descriptors? Default is to weight all descriptors equally (check out
        ZEOLITE_PRIOR_LOOKUP & OSDA_PRIOR_LOOKUP). This might be a good thing to tweak for calibrated ensemble
        uncertainty.

    Returns:

        Output from the load_prior method, which is a DataFrame
    """
    if type(all_data_df.index[0]) == tuple:
        target_index = all_data_df.reset_index()["SMILES"]
    else:
        target_index = tuple(all_data_df.index)

    prior = osda_prior(
        all_data_df,
        identity_weight,
        osda_prior=osda_prior_file,
        normalize=False,
        other_prior_to_concat=other_prior_to_concat,
        prior_map=osda_prior_map,
    )

    prior = prior.to_numpy()

    getaway_prior = load_vector_priors(
        # target_index=all_data_df.index,
        target_index=target_index,
        vector_feature=vector_feature,
        precomputed_file_name=osda_prior_file,
        identity_weight=identity_weight,
        normalize=False,
        other_prior_to_concat=other_prior_to_concat,
    )
    getaway_prior = getaway_prior.to_numpy()
    prior_stack = [getaway_prior, prior]

    if second_vector_feature is not None:  # TODO: hacky PAUSED HERE
        whims_prior = load_vector_priors(
            # all_data_df.index,
            target_index=target_index,
            vector_feature=second_vector_feature,
            precomputed_file_name=osda_prior_file,
            identity_weight=identity_weight,
            normalize=False,
            other_prior_to_concat=other_prior_to_concat,
        ).to_numpy()
        prior_stack.append(whims_prior)

    # Splitting the original prior and the vector prior 50-50
    # And a quick normalization by dividing by the sum of the row
    normalized_prior_stack = [
        p / (len(prior_stack) * max(p, key=sum).sum()) for p in prior_stack
    ]
    stacked = np.hstack(normalized_prior_stack)

    # Make room for the identity weight
    stacked = (1 - identity_weight) * stacked
    return stacked


def zeolite_prior(
    all_data_df,
    feature_lookup=ZEOLITE_PRIOR_LOOKUP,
    identity_weight=0.01,
    normalize=True,
    precomputed_file_name=HANDCRAFTED_ZEOLITE_PRIOR_FILE,
    # PERSISTENCE_ZEOLITE_PRIOR_FILE,  # includes handcrafted
    # ZEOLITE_GCNN_EMBEDDINGS_FILE,
    # ZEOLITE_PRIOR_FILE, # includes handcrafted but missing a few features
    # ZEOLITE_ALL_PRIOR_FILE, # handcraft, persistent, gcnn
    # TEMP_0D_PRIOR_FILE,
):
    """
    Takes in all the priors and their weights (feature_lookup) and returns priors that are
    normalized (within each column) to (1-identity_weight).
    """
    return load_prior(
        target_index=tuple(all_data_df.index),
        column_weights=feature_lookup,
        precomputed_file_name=precomputed_file_name,
        identity_weight=identity_weight,
        normalize=normalize,
        prior_index_map=ZEOLITE_PRIOR_MAP,
        other_prior_to_concat=None,  # ZEO_1_PRIOR,
    )


def zeolite_vector_prior(
    all_data_df,
    prior_map,
    identity_weight=0.01,
    normalize=True,
    vector_prior_file=ZEOLITE_GCNN_EMBEDDINGS_FILE,
    prior_file=HANDCRAFTED_ZEOLITE_PRIOR_FILE,
):
    """
    Takes in all priors and their weights (prior_map), and returns prior normalized to a
    normalization factor (default 0.99). This function allows for vector priors.
    """
    gcnn_priors = load_vector_priors(
        target_index=all_data_df.index,
        vector_feature="feature_set",
        precomputed_file_name=vector_prior_file,
        normalize=normalize,
        other_prior_to_concat=None,
        already_exploded=True,
        identity_weight=identity_weight,
    ).to_numpy()
    handcrafted_zeolite_priors = zeolite_prior(
        all_data_df, feature_lookup=prior_map, precomputed_file_name=prior_file
    ).to_numpy()
    # weigh handcrafted and gcnn equally
    normalized_gcnn_priors = gcnn_priors / (2 * max(gcnn_priors, key=sum).sum())
    normalized_handcrafted_priors = handcrafted_zeolite_priors / (
        2 * max(handcrafted_zeolite_priors, key=sum).sum()
    )
    stacked = np.hstack([normalized_gcnn_priors, normalized_handcrafted_priors])
    return (1 - identity_weight) * stacked


def osda_zeolite_combined_prior(
    all_data_df,
    identity_weight=0.01,
    normalize=True,
    stack=True,
    osda_prior_file=OSDA_PRIOR_FILE,
    zeolite_prior_file=HANDCRAFTED_ZEOLITE_PRIOR_FILE,  # ZEOLITE_PRIOR_FILE
    osda_prior_map=OSDA_PRIOR_LOOKUP,
    zeolite_prior_map=ZEOLITE_PRIOR_LOOKUP,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
):
    """
    Questions: TODO

    1. Why do we put identity_weight into each individual prior loading, and then again at the end of this method?
    2. Too sleepy to make sense of the normalization if stack=True. Please check again.
    3. load_prior drops NaN rows like nobody's business, and then these 3 priors come back with different shapes - very bad. I don't think the NaN should be dropped or converted, but that makes weighting/ normalization impossible. 

    """
    osda_prior = load_prior(
        target_index=tuple([i[0] for i in all_data_df.index]),
        column_weights=osda_prior_map,
        precomputed_file_name=osda_prior_file,
        identity_weight=identity_weight,
        normalize=normalize,
        other_prior_to_concat=other_prior_to_concat
    )
    osda_vector_prior = load_vector_priors(
        target_index=[i[0] for i in all_data_df.index],
        vector_feature="getaway",
        precomputed_file_name=osda_prior_file,
        identity_weight=identity_weight,
        normalize=normalize,
        other_prior_to_concat=None,
    )
    zeolite_prior = load_prior(
        target_index=tuple([i[1] for i in all_data_df.index]),
        column_weights=zeolite_prior_map,
        precomputed_file_name=zeolite_prior_file,
        identity_weight=identity_weight,
        normalize=normalize,
        prior_index_map=ZEOLITE_PRIOR_MAP,
        other_prior_to_concat=None,
    )

    print("[prior/osda_zeolite_combined_prior] columns are")
    print(osda_prior.columns)
    print(osda_vector_prior.columns)
    print(zeolite_prior.columns)

    osda_prior = osda_prior.to_numpy()
    osda_vector_prior = osda_vector_prior.to_numpy()
    zeolite_prior = zeolite_prior.to_numpy()

    if not stack:
        return (osda_prior, osda_vector_prior, zeolite_prior)
    else:
        normalized_osda_vector_prior = osda_vector_prior / (
            3 * max(osda_vector_prior, key=sum).sum()
        )
        normalized_osda_prior = osda_prior / (3 * max(osda_prior, key=sum).sum())
        normalized_zeolite_prior = zeolite_prior / (
            3 * max(zeolite_prior, key=sum).sum()
        )
        stacked = np.hstack(
            [
                normalized_osda_vector_prior,
                normalized_osda_prior,
                normalized_zeolite_prior,
            ]
        )
        stacked = np.nan_to_num(stacked)

    return (1 - identity_weight) * stacked


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
    normalization_factor=0.001,
    osda_prior_map=OSDA_PRIOR_LOOKUP,
    zeolite_prior_map=ZEOLITE_PRIOR_LOOKUP,
    all_data=None,
    test_train_axis=0,
    stack_combined_priors=True,
    osda_prior_file=OSDA_PRIOR_FILE,
    zeolite_prior_file=HANDCRAFTED_ZEOLITE_PRIOR_FILE,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
):
    """
    train: training set

    test: test set

    method: which prior do you want to use? Hint: probably CustomOSDAVector or CustomZeolite

    normalization_factor: what to normalize the identity matrix we concat to. This is necessary
    to specify since we need all rows in the prior to sum to 1 & if the identity matrix is going to have
    value 0.001 then the rest of the row must sum to at most 0.999.

    osda_prior_map: (str) Path of json file that gets read into a dictionary where the keys are the OSDA
    features and the values their weights. For CustomZeolite, CustomOSDA, and CustomOSDAVector: how do you want
    to weight the individual descriptors? Default is to weight all descriptors equally (check out
    ZEOLITE_PRIOR_LOOKUP & OSDA_PRIOR_LOOKUP). This might be a good thing to tweak for calibrated ensemble
    uncertainty.

    zeolite_prior_map: (str) Path of json file that gets read into a dictionary where the keys are the zeolite
    features and the values their weights. For CustomZeolite, CustomOSDA, and CustomOSDAVector: how do you want
    to weight the individual descriptors? Default is to weight all descriptors equally (check out
    ZEOLITE_PRIOR_LOOKUP & OSDA_PRIOR_LOOKUP). This might be a good thing to tweak for calibrated ensemble
    uncertainty.

    all_data: Use this to provide all the data instead of separately specifying test & train sets. This is for
    when you're no longer testing with 10-fold cross validation; when you are ready to take your method and
    infer energies on a new distribution & want to use all of your data to train the ML model.

    test_train_axis: Originally created to specify joining test or train by row or column. Likely obsolete.

    stack_combined_priors: This is only for the two tower NN where we would like to separate the
    embeddings for zeolite and OSDAs.

    osda_prior_file: (str or list) Path to prior file or list of paths to prior files. Used in osda_vector_prior in CustomOSDAVector and in osda_zeolite_combined_prior in CustomOSDAandZeoliteAsRows.

    zeolite_prior_file: (str or list) Path to prior file or list of paths to prior files. Used in osda_zeolite_combined_prior in CustomOSDAandZeoliteAsRows, and in CustomZeoliteVector
    """
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
        # This is our baseline prior.
        prior = np.eye(all_data.shape[0])
        return prior

    elif method == "CustomOSDA":
        # CustomOSDA uses only the handcrafted OSDA descriptors
        prior = osda_prior(
            all_data_df=all_data_df,
            identity_weight=normalization_factor,
            osda_prior=osda_prior_file,
            prior_map=osda_prior_map,
        ).to_numpy()
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    elif method == "CustomOSDAVector":
        # CustomOSDAVector takes all of the handcrafted OSDA descriptors
        # and appends it to the GETAWAY prior
        prior = osda_vector_prior(
            all_data_df=all_data_df,
            vector_feature="getaway",
            identity_weight=normalization_factor,
            osda_prior_file=osda_prior_file,
            other_prior_to_concat=other_prior_to_concat,  # None is default
            osda_prior_map=osda_prior_map,
        )
        if normalization_factor:
            return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
        return prior

    elif method == "CustomZeolite":
        # CustomZeolite takes all of the priors from the data file specified in zeolite_prior()
        prior = zeolite_prior(
            all_data_df, 
            feature_lookup=zeolite_prior_map, 
            precomputed_file_name=zeolite_prior_file
        ).to_numpy()
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    elif method == "CustomZeoliteEmbeddings":
        prior = zeolite_vector_prior(
            all_data_df,
            zeolite_prior_map=zeolite_prior_map,
            identity_weight=normalization_factor,
            vector_prior_file=zeolite_prior_file[1],  # ZEOLITE_GCNN_EMBEDDINGS_FILE,
            prior_file=zeolite_prior_file[0],
        )
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    # This one is for the failed experiment. i.e. not updated with code changes
    # elif method == "CustomOSDAandZeolite":
    #     osda_axis1_lengths = osda_prior(
    #         all_data_df, column_name="Axis 1 (Angstrom)", normalize=False
    #     ).to_numpy()
    #     zeolite_sphere_diameters = zeolite_prior(all_data_df).to_numpy()

    #     prior = np.zeros((len(osda_axis1_lengths), len(zeolite_sphere_diameters)))
    #     for i, osda_length in enumerate(osda_axis1_lengths):
    #         for j, zeolite_sphere_diameter in enumerate(zeolite_sphere_diameters):
    #             prior[i, j] = zeolite_sphere_diameter - osda_length

    #     if prior.min() < 0:
    #         prior = prior - prior.min()
    #     # Normalize prior across its rows:
    #     max = np.reshape(np.repeat(prior.sum(axis=1), prior.shape[1]), prior.shape)
    #     prior = prior / max
    #     prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
    #     return prior

    # This is the one for really skinny Matrices
    elif method == "CustomOSDAandZeoliteAsRows":
        prior = osda_zeolite_combined_prior(
            all_data_df,
            normalize=False,
            stack=stack_combined_priors,
            osda_prior_file=osda_prior_file,
            zeolite_prior_file=zeolite_prior_file,
            identity_weight=normalization_factor,
            osda_prior_map=osda_prior_map,
            zeolite_prior_map=zeolite_prior_map,
            other_prior_to_concat=other_prior_to_concat,
        )
        if stack_combined_priors:
            # For now remove the identity concat to test eigenpro
            np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
        return prior

    # This is the one for really skinny Matrices with sparse matrices.
    elif method == "CustomOSDAandZeoliteAsSparseMatrix":
        prior = csc_matrix(osda_zeolite_combined_prior(
            all_data_df, 
            normalize=False,
            stack=stack_combined_priors,
            osda_prior_file=osda_prior_file,
            zeolite_prior_file=zeolite_prior_file,
            identity_weight=normalization_factor,
            osda_prior_map=osda_prior_map,
            zeolite_prior_map=zeolite_prior_map,
            other_prior_to_concat=other_prior_to_concat))
        return sp.sparse.hstack(
            [prior, normalization_factor * sp.sparse.identity(all_data.shape[0])]
        )

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
    normalize(prior, axis=1, copy=False)
    return prior
