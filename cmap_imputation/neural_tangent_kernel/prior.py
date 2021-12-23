import numpy as np
from numpy.linalg import norm
import pandas as pd
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
from rdkit import Chem
from rdkit.Chem import rdFreeSASA, Descriptors, Descriptors3D, AllChem, Draw
from functools import lru_cache
from tqdm import tqdm


from sklearn.preprocessing import normalize, OneHotEncoder

VALID_METHODS = {
    "identity",
    "OneHotOSDA",
    "OneHotDrug",
    "OneHotCell",
    "OneHotCombo",
    "CustomOSDA",
    "CustomZeolite",
    "custom",
    "random",
    "only_train_cell_oneHot",
    "only_train_cell_average",
    "CustomOSDAandZeolite",
    "CustomOSDAandZeoliteAsRows",
    "skinny_identity",
}

ZEOLITE_PRIOR_LOOKUP = {
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
    "alpha": 1.0,
    "betta": 1.0,
    "gamma": 1.0,
    "volume": 1.0,
    "rdls": 1.0,
    "framework_density": 1.0,
    "td_10": 1.0,
    "td": 1.0,
    "included_sphere_diameter": 1.0,
    "diffused_sphere_diameter_a": 1.0,
    "diffused_sphere_diameter_b": 1.0,
    "diffused_sphere_diameter_c": 1.0,
    "accessible_volume": 1.0,
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
ZEOLITE_PRIOR_FILE = "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/scraped_zeolite_data.pkl"
OSDA_PRIOR_FILE = "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/precomputed_OSDA_prior.pkl"

ZEOLITE_PRIOR_MAP = {
    "*CTH": "CTH",
    "*MRE": "MRE",
    "*PCS": "PCS",
    "*STO": "STO",
    "*UOE": "UOE",
    "*BEA": "BEA",
}

# is_NaN = precomputed_prior.isnull()
# (Pdb) row_has_NaN = is_NaN.any(axis=1)
# (Pdb) rows_with_NaN = precomputed_prior[row_has_NaN]


def load_prior(
    target_index,
    column_weights,
    precomputed_file_name,
    identity_weight=0.01,
    normalize=True,
    prior_index_map=None,
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    if prior_index_map:
        x = lambda i: prior_index_map[i] if i in prior_index_map else i
        precomputed_prior.index = precomputed_prior.index.map(x)
    precomputed_prior = precomputed_prior.reindex(target_index)
    precomputed_prior = precomputed_prior.apply(pd.to_numeric)
    precomputed_prior = precomputed_prior.fillna(0.0)

    # Normalize down each column to between 0 & 1
    if normalize:
        precomputed_prior = precomputed_prior.apply(lambda x: x / x.max(), axis=0)
    # Now time to weigh each column, taking into account identity_weight to make sure
    # later when we add the identity matrix we don't go over 1.0 total per row...
    precomputed_prior = precomputed_prior.filter(items=list(column_weights.keys()))
    normalization_factor = sum(column_weights.values()) + identity_weight
    results = precomputed_prior.apply(
        lambda x: x * column_weights[x.name] / normalization_factor, axis=0
    )
    return results.to_numpy()


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


def zeolite_prior(
    all_data_df,
    feature_lookup,
    identity_weight=0.01,
    normalize=True,
):
    return load_prior(
        all_data_df.index,
        ZEOLITE_PRIOR_LOOKUP if not feature_lookup else feature_lookup,
        ZEOLITE_PRIOR_FILE,
        identity_weight,
        normalize,
        ZEOLITE_PRIOR_MAP,
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
    only_train,
    test,
    method="identity",
    normalization_factor=1.5,
    test_train_axis=0,
    feature=None,
):
    assert method in VALID_METHODS, f"Invalid method used, pick one of {VALID_METHODS}"
    if test_train_axis == 0:
        all_data = np.vstack((train.to_numpy(), test.to_numpy()))
        all_data_df = pd.concat([train, test])
    elif test_train_axis == 1:
        all_data = np.hstack((train.to_numpy(), test.to_numpy()))
        all_data_df = pd.concat([train, test], 1)
    else:
        # TODO: clean this up...
        all_data = None
        all_data_df = pd.concat([train, test], test_train_axis)

    prior = None

    if method == "identity":
        prior = np.eye(all_data.shape[0])
        return prior

    if method == "skinny_identity":
        # prior = np.ones((all_data.shape[0], 1))
        prior = np.eye(all_data.shape[0])
        return prior

    elif method == "OneHotDrug":
        encoder = OneHotEncoder()
        prior = encoder.fit_transform(
            all_data_df.reset_index().intervention.to_numpy().reshape(-1, 1)
        ).toarray()

    elif method == "OneHotOSDA":
        encoder = OneHotEncoder()
        # Isn't this just the same as an identity matrix???
        prior = encoder.fit_transform(
            all_data_df.reset_index().SMILES.to_numpy().reshape(-1, 1)
        ).toarray()

    elif method == "CustomOSDA":
        # Volume (Angstrom3)
        # Axis 2 (Angstrom)
        # all_data_df = all_data_df.head(100)
        # prior = osda_prior_helper(all_data_df)
        # why_this_prior = secondary_osda_prior_helper(all_data_df)

        prior = osda_prior(all_data_df, normalization_factor)
        # We're just tacking on the I matrix without doing our due diligence to make sure
        # it doesn't go over...
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    elif method == "CustomZeolite":
        prior = zeolite_prior(all_data_df, feature)

    elif method == "CustomOSDAandZeolite":
        # osda_prior.shape = (1194, 1) ... a prior over all the osda volumes...
        osda_axis1_lengths = osda_prior(
            all_data_df, column_name="Axis 1 (Angstrom)", normalize=False
        )
        # zeolite_prior.shape = (1, 209)... a prior over all the possible zeolite sphere diameters...
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

    # this is the one for really skinny Matrices
    elif method == "CustomOSDAandZeoliteAsRows":
        prior = osda_zeolite_combined_prior(all_data_df, normalize=True)
        pdb.set_trace()
        # osda_prior.shape = (1194, 1) ... a prior over all the osda volumes...
        # osda_axis1_lengths = osda_prior(all_data_df, normalize=False)
        # # This is gross... If you're going to use this then fix this and grab zeolite names based on the index name.
        # zeolites = [v[1] for v in all_data_df.index.values]

        # # TODO: this might need to be fixed...
        # zeolite_sphere_diameters = zeolite_prior(zeolites)
        # stacked = np.hstack([zeolite_sphere_diameters, osda_axis1_lengths])
        # prior = stacked / (stacked.max() * 1.111)
        # return np.hstack([prior, 0.098 * np.eye(all_data.shape[0])])
    elif method == "OneHotCell":
        encoder = OneHotEncoder()
        prior = encoder.fit_transform(
            all_data_df.reset_index().unit.to_numpy().reshape(-1, 1)
        ).toarray()

    elif method == "OneHotCombo":
        drug_scale_factor = 0.75
        cell_encoding = {}
        drug_encoding = {}

        for idx, unique_cell in enumerate(
            set(train.index.get_level_values("unit")).union(
                set(test.index.get_level_values("unit"))
            )
        ):
            cell_encoding[unique_cell] = idx

        for idx, unique_drug in enumerate(
            set(train.index.get_level_values("intervention")).union(
                set(test.index.get_level_values("intervention"))
            )
        ):
            drug_encoding[unique_drug] = idx

        prior = np.zeros((all_data.shape[0], len(cell_encoding) + len(drug_encoding)))

        x_row_index = 0
        for index, row in train.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            prior[x_row_index, cell_encoding[Cell_id]] = 1
            prior[
                x_row_index, len(cell_encoding) + drug_encoding[Drug_id]
            ] = drug_scale_factor
            x_row_index += 1

        for index, row in test.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            prior[x_row_index, cell_encoding[Cell_id]] = 1
            prior[
                x_row_index, len(cell_encoding) + drug_encoding[Drug_id]
            ] = drug_scale_factor
            x_row_index += 1

    elif method == "random":
        dim = 100
        prior = np.random.rand(all_data.shape[0], dim)

    elif method == "only_train_cell_oneHot":
        cell_scale_factor = 0.75

        cell_encoding = {}
        only_train = only_train.reset_index().groupby("intervention").agg("mean")
        dim = all_data.shape[1]

        encoding = only_train.to_numpy()
        encoding = np.vstack([encoding, np.average(encoding, axis=0)])
        only_train.reset_index(inplace=True)

        for idx, unique_cell in enumerate(
            set(train.index.get_level_values("unit")).union(
                set(test.index.get_level_values("unit"))
            )
        ):
            cell_encoding[unique_cell] = idx

        prior = np.zeros((all_data.shape[0], len(cell_encoding) + dim))

        x_row_index = 0
        failed = 0
        for index, row in train.iterrows():
            Cell_id = str(index[0]) if str(index[0]) in cell_encoding else None
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            if Cell_id is not None:
                prior[
                    x_row_index, cell_encoding[Cell_id]
                ] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :])
            prior[x_row_index, len(cell_encoding) :] = encoding[drug_idx, :]

            x_row_index += 1

        for index, row in test.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            if Cell_id is not None:
                prior[
                    x_row_index, cell_encoding[Cell_id]
                ] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :])
            prior[x_row_index, len(cell_encoding) :] = encoding[drug_idx, :]

            x_row_index += 1

    elif method == "only_train_cell_average":
        cell_scale_factor = 0.75

        only_train = only_train.reset_index().groupby("intervention").agg("mean")
        dim = all_data.shape[1]

        encoding = only_train.to_numpy()
        encoding = np.vstack([encoding, np.average(encoding, axis=0)])
        only_train.reset_index(inplace=True)

        prior = np.zeros((all_data.shape[0], dim + dim))

        x_row_index = 0
        failed = 0
        for index, row in train.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            cell_repr = (
                train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
            )
            cell_repr /= np.linalg.norm(cell_repr)
            prior[x_row_index, :dim] = (
                cell_scale_factor * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
            )
            prior[x_row_index, dim:] = encoding[drug_idx, :]

            x_row_index += 1

        for index, row in test.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            cell_repr = (
                train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
            )
            cell_repr /= np.linalg.norm(cell_repr)
            prior[x_row_index, :dim] = (
                cell_scale_factor * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
            )
            prior[x_row_index, dim:] = encoding[drug_idx, :]

            x_row_index += 1

    elif method == "custom":
        raise NotImplementedError("Custom prior not implemented")

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
