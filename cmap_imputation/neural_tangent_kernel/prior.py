import numpy as np
from numpy.linalg import norm
import pandas as pd
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    "skinny_identity"
}

def osda_prior_helper(all_data_df, column_name = 'Volume (Angstrom3)', normalize = True):
    # TODO: this is horrible. PLEASE PLEASE FIX!
    # TODO TODO: this is so so so gross...
    rafa_data = "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/completeZeoliteData.pkl"
    if not os.path.exists(rafa_data):
        raise AssertionError(f"Path to matrix, {rafa_data} does not exist")
    osda_df = pd.read_pickle(rafa_data)
    # zeolite_data.set_index('SMILES')
    metadata_by_osda = osda_df.set_index("SMILES").T.to_dict()
    # 1108 unique volume values...
    max_volume = max(
        [dict[column_name] for dict in metadata_by_osda.values()]
    ) if normalize else 1.0
    return np.array([
        [1.0 * metadata_by_osda[smile][column_name] / max_volume]
        for smile in all_data_df.reset_index().SMILES
    ])

# TODO: make this a frozen dict (https://pypi.org/project/frozendict/)
# This is a lookup for all the maximum sphere diameters of zeolites that are not
# present in olivetti et al.'s data.
ZEOLITE_SPHERE_DIAMETER_LOOKUP = {
    'AET': 8.41,
    'AFG': 6.37,
    'AFT': 7.75,
    'BOF': 5.58,
    'BRE': 5.29,
    'CAN': 6.27,
    'DAC': 5.28,
    'EPI': 5.47,
    'EWO': 5.41,
    'FAR': 6.36,
    'FRA': 6.67,
    'GIU': 6.32,
    'HEU': 5.97,
    'IFY': 6.94,
    'JNT': 4.72,
    'JOZ': 4.92,
    'JSN': 5.12,
    'JSR': 7.83,
    'LIO': 6.05,
    'LOV': 5.15,
    'LTF': 8.16,
    'MAR': 6.35,
    'MEP': 5.49,
    'MSO': 7.23,
    'NPO': 4.23,
    'NPT': 10.28,
    'OBW': 9.26,
    'OKO': 6.7,
    'OSI': 6.66,
    'OSO': 6.07,
    'PCR': 6.03,
    'PCS': 6.84,
    'POR': 6.14,
    'SBS': 11.45,
    'SBT': 11.17,
    'TER': 6.94,
    'TOL': 6.37,
    'TSC': 16.45,
    'UEI': 5.6,
    'VNI': 4.8,
}

OLIVETTI_CODE_SWITCH = {
    '*BEA': 'BEA', 
    '*CTH': 'CTH', 
    '*MRE': 'MRE', 
    '*STO': 'STO', 
    '*UOE': 'UOE', 
}

def zeolite_prior_helper(target_codes, normalize = True):
    # TODO: this is horrible. PLEASE PLEASE FIX!
    # TODO TODO: this is so so so gross...
    olivetti_table = "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/Jensen_et_al_CentralScience_OSDA_Zeolite_data.tsv"
    olivetti_df = pd.read_csv(olivetti_table, index_col=0, delimiter="\t")
    metadata_by_zeolite = olivetti_df.set_index("Code").T.to_dict()
    for olivetti_code, target_code in OLIVETTI_CODE_SWITCH.items():
        metadata_by_zeolite[target_code] = metadata_by_zeolite.pop(olivetti_code)

    # Double check that all target codes have corresponding data...
    lookup_codes = list(metadata_by_zeolite.keys()) + list(
        ZEOLITE_SPHERE_DIAMETER_LOOKUP.keys()
    )
    assert (
        len(np.setdiff1d(target_codes, lookup_codes)) == 0
    ), "all target codes must be covered."

    prior = [
        metadata_by_zeolite[code]["inc_diameter"]
        if code in metadata_by_zeolite
        else ZEOLITE_SPHERE_DIAMETER_LOOKUP[code]
        for code in target_codes
    ]
    max_diameter = max(prior) if normalize else 1.0
    prior = np.array([
        [1.0 * diameter  / max_diameter]
        for diameter in prior
    ])
    return prior


def make_prior(train, only_train, test, method='identity', normalization_factor=1.5, test_train_axis=0):
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

    if method == 'identity':
        prior = np.eye(all_data.shape[0])
        return prior

    if method == 'skinny_identity':
        prior = np.ones((all_data.shape[0],1))
        return prior

    elif method == 'OneHotDrug':
        encoder = OneHotEncoder()
        prior = encoder.fit_transform(all_data_df.reset_index().intervention.to_numpy().reshape(-1, 1)).toarray()

 
    elif method == "OneHotOSDA":
        encoder = OneHotEncoder()
        # Isn't this just the same as an identity matrix???
        prior = encoder.fit_transform(
            all_data_df.reset_index().SMILES.to_numpy().reshape(-1, 1)
        ).toarray()

    elif method == "CustomOSDA":
        prior = osda_prior_helper(
            all_data_df, column_name="Axis 1 (Angstrom)", normalize=True
        )

    elif method == "CustomZeolite":
        prior = zeolite_prior_helper(target_codes=all_data_df.index, normalize=True)

    elif method == "CustomOSDAandZeolite":
        # osda_prior.shape = (1194, 1) ... a prior over all the osda volumes...
        osda_axis1_lengths = osda_prior_helper(
            all_data_df, column_name="Axis 1 (Angstrom)", normalize=False
        )
        # zeolite_prior.shape = (1, 209)... a prior over all the possible zeolite sphere diameters...
        zeolite_sphere_diameters = zeolite_prior_helper(
            target_codes=(all_data_df.columns).to_numpy(), normalize=False
        )

        prior = np.zeros((len(osda_axis1_lengths), len(zeolite_sphere_diameters)))
        for i, osda_length in enumerate(osda_axis1_lengths):
            for j, zeolite_sphere_diameter in enumerate(zeolite_sphere_diameters):
                prior[i, j] = zeolite_sphere_diameter - osda_length

        if prior.min() < 0:
            prior = prior - prior.min()
        # TODO: is this necessary to normalize all of the values in prior to maximum?
        prior = prior / prior.max()
        return prior
        # TODO: DO I ALSO NEED to normalize all of the rows to 1... DO I???
        # plot_matrix(prior, 'prior', vmin=0, vmax=1)
        # prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
        # return prior

    # this is the one for really skinny Matrices
    elif method == 'CustomOSDAandZeoliteAsRows':
        # osda_prior.shape = (1194, 1) ... a prior over all the osda volumes...
        osda_axis1_lengths = osda_prior_helper(
            all_data_df, column_name="Axis 1 (Angstrom)", normalize=False
        )
        # zeolite_prior.shape = (1, 209)... a prior over all the possible zeolite sphere diameters...
        zeolite_sphere_diameters = zeolite_prior_helper(
            target_codes=(all_data_df.columns).to_numpy(), normalize=False
        )
        pdb.set_trace()
    elif method == 'OneHotCell':
        encoder = OneHotEncoder()
        prior = encoder.fit_transform(all_data_df.reset_index().unit.to_numpy().reshape(-1, 1)).toarray()

    elif method == 'OneHotCombo':
        drug_scale_factor = 0.75
        cell_encoding = {}
        drug_encoding = {}

        for idx, unique_cell in enumerate(set(train.index.get_level_values('unit'))
                                    .union(set(test.index.get_level_values('unit')))):
            cell_encoding[unique_cell] = idx

        for idx, unique_drug in enumerate(set(train.index.get_level_values('intervention'))
                                    .union(set(test.index.get_level_values('intervention')))):
            drug_encoding[unique_drug] = idx

        prior = np.zeros((all_data.shape[0], len(cell_encoding) + len(drug_encoding)))

        x_row_index = 0
        for index, row in train.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            prior[x_row_index, cell_encoding[Cell_id]] = 1
            prior[x_row_index, len(cell_encoding) + drug_encoding[Drug_id]] = drug_scale_factor
            x_row_index += 1

        for index, row in test.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            prior[x_row_index, cell_encoding[Cell_id]] = 1
            prior[x_row_index, len(cell_encoding) + drug_encoding[Drug_id]] = drug_scale_factor
            x_row_index += 1

    elif method == 'random':
        dim = 100
        prior = np.random.rand(all_data.shape[0], dim)

    elif method == 'only_train_cell_oneHot':
        cell_scale_factor = 0.75

        cell_encoding = {}
        only_train = only_train.reset_index().groupby("intervention").agg('mean')
        dim = all_data.shape[1]

        encoding = only_train.to_numpy()
        encoding = np.vstack([encoding, np.average(encoding, axis=0)])
        only_train.reset_index(inplace=True)

        for idx, unique_cell in enumerate(set(train.index.get_level_values('unit'))
                                    .union(set(test.index.get_level_values('unit')))):
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
                prior[x_row_index, cell_encoding[Cell_id]] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :])
            prior[x_row_index, len(cell_encoding):] = encoding[drug_idx, :]

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
                prior[x_row_index, cell_encoding[Cell_id]] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :])
            prior[x_row_index, len(cell_encoding):] = encoding[drug_idx, :]

            x_row_index += 1

    elif method == 'only_train_cell_average':
        cell_scale_factor = 0.75

        only_train = only_train.reset_index().groupby("intervention").agg('mean')
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

            cell_repr = train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
            cell_repr /= np.linalg.norm(cell_repr)
            prior[x_row_index, :dim] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
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

            cell_repr = train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
            cell_repr /= np.linalg.norm(cell_repr)
            prior[x_row_index, :dim] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
            prior[x_row_index, dim:] = encoding[drug_idx, :]

            x_row_index += 1

    elif method == 'custom':
        raise NotImplementedError("Custom prior not implemented")

    if test_train_axis == 0:
        prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
    elif test_train_axis == 1:
        prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[1])[0:all_data.shape[0], 1:]])
        # TODO: this is quite gross... is this the right way to be making this?
        # TODO: there is a better way to do this... add the bottom to the prior col first then just hstack the eye once.
        # prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[1])[0:all_data.shape[0]]])
        # lower_buffer = np.eye(all_data.shape[1])[all_data.shape[0]:]
        # lower_buffer = np.hstack([np.zeros((lower_buffer.shape[0], 1)), lower_buffer])
        # prior = np.vstack([prior, lower_buffer])
    # TODO: big debate... do I like this normalize?? probably not...
    normalize(prior, axis=1, copy=False)
    return prior
