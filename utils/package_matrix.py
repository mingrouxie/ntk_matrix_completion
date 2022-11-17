import sys
import pathlib
import os
import pdb
import torch
import numpy as np
import pandas as pd
from enum import Enum

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from ntk_matrix_completion.utils.path_constants import (
    BINDING_GROUND_TRUTH,
    TEMPLATING_GROUND_TRUTH,
    BINDING_CSV,
    OSDA_PRIOR_FILE,
)
from ntk_matrix_completion.features.prior import make_prior
from ntk_matrix_completion.features.precompute_osda_priors import (
    precompute_priors_for_780K_Osdas,
)
from ntk_matrix_completion.utils.random_seeds import PACKAGE_LOADER_SEED
from ntk_matrix_completion.utils.non_binding import NonBinding, fill_non_bind

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utils.utilities import (
    save_matrix,
)

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)


class Energy_Type(Enum):
    TEMPLATING = 1
    BINDING = 2


def format_ground_truth_pkl():
    ground_truth_df = pd.read_csv(BINDING_CSV, index_col=0)
    binding_matrix = ground_truth_df.pivot(
        index="SMILES", columns="Zeolite", values="Binding (SiO2)"
    )
    print(
        "The binding matrix has these many values: ",
        binding_matrix.notna().sum().sum(),
        " out of these many total cells",
        binding_matrix.isna().sum().sum() + binding_matrix.notna().sum().sum(),
    )
    save_matrix(binding_matrix, BINDING_GROUND_TRUTH)

    templating_matrix = ground_truth_df.pivot(
        index="SMILES", columns="Zeolite", values="Templating"
    )
    print(
        "The templating matrix has these many values: ",
        templating_matrix.notna().sum().sum(),
        " out of these many total cells",
        templating_matrix.isna().sum().sum() + templating_matrix.notna().sum().sum(),
    )

    save_matrix(templating_matrix, TEMPLATING_GROUND_TRUTH)


"""
energy_type: Do you want binding energy or templating energy?

desired_shape: 2d shape (e.g., (100,100)) to pare the matrix down into. 
really only useful when the matrix is so big that you can only work
on a section of it at a time (e.g., with skinny_ntk).

minimum_row_length: Rows that have < minimum_row_length energies are filtered out.
2 is the default since we need at least 2 values to calculate rmse and Spearman correlation.

transpose: Applies a transpose on the energies before doing the minimum_row_length filtering.
Aka do you want to predict new OSDAs or new zeolites? 

arbitrary_high_energy: You can give it some arbitrary high energy to fill non-binding cells.
If nothing is given then it takes the row mean.
"""
# minimum_row_length default set to 2 so we can perform rmse and Spearman correlation.
def get_ground_truth_energy_matrix(
    energy_type=Energy_Type.BINDING,
    desired_shape=None,
    minimum_row_length=2,
    transpose=False,
    prune_index=None,
    non_binding=NonBinding.ROW_MEAN,
):
    if energy_type == Energy_Type.TEMPLATING:
        ground_truth = pd.read_pickle(TEMPLATING_GROUND_TRUTH)
    elif energy_type == Energy_Type.BINDING:
        ground_truth = pd.read_pickle(BINDING_GROUND_TRUTH)
    else:
        # REFERENCE format_ground_truth_pkl()
        raise ValueError(
            "Sorry, but if you want to use a different ground truth for the energy then create the matrix first."
        )
    if transpose:
        ground_truth = ground_truth.T
        ground_truth.index.name = "Zeolite"
    breakpoint()
    if prune_index is not None:
        ground_truth = ground_truth.loc[prune_index]

    # Filter down to desired_shape & filter by min_row_length
    if desired_shape is not None:
        assert (
            desired_shape <= ground_truth.shape
        ), "Can't be asking for a shape bigger than the full matrix"
    else:
        desired_shape = ground_truth.shape

    # Reshape ground_truth & enforce minimum_row_length
    shaped_ground_truth = pd.DataFrame()
    shaped_rows = []
    for _index, row in ground_truth.iterrows():
        if len(shaped_rows) >= desired_shape[0]:
            break
        shaped_row = row[: desired_shape[1]]
        if len(shaped_row.dropna()) < minimum_row_length:
            continue
        shaped_rows.append(shaped_row)
    shaped_ground_truth = pd.concat(shaped_rows, axis=1).T
    shaped_ground_truth.index.name = ground_truth.index.name
    ground_truth = shaped_ground_truth

    # Create the mask (binary_data) for binding_energies
    binary_data = ground_truth.fillna(0)
    binary_data[binary_data != 0] = 1

    # treat non-binding entries
    ground_truth = pd.DataFrame(fill_non_bind(ground_truth, non_binding), index=ground_truth.index, columns=ground_truth.columns)

    ground_truth = ground_truth.dropna(thresh=1)
    # Let's take out rows that have just no  energies at all...
    # They are only in the dataset since literature reports values for them...
    # (strange that we don't have binding energies then huh?)
    # These are the four offenders: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    ground_truth = ground_truth[ground_truth.max(axis=1) != ground_truth.min(axis=1)]
    binary_data = binary_data.reindex(ground_truth.index)
    return ground_truth, binary_data


def make_skinny(all_data, col_1="variable", col_2="SMILES"):
    """
    Take a 2D array and make it 1D by stacking each column.
    """
    reset_all_data = all_data.reset_index()
    melted_matrix = pd.melt(
        reset_all_data, id_vars=col_2, value_vars=list(reset_all_data.columns[1:])
    )
    # Sort by col_2 so the iterator can correctly split by rows.
    melted_matrix = melted_matrix.sort_values(by=col_2)
    return melted_matrix.set_index([col_2, col_1])


def unmake_skinny(skinny_pred):
    """
    Take a 1D array and make it 2D by unstacking each column.
    """
    predicted_zeolites_per_osda = {}
    for index in range(len(skinny_pred)):
        osda, zeolite = skinny_pred.iloc[index].name
        pred_value = skinny_pred.iloc[index][0]
        if zeolite not in predicted_zeolites_per_osda:
            predicted_zeolites_per_osda[zeolite] = {}
        predicted_zeolites_per_osda[zeolite][osda] = pred_value
    return pd.DataFrame.from_dict(predicted_zeolites_per_osda)


# TODO(yitong): Big Big Big Ginormous TODO is to do a structural split
# or at least make sure that all train osda don't appear in test and vice versa
# We can use the isomeric split we already created! utililties.get_isomer_chunks()

# TODO(yitong): y_nan_fill is a bad solution. think more about this.
# BIG TODO(yitong): Need to get lowest energy conformer for the osda molecules...
# TODO: we probably want to fill nan values in the X priors too...
# TODO: NORMALIZE THE EMBEDDINGS!!!!!
# TODO: Do k-cross validation same as we did for NTK so we can benchmark.
def package_dataloader(
    device,
    energy_type=Energy_Type.BINDING,
    y_nan_fill=30,
    batch_size=256,
    test_proportion=0.1,
    random_seed=PACKAGE_LOADER_SEED,
    osda_prior_file=OSDA_PRIOR_FILE,
):
    if energy_type == Energy_Type.TEMPLATING:
        ground_truth = pd.read_pickle(TEMPLATING_GROUND_TRUTH)
    elif energy_type == Energy_Type.BINDING:
        ground_truth = pd.read_pickle(BINDING_GROUND_TRUTH)
    else:
        raise ValueError(
            "Sorry, but if you want to use a different ground truth for the energy then create the matrix first."
        )
    skinny_ground_truth = make_skinny(ground_truth, col_1="Zeolite", col_2="SMILES")

    # TODO(Yitong): We're going with an arbitrary high energy which is gross.
    # But We almost certainly certainly don't want to take the row mean anymore...
    # For the sake of comparison maybe we should? How do we handle the non-binding cases now???
    skinny_ground_truth = skinny_ground_truth.fillna(y_nan_fill)
    X = skinny_ground_truth.index.to_numpy()
    y = skinny_ground_truth["value"].to_numpy()

    # We call prior in this method! Just an FYI.
    # Necessary because torch tensors cannot be strings
    X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior = make_prior(
        test=None,
        train=None,
        method="CustomOSDAandZeoliteAsRows",
        normalization_factor=0,
        all_data=skinny_ground_truth,
        stack_combined_priors=False,
        osda_prior_file=osda_prior_file,
    )
    # This is taking advantage of the fact that embedding_shapes
    # contains information for the shape of zeolite & osda priors
    X_osda_prior = np.hstack([X_osda_getaway_prior, X_osda_handcrafted_prior])
    (
        X_osda_train,
        X_osda_test,
        X_zeolite_train,
        X_zeolite_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_osda_prior,
        X_zeolite_prior,
        y,
        test_size=test_proportion,
        shuffle=True,
        random_state=random_seed,
    )
    X_osda_train = torch.tensor(X_osda_train, device=device).float()
    X_osda_test = torch.tensor(X_osda_test, device=device).float()
    X_zeolite_train = torch.tensor(X_zeolite_train, device=device).float()
    X_zeolite_test = torch.tensor(X_zeolite_test, device=device).float()
    y_train = torch.tensor(y_train, device=device).float()
    y_test = torch.tensor(y_test, device=device).float()

    train_dataset = TensorDataset(X_osda_train, X_zeolite_train, y_train)
    test_dataset = TensorDataset(X_osda_test, X_zeolite_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # shuffle=True
    )
    # Make fully sure train & test are disjoint...
    # TODO(yitong): We might want to be checking that X_osda_train & X_osda_test
    # contain distinct OSDAs.... I'm actually pretty sure they're not right now
    # That's some structure bleed right there...
    assert set(train_dataset).isdisjoint(set(test_dataset))
    return (train_dataset, test_dataset, train_loader, test_loader)


if __name__ == "__main__":
    format_ground_truth_pkl()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    package_dataloader(device)
    # format_ground_truth_pkl()
