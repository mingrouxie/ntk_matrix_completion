import sys
import pathlib
import os
import pdb
import pandas as pd
from enum import Enum


sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import (
    save_matrix,
)

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)


class Energy_Type(Enum):
    TEMPLATING = 1
    BINDING = 2


TEMPLATING_SAVE_FILENAME = "data/TemplatingGroundTruth.pkl"
BINDING_SAVE_FILENAME = "data/BindingSiO2GroundTruth.pkl"


def format_ground_truth_pkl():
    """
    Format binding.csv from Schwalbe Coda's work, requires binding.csv downloaded into the /data folder
    https://github.com/learningmatter-mit/Zeolite-Phase-Competition/blob/main/data/binding.csv
    """
    ground_truth_df = pd.read_csv("data/binding.csv", index_col=0)
    binding_matrix = ground_truth_df.pivot(
        index="SMILES", columns="Zeolite", values="Binding (SiO2)"
    )
    print(
        "The binding matrix has these many values: ",
        binding_matrix.notna().sum().sum(),
        " out of these many total cells",
        binding_matrix.isna().sum().sum() + binding_matrix.notna().sum().sum(),
    )
    save_matrix(binding_matrix, BINDING_SAVE_FILENAME)

    templating_matrix = ground_truth_df.pivot(
        index="SMILES", columns="Zeolite", values="Templating"
    )
    print(
        "The templating matrix has these many values: ",
        templating_matrix.notna().sum().sum(),
        " out of these many total cells",
        templating_matrix.isna().sum().sum() + templating_matrix.notna().sum().sum(),
    )
    save_matrix(templating_matrix, TEMPLATING_SAVE_FILENAME)


# minimum_row_length default set to 2 so we can perform r^2 and Spearman correlation.
def get_ground_truth_energy_matrix(
    energy_type=Energy_Type.TEMPLATING,
    desired_shape=None,
    minimum_row_length=2,
):
    if energy_type == Energy_Type.TEMPLATING:
        ground_truth = pd.read_pickle("data/TemplatingGroundTruth.pkl")
    elif energy_type == Energy_Type.BINDING:
        ground_truth = pd.read_pickle("data/BindingSiO2GroundTruth.pkl")
    else:
        # REFERENCE format_ground_truth_pkl()
        raise ValueError(
            "Sorry, but if you want to use a different ground truth for the energy then create the matrix first."
        )

    # Filter down to desired_shape & filter by min_row_length
    if desired_shape is not None:
        assert (
            desired_shape <= ground_truth.shape
        ), "Can't be asking for a shape bigger than the full matrix"
    else:
        desired_shape = ground_truth.shape

    shaped_ground_truth = pd.DataFrame()
    for _index, row in ground_truth.iterrows():
        if shaped_ground_truth.shape >= desired_shape:
            break
        shaped_row = row[: desired_shape[1]]
        if len(shaped_row.dropna()) < minimum_row_length:
            continue
        shaped_ground_truth = shaped_ground_truth.append(shaped_row)
    shaped_ground_truth.index.name = "SMILES"
    ground_truth = shaped_ground_truth
    binary_data = ground_truth.fillna(0)
    binary_data[binary_data != 0] = 1

    # Set all empty spots in the matrix to be the row mean
    ground_truth = ground_truth.apply(lambda row: row.fillna(row.mean()), axis=1)
    ground_truth = ground_truth.dropna(thresh=1)
    # Let's take out rows that have just no  energies at all...
    # not even sure how they got into the dataset... Worth investigating...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    ground_truth = ground_truth[ground_truth.max(axis=1) != ground_truth.min(axis=1)]
    binary_data = binary_data.reindex(ground_truth.index)
    return ground_truth, binary_data


def make_skinny(all_data, col_1="variable", col_2="SMILES"):
    reset_all_data = all_data.reset_index()
    melted_matrix = pd.melt(
        reset_all_data, id_vars=col_2, value_vars=list(reset_all_data.columns[1:])
    )
    # Sort by col_2 so the iterator can correctly split by rows.
    melted_matrix = melted_matrix.sort_values(by=col_2)
    return melted_matrix.set_index([col_2, col_1])


def unmake_skinny(skinny_pred):
    predicted_zeolites_per_osda = {}
    for index in range(len(skinny_pred)):
        osda, zeolite = skinny_pred.iloc[index].name
        pred_value = skinny_pred.iloc[index][0]
        if zeolite not in predicted_zeolites_per_osda:
            predicted_zeolites_per_osda[zeolite] = {}
        predicted_zeolites_per_osda[zeolite][osda] = pred_value
    return pd.DataFrame.from_dict(predicted_zeolites_per_osda)


if __name__ == "__main__":
    format_ground_truth_pkl()
