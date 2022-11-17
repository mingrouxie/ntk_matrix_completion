import pandas as pd
import numpy as np
import os
import sys
import django
import argparse
from math import nan
from itertools import product
from copy import deepcopy

HTVSPATH = "/home/mrx/htvs"
DJANGOCHEMPATH = "/home/mrx/htvs/djangochem"
DBNAME = "djangochem.settings.orgel"
sys.path.append(HTVSPATH)
sys.path.append(DJANGOCHEMPATH)
os.environ["DJANGO_SETTINGS_MODULE"] = DBNAME
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rest.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()
from dbsettings import *

sys.path.append("/home/mrx/general/")
from zeolite.queries.scripts.exclusions import *
from zeolite.queries.scripts.database import get_affinities, get_failed_dockings
from utils.non_binding import fill_non_bind, NonBinding


def main(kwargs):
    if not os.path.isdir(kwargs["op"]):
        os.mkdir(kwargs["op"])
    affs = get_affinities(
        substrate=None,
        ligand=None,
        exclusion=kwargs["affs_exc"],
        included_sphere_gte=4.0,
        to_write=False,
        complex_sets=kwargs["cs"],  # ['iza'],
        mol_sets=kwargs[
            "ms"
        ],  # ['210720_omar_quaternary', '210720_omar_diquaternary'],
        output_file=None,
        get_best_per_complex=False,
        performance_metric="bindingatoms",
        substrate_ls=None,
        ligand_ls=None,
        num_ligands=1,
        loading=None,
        to_count=True,
        count_only=False,
    )
    affs.to_csv(os.path.join(kwargs["op"], "affs.csv"))

    failed = get_failed_dockings(
        substrate=None,
        ligand=None,
        exclusion=kwargs["failed_exc"],
        included_sphere_gte=4.0,
        to_write=False,
        complex_sets=kwargs["cs"],  # ['iza'],
        mol_sets=kwargs[
            "ms"
        ],  # ['210720_omar_quaternary', '210720_omar_diquaternary'],
        output_file=None,
        qs_to_df=True,
        to_count=True,
        count_only=False,
    )
    failed.to_csv(os.path.join(kwargs["op"], "failed.csv"))

    ## TODO: add in failed dreiding 

    # combine
    truth = pd.concat(
        [
            affs[["ligand", "substrate", "bindingatoms", "ligand_inchikey"]],
            failed[["ligand", "substrate", "ligand_inchikey"]],
        ]
    )
    truth.columns = ["SMILES", "Zeolite", "Binding (SiO2)", "InchiKey"]

    # Change docked pairs with unfeasible energies to failed dockings. 
    # Using 0 because the duplicate removal in the next section does not work with NaNs
    truth.loc[truth["Binding (SiO2)"] > 0, "Binding (SiO2)"] = 0.0
    truth.loc[truth["Binding (SiO2)"] < -35, "Binding (SiO2)"] = 0.0
    truth.loc[truth["Binding (SiO2)"].isna(), "Binding (SiO2)"] = 0.0
    print("[main] truth size", truth.shape)

    # remove duplicates keeping the most negative binding energy
    truth = truth.drop_duplicates(
        subset=["Zeolite", "InchiKey", "SMILES"], keep="first"
    )
    print("[main] deduplicated truth size", truth.shape)

    # Make mask where 1=binding, 0=non-binding
    mask = deepcopy(truth)
    mask["exists"] = 1
    mask.loc[mask["Binding (SiO2)"] == 0.0, "exists"] = 0
    mask.reset_index()[["SMILES", "Zeolite", "exists", "InchiKey"]].to_csv(
        os.path.join(kwargs["op"], "mask.csv")
    )
    breakpoint()
    # non-binding treatment
    if kwargs["nb"]:
        # fill all NaN entries, then select the relevant ones
        truth.loc[truth["Binding (SiO2)"] == 0.0, "Binding (SiO2)"] = nan
        truth_mat = truth.pivot(values="Binding (SiO2)", index="SMILES", columns="Zeolite")
        truth_mat = pd.DataFrame(fill_non_bind(truth_mat, nb_type=kwargs["nb"]), index=truth_mat.index, columns=truth_mat.columns,)

        if kwargs["nan_after_nb"] == "drop":
            print("[main] Dropping rows with NaN post-NB treatment", truth_mat.shape, "-->", truth_mat.dropna().shape)
            truth_mat = truth_mat.dropna()
        if kwargs["nan_after_nb"] == "keep":
            print("[main] Filling rows with NaN post-NB treatment with hardcoded one (arbitrary)")
            truth_mat = truth_mat.fillna(1.0)

        truth_filled = truth_mat.stack().reset_index().rename(columns={0: "Binding (SiO2)"}).set_index(["Zeolite", "SMILES"])
        truth = truth_filled.loc[truth_filled.index.isin(truth.set_index(["Zeolite", "SMILES"]).index)]

    # if os.path.isfile(os.path.join(kwargs["op"], "energies.csv")):
    #     print("file already exists please double check")
    #     breakpoint()
    truth.to_csv(os.path.join(kwargs["op"], "energies.csv"))
    print("[main] Output dir:", kwargs['op'])

def preprocess(input):
    kwargs = {}
    input = input.__dict__
    # do stuff if needed #
    kwargs.update(input)
    if kwargs["nb"]:
        kwargs["nb"] = NonBinding(kwargs["nb"])
    if kwargs["exc"]:
        kwargs["affs_exc"] = AFFINITY_EXCLUSIONS[kwargs["exc"] - 1]
        kwargs["failed_exc"] = COMPLEX_EXCLUSIONS[kwargs["exc"] - 1]
    return kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truth file creation")
    parser.add_argument("--op", help="Output directory", type=str, required=True)
    # parser.add_argument(
    #     "--batch_size", help="Size of each prior file", type=int, required=True
    # )
    parser.add_argument(
        "--ms",
        type=str,
        nargs="+",
        help="MolSets the desired OSDAs belong to. Gets the union of all of them",
        default=None,
    )
    parser.add_argument(
        "--cs",
        type=str,
        nargs="+",
        help="ComplexSets desired. Gets the union of all of them",
        default=None,
    )
    parser.add_argument(
        "--nb",
        type=int,
        help="If specified, assigns pseudo value to non-binding pairs",
        default=None,
    )
    parser.add_argument(
        "--exc",
        type=int,
        help="Exclusions. Uses custom code, see exclusions in Mingrou's general folder. 1) literature 2) quaternary 3) diquaternary. Applies to both affinities and failed dockings",
        default=None,
    )
    parser.add_argument(
        "--nan_after_nb",
        type=str,
        help="What to do with NaN entries of the truth matrix after non-binding treatment has been applied and NaN entries still exist",
        default=None
    )
    args = parser.parse_args()
    kwargs = preprocess(args)
    print("TODO: bindingatoms are still hardcoded")
    main(kwargs)


# NOTES
# non-binding values - have been making those files manually
# oh wait we need a paradigm shift because we don't have row means anymore
# I wanted to make the non-binding energies
