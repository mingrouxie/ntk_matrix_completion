"""" Temporary file for coding changes while create_truth.py was in use. Copy paste into create_truth.py to see what changes were made"""


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
import datetime

sys.path.append("/home/mrx/general/")
from zeolite.queries.scripts.exclusions import *
from zeolite.queries.scripts.database import get_affinities, get_failed_dockings
from utils.non_binding import fill_non_bind, NonBinding

SCIENCE = "/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/daniels_data/science_paper/binding.csv"

def main(kwargs):
    if not os.path.isdir(kwargs["op"]):
        os.mkdir(kwargs["op"])

    now = "%d%d%d_%d%d%d" % (
        datetime.datetime.now().year,
        datetime.datetime.now().month,
        datetime.datetime.now().day,
        datetime.datetime.now().hour,
        datetime.datetime.now().minute,
        datetime.datetime.now().second,
    )

    if kwargs["science"]:
        science = pd.read_csv(SCIENCE)
        kwargs["substrate_ls"] = list(set(science['Zeolite']))
        # kwargs["ligand_ls"] = list(set(science["SMILES"]))
        kwargs["ik_ls"] = list(set(science["InchiKey"]))

    print("[main] kwargs are\n", kwargs)

    affs, affs_failed = get_affinities(
        substrate=kwargs["substrate"],
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
        substrate_ls=kwargs["substrate_ls"],
        ligand_ls=None,
        num_ligands=1,
        loading=None,
        to_count=True,
        count_only=False,
        ik_ls=kwargs["ik_ls"]
    )
    affs.to_csv(os.path.join(kwargs["op"], now + "_affs.csv"))
    affs_failed.to_csv(os.path.join(kwargs["op"], now + "_affs_failed.csv"))
    # breakpoint() # check affs and affs_failed

    if not kwargs["assume_nb"]:
        failed = get_failed_dockings(
            substrate=kwargs["substrate"],
            ligand=None,
            exclusion=kwargs["failed_exc"],
            included_sphere_gte=4.0,
            to_write=False,
            complex_sets=kwargs["cs"],  # ['iza'],
            mol_sets=kwargs["ms"],  
            # ['210720_omar_quaternary', '210720_omar_diquaternary'],
            output_file=None,
            qs_to_df=True,
            to_count=True,
            count_only=False,
            substrate_ls=kwargs["substrate_ls"],
            ik_ls=kwargs["ik_ls"]
        )
        failed.to_csv(os.path.join(kwargs["op"], now + "_failed.csv"))

    ## TODO: add in failed dreiding
    # breakpoint() # check failed. is it a problem with database.py

    # combine
    if not kwargs["assume_nb"]:
        truth = pd.concat([affs, affs_failed, failed])
    else:
        truth = affs.reset_index().set_index(['substrate', 'ligand_inchikey'])
        from itertools import product
        science_idx = pd.MultiIndex.from_tuples(list(product(set(affs.substrate), set(affs.ligand_inchikey))))
        ligand_info = dict(zip(affs.ligand_inchikey, affs.ligand))
        truth = truth.reindex(science_idx).reset_index().rename(columns={'level_0':'substrate', 'level_1':'ligand_inchikey'}).set_index('crystal_id')
        truth['ligand'] = truth.ligand_inchikey.apply(lambda ik: ligand_info[ik])

    # safety, should be covered in get_affinities
    truth.loc[(truth["bindingatoms"] > 0), "bindingatoms"] = 10
    truth.loc[(truth["bindingatoms"] < -35),"bindingatoms"] = 10

    # treatment for failed
    truth.loc[(truth["bindingatoms"].isna()),"bindingatoms"] = 10  # again, arbitrary

    # treatment for all non-binding TODO: need to make it SOMETHING else at the end
    truth.loc[truth["bindingatoms"]==10][["bindingatoms", "bindingosda", "bindingosdaatoms"]] = 10

    print("[main] truth size before deduplication", truth.shape)

    # breakpoint() # is the deduplication logic wrong?
    # remove duplicates keeping the most negative binding energy
    truth = truth.sort_values(["substrate", "ligand_inchikey", "bindingatoms"])
    truth = truth.drop_duplicates(
        subset=["substrate", "ligand_inchikey", "ligand"], keep="first"
    )
    print("[main] deduplicated truth size", truth.shape)

    # Rename columns because bad code
    truth = truth.rename(
        columns={"substrate": "Zeolite","ligand": "SMILES","ligand_inchikey": "InchiKey","ligand_formula": "Ligand formula","loading": "Loading","bindingatoms": "Binding (SiO2)","bindingosda": "Binding (OSDA)","directivity": "Directivity (SiO2)","competition": "Competition (SiO2)","solvation": "Competition (OSDA)","logp": "Templating",})
    truth.to_csv(os.path.join(kwargs["op"], now + "_truth_before_nb.csv"))

    # Make mask where 1=binding, 0=non-binding
    mask = deepcopy(truth)
    mask["exists"] = 1
    mask.loc[mask["Binding (SiO2)"].gt(0), "exists"] = 0
    mask = mask.reset_index()[["SMILES", "Zeolite", "exists", "InchiKey"]]
    # breakpoint() # is it a problem w the non binding

    # non-binding treatment
    if kwargs["nb"]:
        # TODO: drops empty ones so cannot use in hypothetical space
        # fill all NaN entries, then select the non-binding ones
        breakpoint() # TODO: debugging
        op_cols = ["Binding (SiO2)", "Binding (OSDA)", "Directivity (SiO2)", "Competition (SiO2)", "Templating", "Loading"]
        energies = []
        for op in op_cols:
            truth.loc[truth[op].gt(0), op] = nan    
            energies.append(fill_nb_parallel(
                truth, op, kwargs["index"], kwargs["columns"], kwargs
                # truth, "Binding (SiO2)", kwargs["index"], kwargs["columns"], kwargs
            ))
        energies = pd.concat(energies, axis=1)
        truth = truth.reset_index().rename(columns={'index': 'crystal_id'})
        truth = truth.set_index([kwargs["columns"], kwargs["index"]]).drop(columns=op_cols)
        truth = pd.concat([truth, energies], axis=1)
        mask = mask.set_index([kwargs["columns"], kwargs["index"]]).reindex(truth.index).reset_index()
        truth = truth.reset_index().set_index('crystal_id')

    mask.to_csv(os.path.join(kwargs["op"], now + "_mask.csv"))
    truth.to_csv(os.path.join(kwargs["op"], now + "_truth.csv"))
    print("[main] Output dir:", kwargs["op"] + "/" + now)
    return


def fill_nb_single(df, values, index, columns, kwargs):
    mat = df.pivot(values=values, index=index, columns=columns)
    mat = pd.DataFrame(
        fill_non_bind(mat.values, nb_type=kwargs["nb"]), index=mat.index, columns=mat.columns
    )
    if kwargs["nan_after_nb"] == "drop":
        print(
            "[fill_nb_single] Dropping rows with NaN post-NB treatment",
            mat.shape,
            "-->",
            mat.dropna().shape,
        )
        mat = mat.dropna()
    if kwargs["nan_after_nb"] == "keep":
        print(
            "[fill_nb_single] Filling rows with NaN post-NB treatment with hardcoded one (arbitrary)"
        )
        mat = mat.fillna(1.0)

    df_filled = (
        mat.stack()
        .reset_index()
        .rename(columns={0: "Binding (SiO2)"})
        .set_index([columns, index])
    )
    df_filled = df_filled.loc[
        df_filled.index.isin(df.set_index([columns, index]).index)
    ]
    print("[fill_nb_single] Returning df of shape", df_filled.shape)
    return df_filled


def fill_nb_parallel(df, values, index, columns, kwargs, chunk_size=1000):
    """
    Fills non-binding entries in a specified column in a DataFrame with specified treatment.

    Inputs:
    - df: full DataFrame containing columns of interest: values, index, columns
    - values: column name to use as values in df.pivot
    - index: column name to use as index in df.pivot, is also used for chunking
    - columns: column name to use as columns in df.pivot
    - kwargs: dictionary containing "nb" and "nan_after_nb" keys
    - chunk_size: number of distinct rows in the resultant binding matrix
    
    Returns:
    - DataFrame with the same ordering as df, containing a column with binding and non-binding entries 
    """
    rows = sorted(list(set(df[index])))
    rows_chunked = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    dfs_filled = [
        fill_nb_single(df[df[index].isin(chunk)], values, index, columns, kwargs)
        for chunk in rows_chunked
    ]
    dfs_filled = pd.concat(dfs_filled)
    return dfs_filled


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
    else:
        kwargs["affs_exc"] = None
        kwargs["failed_exc"] = None
    print("[preprocess] excs:", kwargs["affs_exc"], kwargs["failed_exc"])
    
    return kwargs


if __name__ == "__main__":
    print("\n==============================================\n")
    print("[create_truth] start")
    start = time.time()
    parser = argparse.ArgumentParser(description="Truth file creation")
    parser.add_argument("--op", help="Output directory", type=str, required=True)
    # parser.add_argument(
    #     "--batch_size", help="Size of each prior file", type=int, required=True
    # )
    parser.add_argument(
        "--substrate",
        type=str,
        nargs="+",
        help="Parentjob.config.name of complex.substrate",
        default=None,
    )
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
        default=None,
    )
    parser.add_argument(
        "--index",
        type=str,
        help="InchiKey or Zeolite in the index for binding matrix?",
        default="InchiKey",
    )
    parser.add_argument(
        "--columns",
        type=str,
        help="InchiKey or Zeolite in the columns for binding matrix?",
        default="Zeolite",
    )
    parser.add_argument(
        "--science",
        help="If specified, only retrieves for ligands and substrates found in Science paper CSV file ",
        action='store_true'
    )
    parser.add_argument(
        "--assume_nb",
        help="If true, assumes for given set of complexes that they are non-binding if no affinities within range are found",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--substrate_ls",
        type=str,
        nargs="+",
        help="list of substrate names to sieve by",
        default=None,   
    )
    parser.add_argument(
        "--ik_ls",
        type=str,
        nargs="+",
        help="list of InchiKeys to sieve by",
        default=None,   
    )
    args = parser.parse_args()
    kwargs = preprocess(args)
    print("TODO: bindingatoms are still hardcoded")
    main(kwargs)
    print(f"[create_truth] Finished after {(time.time()-start/60)} min \n")
    print("================================================")


# NOTES
# non-binding values - have been making those files manually
# oh wait we need a paradigm shift because we don't have row means anymore
# I wanted to make the non-binding energies
