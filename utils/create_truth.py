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
from zeolite.queries.scripts.database import get_affinities, get_failed_dockings, get_failed_dreiding
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
        to_count=False,
        count_only=False,
        ik_ls=kwargs["ik_ls"],
        pos_is_nb=True,
        return_crystals=False
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
            substrate_ls=kwargs["substrate_ls"],
            output_file=None,
            qs_to_df=True,
            to_count=False,
            count_only=False,
            ik_ls=kwargs["ik_ls"],
        )
        failed.to_csv(os.path.join(kwargs["op"], now + "_failed.csv"))

        dreid_failed = get_failed_dreiding(
            substrate=kwargs["substrate"],
            ligand=None,
            exclusion=kwargs["failed_exc"],
            included_sphere_gte=4.0,
            to_write=False,
            complex_sets=kwargs["cs"],  # ['iza'],
            mol_sets=kwargs["ms"],  
            # ['210720_omar_quaternary', '210720_omar_diquaternary'],
            substrate_ls=kwargs["substrate_ls"],
            ligand_ls=None,
            output_file=None,
            qs_to_df=True,
            to_count=False,
            count_only=False,
            ik_ls=kwargs["ik_ls"],
        )

    # combine
    if not kwargs["assume_nb"]:
        truth = pd.concat([affs, affs_failed, failed, dreid_failed])
    else:
        # assume_nb=True => create all possible substrate-ligand pairs, and assign non-binding to the pairs that do not exist in affs
        truth = affs.reset_index().set_index(['substrate', 'ligand_inchikey'])
        from itertools import product
        science_idx = pd.MultiIndex.from_tuples(list(product(set(affs.substrate), set(affs.ligand_inchikey))))
        ligand_info = dict(zip(affs.ligand_inchikey, affs.ligand))
        truth = truth.reindex(science_idx).reset_index().rename(columns={'level_0':'substrate', 'level_1':'ligand_inchikey'}).set_index('crystal_id')
        truth['ligand'] = truth.ligand_inchikey.apply(lambda ik: ligand_info[ik])

    # safety, should be covered in get_affinities
    truth.loc[(truth["bindingatoms"] > 0), "bindingatoms"] = 10
    truth.loc[(truth["bindingatoms"] < -35),"bindingatoms"] = 10
    # treatment for failed/ missing pairs
    truth.loc[(truth["bindingatoms"].isna()),"bindingatoms"] = 10  # again, arbitrary

    # treatment for all non-binding TODO: need to make it SOMETHING else at the end
    truth.loc[truth["bindingatoms"]==10][["bindingatoms", "bindingosda", "bindingosdaatoms"]] = 10
    truth.loc[truth["bindingatoms"]==10][["loading", "total_loading", "loading_norm"]] = 0

    print("[main] truth size before deduplication", truth.shape)

    # remove duplicates keeping the most negative binding energy
    truth = truth.sort_values(["substrate", "ligand_inchikey", "bindingatoms"])
    truth = truth.drop_duplicates(
        subset=["substrate", "ligand_inchikey", "ligand"], keep="first"
    )
    print("[main] deduplicated truth size", truth.shape)
    truth = truth.drop_duplicates(
        subset=["substrate", "ligand_inchikey"], keep="first"
    )
    print("[main] deduplicated truth size", truth.shape)
    # Rename columns because bad code
    truth = truth.rename(
        columns={"substrate": "Zeolite","ligand": "SMILES","ligand_inchikey": "InchiKey","ligand_formula": "Ligand formula","loading": "Loading","bindingatoms": "Binding (SiO2)","bindingosda": "Binding (OSDA)","directivity": "Directivity (SiO2)","competition": "Competition (SiO2)","solvation": "Competition (OSDA)","logp": "Templating",})
    truth.to_csv(os.path.join(kwargs["op"], now + "_truth_before_nb.csv"))

    # Make mask where exists=1=binding, exists=0=non-binding
    mask = deepcopy(truth)
    mask["exists"] = 1
    
    # Following from above where non-binding pairs' bindingatoms is assigned a value of 10, 
    mask.loc[mask["Binding (SiO2)"].gt(0), "exists"] = 0
    mask = mask.reset_index()[["SMILES", "Zeolite", "exists", "InchiKey"]]

    # non-binding treatment
    if kwargs["nb"]:
        print("[main] Applying NB treatment:", kwargs['nb'])
        print('[main] Note that NB treatment drops rows that have empty entries, so cannot be used in the hypothetical space. Proceed with caution')
        print("[main] Note that the same KIND of binding treatment is applied to both loading and energy for the moment")

        # First, treat relevant columns, depending on value of Binding (SiO2). Note fill_nb_parallel preserves order
        # TODO: why are we computing templating energy before dropping it?? 
        cols_to_change = ['Loading', 'loading_norm', 'total_loading', "Binding (OSDA)", "Directivity (SiO2)", "Competition (SiO2)", "Templating"]
        for col in cols_to_change:
            truth.loc[truth['Binding (SiO2)'].gt(0), col] = nan
            truth.loc[:, col] = fill_nb_parallel(truth, col, kwargs["index"], kwargs["columns"], kwargs)
            # TODO: *** ValueError: cannot handle a non-unique multi-index!

        # Finally, treat Binding (SiO2). Note fill_nb_parallel preserves order
        truth.loc[truth["Binding (SiO2)"].gt(0), "Binding (SiO2)"] = nan  
        truth["Binding (SiO2)"] = fill_nb_parallel(truth, "Binding (SiO2)", kwargs["index"], kwargs["columns"], kwargs)

    mask.to_csv(os.path.join(kwargs["op"], now + "_mask.csv"))
    truth.to_csv(os.path.join(kwargs["op"], now + "_truth.csv"))
    print("[main] Output dir:", kwargs["op"] + "/" + now)
    return


def fill_nb_single(df, values, index, columns, kwargs):
    '''
    Fills non-binding entries in a row with a specified value that can be dependent on the rest of the row.

    Inputs: 
        df: DataFrame containing columns with names specified by values, index and columns
        values: DataFrame pivot values option
        index: DataFrame pivot index option
        columns: DataFrame pivot columns option
        kwargs: Dictionary of aditional arguments `nan_after_nb` and `nb`

    Returns:
        A DataFrame with only the columns with names specified by values, index and columns
    '''
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
            "[fill_nb_single] Filling rows with NaN post-NB treatment with hardcoded 1.0 (arbitrary)"
        )
        mat = mat.fillna(1.0)

    df_filled = (
        mat.stack()
        .reset_index()
        .rename(columns={0: "Binding (SiO2)"})
        .set_index([index, columns])
    )

    # grab only the relevant rows
    df_filled = df_filled.loc[
        df_filled.index.isin(df.set_index([index, columns]).index)
    ].reset_index()
    print("[fill_nb_single] Returning df of shape", df_filled.shape)
    return df_filled


def fill_nb_parallel(df, values, index, columns, kwargs, chunk_size=1000):
    """
    Fills non-binding entries in a specified column in a DataFrame with specified treatment.

    Inputs:
    - df: full (not necessarily) DataFrame containing columns of interest: values, index, columns
    - values: column name to use as values in df.pivot
    - index: column name to use as index in df.pivot, is also used for chunking
    - columns: column name to use as columns in df.pivot
    - kwargs: dictionary containing "nb" and "nan_after_nb" keys
    - chunk_size: number of distinct rows in the resultant binding matrix
    
    Returns:
    - numpy array with the same [index, columns] ordering as df, containing one column with binding and non-binding entries 
    """

    # order is not preserved to prevent non-unique indexing problems. Technically there shouldn't be any, or the return original order below will break as well
    # TODO: obscure code IDK how to fix
    rows = sorted(list(set(df[index]))) 
    rows_chunked = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    dfs_filled = [
        fill_nb_single(df[df[index].isin(chunk)], values, index, columns, kwargs)
        for chunk in rows_chunked
    ]
    dfs_filled = pd.concat(dfs_filled)
    # return original order 
    try:
        dfs_filled = dfs_filled.set_index([index, columns]).reindex(df.set_index([index, columns]).index)
    except ValueError: # debugging purposes
        breakpoint()
    
    return dfs_filled.values


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
        help="If specified, assigns pseudo value specified by this argument to non-binding pairs (see non_binding.py). Right now, will assign the same kind (mean, zero, etc.) to both loading and all energies, so beware",
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
        help="InchiKey or Zeolite in the index for binding matrix? For filling in NA values per row",
        default="InchiKey",
    )
    parser.add_argument(
        "--columns",
        type=str,
        help="InchiKey or Zeolite in the columns for binding matrix? For filling in NA values per row",
        default="Zeolite",
    )
    parser.add_argument(
        "--science",
        help="If specified, only retrieves for ligands and substrates found in Science paper CSV file ",
        action='store_true'
    )
    parser.add_argument(
        "--assume_nb",
        help="If true, creates all possible combinations for extracted substrates and ligands and assume non-existent pairs are non-binding",
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
