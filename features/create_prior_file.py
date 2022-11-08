from genericpath import isfile
import multiprocessing
from math import ceil, nan
from itertools import product
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
from copy import deepcopy

# from auto_tqdm import tqdm
from sklearn.preprocessing import normalize, OneHotEncoder
import time
import argparse

from ntk_matrix_completion.utils.path_constants import (
    BINDING_GROUND_TRUTH,
    OSDA_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_CONFORMER_PRIOR_FILE,
)

from dbsettings import *


def get_osda_features(kwargs):
    if kwargs["inchikeys"]:
        specs = list(Species.objects.filter(inchikey__in=kwargs["inchikeys"]))
    if kwargs["ms"]:
        specs = list(Species.objects.filter(mol__sets__name__in=kwargs["ms"]))

    print("[get_osda_features] Number of Species:", len(specs))

    def batched_specs(specs, batch_size):
        for idx, s in enumerate(range(0, len(specs), batch_size)):
            yield idx, specs[s : s + batch_size]

    for idx, sp in batched_specs(specs, kwargs["batch_size"]):
        print("[get_osda_features] Batch length of Species:", len(specs))
        batch_kwargs = deepcopy(kwargs)
        batch_kwargs["species"] = sp
        batch_kwargs["osda_file"] = (
            batch_kwargs["osda_file"].split(".")[0]
            + "_"
            + str(idx)
            + "."
            + batch_kwargs["osda_file"].split(".")[-1]
        )
        get_osda_features_single(batch_kwargs)
    return


def get_osda_features_single(kwargs):
    """Returns a DataFrame with SMILES as the index and columns the desired features"""
    geoms = Geom.objects.filter(
        confnum__isnull=False,
        method__name="molecular_mechanics_mmff94",
        species__in=kwargs["species"],
    )
    # if kwargs["inchikeys"]: # TODO: we need to batch this...
    #     geoms = geoms.filter(species__inchikey__in=kwargs["inchikeys"]).distinct()
    # elif kwargs["ms"]:
    #     geoms = geoms.filter(species__mol__sets__name__in=kwargs["ms"]).distinct()

    columns = {
        "geom": "id",
        "smiles": "species__smiles",
        "inchikey": "species__inchikey",
        # 'fps': 'continuousfps',
        "fps_name": "continuousfps__method__name",
        "fps_val": "continuousfps__fingerprint",
    }
    data = pd.DataFrame(
        geoms.values_list(*list(columns.values())), columns=list(columns.keys())
    )
    print("[get_osda_features] data size", data.size)
    # TODO: It does not really matter because I'm doing this for all the descriptors eventually,
    # but maybe we should sieve for the features at this step zzz 
    # ThreeDContinuousFingerprint.objects.filter(fps__name__in=kwargs["features"], geom__in=geoms)

    # separate scalar and vector features
    data_sc = data.loc[~data.fps_name.isin(kwargs["vector_fps"])]
    data_ve = data.loc[data.fps_name.isin(kwargs["vector_fps"])]

    # process scalar
    data_sc = data_sc.applymap(lambda x: x[0] if type(x) == list else x)
    data_sc = pd.pivot_table(
        data_sc,
        values="fps_val",
        columns="fps_name",
        index="smiles",
        aggfunc=np.mean,
        fill_value=nan,
    ).sort_index()
    data_sc = data_sc.reindex(sorted(data_sc.columns), axis=1)

    # process vector
    def vec_mean(input):
        return np.mean(np.array(input.tolist()), axis=0)

    data_ve = pd.pivot_table(
        data_ve,
        values="fps_val",
        columns="fps_name",
        index="smiles",
        aggfunc=vec_mean,
        fill_value=nan,
    ).sort_index()
    data_ve = data_ve.reindex(sorted(data_ve.columns), axis=1)

    data = pd.concat([data_sc, data_ve], axis=1)
    data = data[[x for x in data.columns.tolist() if x in kwargs["features"]]]

    print("[single] Data size", data.shape)
    data.to_hdf(kwargs["osda_file"], key="osda_priors")

    return data


def get_fw_features(kwargs):
    """Returns a DataFrame with framework names as the index and columns as the desired features"""
    return NotImplementedError
    # fws = Framework.objects.filter(name__in=kwargs["fws"])

    # columns = {
    #     "fw": "name",
    #     "details": "prototype__details",
    #     "substrate_atoms": "prototype__xyz__len",
    # }

    # data = pd.DataFrame(fws.values_list(columns.values), columns=columns.keys)
    # data = pd.concat(
    #     [data.drop(["details"], axis=1), data.details.apply(pd.Series)], axis=1
    # )
    # data = data[[x for x in data.columns.tolist() if x in kwargs["features"]]]
    #
    # if kwargs["zeolite"]:
    #     fws_data.to_hdf(kwargs["fws_file"], key="zeolite_priors")
    # return data


def main(kwargs):
    """
    Main function to make a prior file depending on user specified inputs
    """
    # set up file names and dir
    if not os.path.isdir(kwargs["op"]):
        os.mkdir(kwargs["op"])

    kwargs["osda_file"] = os.path.join(kwargs["op"], "osda_priors.h5")
    kwargs["fws_file"] = os.path.join(kwargs["op"], "zeolite_priors.h5")
    if os.path.isfile(kwargs["osda_file"]) | os.path.isfile(kwargs["fws_file"]):
        print("[main] Output files already exists, adding time to file name")
        now = "_%d%d%d_%d%d%d" % (
            datetime.datetime.now().year,
            datetime.datetime.now().month,
            datetime.datetime.now().day,
            datetime.datetime.now().hour,
            datetime.datetime.now().minute,
            datetime.datetime.now().second,
        )
        kwargs["osda_file"] = (
            kwargs["osda_file"].split(".")[0]
            + now
            + "."
            + kwargs["osda_file"].split(".")[-1]
        )
        kwargs["fws_file"] = (
            kwargs["fws_file"].split(".")[0]
            + now
            + "."
            + kwargs["fws_file"].split(".")[-1]
        )

    if kwargs["osda"]:
        get_osda_features(kwargs)
    if kwargs["zeolite"]:
        get_fw_features(kwargs)

    print(
        "[main] Created prior files",
        "\n",
        kwargs["osda_file"].split(".")[0],
        "\n",
        kwargs["fws_file"].split(".")[0],
    )
    # should be able to read in chunks if needed: https://stackoverflow.com/questions/40348945/reading-data-by-chunking-with-hdf5-and-pandas


def preprocess(input):
    kwargs = {"vector_fps": ["getaway", "whim", "box", "axes"]}
    input = input.__dict__
    # do stuff if needed #
    kwargs.update(input)
    return kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prior file creation")
    parser.add_argument("--op", help="Output directory", type=str, required=True)
    # parser.add_argument(
    #     "--op_file", help="Output file name", type=str, default="priors.h5"
    # )  # TODO: h5? npy? what?
    parser.add_argument(
        "--batch_size", help="Size of each prior file", type=int, required=True
    )
    parser.add_argument(
        "--osda",
        help="Make OSDA prior file if specified",
        action="store_true",
        dest="osda",
    )
    parser.add_argument(
        "--zeolite",
        help="Make zeolite prior file if specified",
        action="store_true",
        dest="zeolite",
    )
    parser.add_argument(
        "--features", help="Desired features", type=str, nargs="+", required=True
    )
    parser.add_argument(
        "--inchikeys",
        type=str,
        nargs="+",
        help="InchiKeys of desired OSDAs",
        default=None,
    )
    parser.add_argument(
        "--fws", type=str, nargs="+", help="Names of desired frameworks", default=None
    )
    parser.add_argument(
        "--ms",
        type=str,
        nargs="+",
        help="MolSets the desired OSDAs belong to. Filters by all of them",
        default=None,
    )
    # parser.add_argument(
    #     "--cs",
    #     type=str,
    #     nargs="+",
    #     help="ComplexSets the desired complexes belong to. Filters by all of them",
    #     default=None,
    # )
    # parser.add_argument(
    #     "--fwfam",
    #     type=str,
    #     nargs="+",
    #     help="FrameworkFamily-es the desired complexes belong to. Filters by all of them",
    #     default=None,
    # )
    # parser.add_argument(
    #     "--ms_exclude",
    #     type=str,
    #     nargs="+",
    #     help="Undesired MolSets. Excludes by all of them",
    #     default=None,
    # )
    # parser.add_argument(
    #     "--cs_exclude",
    #     type=str,
    #     nargs="+",
    #     help="Undesired ComplexSets. Excludes by all of them",
    #     default=None,
    # )
    # parser.add_argument(
    #     "--fwfam_exclude",
    #     type=str,
    #     nargs="+",
    #     help="Undesired FrameworkFamily-es. Excludes by all of them",
    #     default=None,
    # )
    # parser.add_argument(
    #     "--avg",
    #     help="If specified, returns priors averaged over all conformers",
    #     action="store_true",
    #     dest="avg",
    # )
    args = parser.parse_args()
    kwargs = preprocess(args)
    # breakpoint - please get all the frameworks into their respective families thank you vm
    main(kwargs)
