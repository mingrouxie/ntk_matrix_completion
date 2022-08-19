import multiprocessing
from math import ceil
from itertools import product
from utils.path_constants import (
    BINDING_GROUND_TRUTH,
    OSDA_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_CONFORMER_PRIOR_FILE,
)
import pandas as pd
import os
# from auto_tqdm import tqdm
import os
from sklearn.preprocessing import normalize, OneHotEncoder
import time

def source_prior_files(): # TODO: hard coded, will not need this after the prior extraction is revamped
    file_names = [
        "/home/mrx/general/zeolite/queries/output_files/20220612/conformers_from_20220525_affs/priors/conformers_"
        + str(i)
        + "_priors.pkl"
        for i in range(501)
    ]
    file_names.extend(
        [
            "/home/mrx/general/zeolite/queries/output_files/20220612/conformers_from_20220611_failed/priors/conformers_"
            + str(i)
            + "_priors.pkl"
            for i in range(501)
        ]
    )
    return file_names

def get_osda_conformer_prior_file(smiles, output_file=OSDA_CONFORMER_PRIOR_FILE):
    if not output_file:
        output_file = OSDA_CONFORMER_PRIOR_FILE
    if os.path.isfile(output_file):
        print("File already exists! Is this the conformer prior file you want?")
        breakpoint()

    start = time.time()
    precomputed_file_names = source_prior_files()
    # parallel
    num_chunks = 5
    chunk_size = ceil(len(precomputed_file_names) / num_chunks)
    splits = [
        precomputed_file_names[idx * chunk_size : (idx + 1) * chunk_size]
        for idx in range(num_chunks)
    ]
    ls_smiles = [smiles]
    with multiprocessing.Pool(processes=5) as pool:
        priors = pool.starmap(
            extract_priors_from_conformer_files, product(splits, ls_smiles)
        )
    priors = pd.concat(priors)
    # # series - took more than 5 mins for 1 round so dropped it (iirc it took 13mins last time)
    # priors = [extract_priors_from_conformer_files(
    #     precomputed_file_names, smiles)] # [0:10], smiles)]
    # priors = pd.concat(priors)
    # print(f"{(time.time() - start)/60} minutes taken to extract from all the files")

    priors = priors.drop_duplicates("ligand")
    # sorry for the messy data files :(
    if "npr3" in priors.columns:
        priors = priors.drop(columns="npr3")
    if "mol" in priors.columns:
        priors = priors.drop(columns="mol")
    if "molecule_xyz" in priors.columns:
        priors = priors.drop(columns="molecule_xyz")
    if "geom_id" in priors.columns:
        priors = priors.drop(columns="geom_id")
    if not len(smiles) == priors.shape[0]:
        print(f"WARNING: {len(smiles)} SMILES but {priors.shape[0]} priors grabbed")

    priors.to_pickle(output_file)
    return output_file


def extract_priors_from_conformer_files(files, smiles):
    priors = []
    for file in files:
        precomputed_df = pd.read_pickle(file)
        priors.append(precomputed_df[precomputed_df.ligand.isin(smiles)])
    priors = pd.concat(priors).drop_duplicates(subset="ligand")
    return priors


if __name__ == '__main__':
    smiles = list(set(pd.read_pickle(BINDING_GROUND_TRUTH).index))
    output_file = get_osda_conformer_prior_file(smiles, output_file=OSDA_CONFORMER_PRIOR_FILE)
    breakpoint()