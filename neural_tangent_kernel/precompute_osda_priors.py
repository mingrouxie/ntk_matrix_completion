import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
from rdkit import Chem
from rdkit.Chem import rdFreeSASA, Descriptors, Descriptors3D, AllChem, Draw
from functools import lru_cache
from tqdm import tqdm
import os
import pathlib
import time

from path_constants import (
    TEMPLATING_GROUND_TRUTH,
    HYPOTHETICAL_OSDA_ENERGIES,
    HYPOTHETICAL_OSDA_BOXES,
    OSDA_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
)
from utilities import save_matrix


@lru_cache(maxsize=16384)
def get_conformers(smile, debug=False, number_of_conformers=2000):
    start = time.time()
    conformer_features = dict()
    # TODO: look into other properties too perhaps...
    conformer_features["volumes"] = []
    conformer_features["energies"] = []
    conformer_features["whim"] = []
    m = Chem.MolFromSmiles(smile)
    m2 = Chem.AddHs(m)
    try:
        conformers = AllChem.EmbedMultipleConfs(
            m2, numConfs=number_of_conformers, pruneRmsThresh=0.5, numThreads=100
        )
        optimised_and_energies = AllChem.MMFFOptimizeMoleculeConfs(
            m2, maxIters=600, numThreads=100, nonBondedThresh=100.0
        )
    except:
        return conformer_features
    ids = []
    all_rms = []
    for c in conformers:
        if optimised_and_energies[c][0] != 0:
            continue
        dont_add = False
        for c2 in conformers[(c + 1) :]:
            rms = AllChem.GetConformerRMS(m2, c, c2)
            if rms < 1.0:
                dont_add = True
        if not dont_add:
            ids.append(c)
    # Root mean square deviation to dedup conformers.
    conformer_features["rms"] = all_rms
    for i in ids:
        conformer_features["volumes"].append(
            AllChem.ComputeMolVolume(m2, conformers[i])
        )
        conformer_features["energies"].append(optimised_and_energies[i][1])
        conformer_features["whim"].append(
            Chem.rdMolDescriptors.CalcWHIM(m2, conformers[i])
        )
    if debug:
        print(
            len(conformer_features["volumes"]),
            len(conformer_features["energies"]),
            len(conformer_features["whim"]),
        )
        print(round(time.time() - start, 3))
    return conformer_features


# Big TODO: Get the lowest energy conformers for each molecule
# This will take a while. Should maybe run on server.
def smile_to_property(
    smile,
    process_conformers=False,
    debug=False,
    save_file=None,
    struggle_against_bad_conformer_errors=False,
):
    """
    process_conformers: Generate priors for 2,000 conformers (too slow)
    save_file: If supplied then we save the molecule's render
    struggle_against_bad_conformer_errors: If the smile is a bad conformer id then generate
    100 other conformers on the hope that one of them will actually embed (be warned: slow).
    """
    properties = {}
    m = Chem.MolFromSmiles(smile)
    num_bonds = len(Chem.RemoveAllHs(m).GetBonds())
    if save_file is not None:
        m2 = Chem.RemoveAllHs(m)
        Draw.MolToFile(m2, save_file + ".png")
    m2 = Chem.AddHs(m)
    rc = AllChem.EmbedMolecule(m2)
    if rc < 0:
        rc = Chem.AllChem.EmbedMolecule(
            m2,
            useRandomCoords=True,
            enforceChirality=False,
            ignoreSmoothingFailures=False,
        )
    try:
        AllChem.MMFFOptimizeMolecule(m2)
    except ValueError as e:
        if str(e) == "Bad Conformer Id" and struggle_against_bad_conformer_errors:
            print(
                "we ran into an issue with a bad conformer for ",
                smile,
                " spawning 100 conformers that hopefully work, this may take some time...",
            )
            conformers = AllChem.EmbedMultipleConfs(
                m2, numConfs=100, pruneRmsThresh=0.5, numThreads=10
            )
            if len(conformers) > 0:
                m2 = conformers[0]
            else:
                return properties
        else:
            return properties
    # WHIMs and GETAWAY are the big ones I think...
    properties["whims"] = Chem.rdMolDescriptors.CalcWHIM(m2)
    properties["getaway"] = Chem.rdMolDescriptors.CalcGETAWAY(m2)
    # TODO: do we need to do PCA here?
    if process_conformers:
        properties["conformer"] = get_conformers(smile)

    properties["mol_weights"] = Descriptors.MolWt(m2)
    properties["volume"] = AllChem.ComputeMolVolume(m2)
    # Note https://issueexplorer.com/issue/rdkit/rdkit/4524
    properties["normalized_num_rotatable_bonds"] = (
        1.0 * Chem.rdMolDescriptors.CalcNumRotatableBonds(m2) / num_bonds
    )
    properties["formal_charge"] = Chem.GetFormalCharge(m2)
    if debug:
        print(
            "volume ",
            properties["volume"],
            "mol_weights ",
            properties["mol_weights"],
            "normalized_num_rotatable_bonds",
            properties["normalized_num_rotatable_bonds"],
            "formal_charge",
            properties["formal_charge"],
        )

    # 0.5 * ((pm3-pm2)**2 + (pm3-pm1)**2 + (pm2-pm1)**2)/(pm1**2+pm2**2+pm3**2)
    properties["asphericity"] = Descriptors3D.Asphericity(m2)
    # sqrt(pm3**2 -pm1**2) / pm3**2
    properties["eccentricity"] = Descriptors3D.Eccentricity(m2)
    # pm2 / (pm1*pm3)
    properties["inertial_shape_factor"] = Descriptors3D.InertialShapeFactor(m2)
    # 3 * pm1 / (pm1+pm2+pm3) where the moments are calculated without weights
    properties["spherocity"] = Descriptors3D.SpherocityIndex(m2)
    # for planar molecules: sqrt( sqrt(pm3*pm2)/MW )
    # for nonplanar molecules: sqrt( 2*pi*pow(pm3*pm2*pm1,1/3)/MW )
    properties["gyration_radius"] = Descriptors3D.RadiusOfGyration(m2)
    if debug:
        print(
            "asphericity ",
            properties["asphericity"],
            "eccentricity ",
            properties["eccentricity"],
            ", inertial_shape_factor ",
            properties["inertial_shape_factor"],
            "spherocity ",
            properties["spherocity"],
            "gyration_radius ",
            properties["gyration_radius"],
        )

    # first to third principal moments of inertia
    properties["pmi1"] = Descriptors3D.PMI1(m2)
    properties["pmi2"] = Descriptors3D.PMI2(m2)
    properties["pmi3"] = Descriptors3D.PMI3(m2)
    if debug:
        print(
            "pmi1 ",
            properties["pmi1"],
            "pmi2 ",
            properties["pmi2"],
            ", pmi3 ",
            properties["pmi3"],
        )

    # normalized principal moments ratio
    # https://doi.org/10.1021/ci025599w
    properties["npr1"] = Descriptors3D.NPR1(m2)
    properties["npr2"] = Descriptors3D.NPR2(m2)
    if debug:
        print("npr1 ", properties["npr1"], ", npr2 ", properties["npr2"])

    radii = rdFreeSASA.classifyAtoms(m2)
    properties["free_sas"] = rdFreeSASA.CalcSASA(m2, radii)
    if debug:
        print("free_sas ", properties["free_sas"])

    # bertz_ct quantifies the complexity of a molecule? vague...
    properties["bertz_ct"] = Chem.GraphDescriptors.BertzCT(m2)
    # do we want other weird things like HallKierAlpha?
    # I want some measure of flexibility. It seems like they calculated that by taking all the conformers.
    # https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.6b00565?rand=xovj8tmp
    return properties


@lru_cache(maxsize=16384)
def average_properties(smile, num_runs=1):
    df = pd.DataFrame(
        [smile_to_property(smile, process_conformers=False) for i in range(num_runs)]
    )
    meaned_df = df.mean(numeric_only=True)
    # Pretty certain it's okay to take the average over WHIMs and GETAWAY...
    # Now let's take care of the columns that are not numeric.
    for col in df.columns:
        scrubbed_col = df[col].dropna()
        if df.dtypes[col] in (np.dtype("float64"), np.dtype("int64")):
            continue
        if all([isinstance(i, list) for i in scrubbed_col]):
            meaned_df[col] = np.mean(
                np.array([np.array(v) for v in scrubbed_col.values]), axis=0
            )
        elif all([isinstance(i, dict) for i in scrubbed_col]):
            # For now since only conformers are lists and because all conformer requests are
            # cached and identical, let's just pick one at random.
            meaned_df[col] = scrubbed_col.sample()
    return dict(meaned_df)


def osda_prior_helper(all_data_df, num_runs, save_file=None):
    prior = pd.DataFrame()
    bad_apples = np.array([])
    for smile in tqdm(all_data_df.SMILES):
        # Some prior values might be 0.0 if the smiles string couldn't be embedded.
        properties = average_properties(smile, num_runs)
        series = pd.Series(properties)
        series.name = smile
        prior = prior.append(series)
        if save_file:
            prior.dropna()
            save_matrix(prior, save_file)

        if len(properties) == 0:
            bad_apples = np.append(bad_apples, smile)
    print(
        " we got this many bad apples ",
        len(bad_apples),
        " check them out: ",
        bad_apples,
    )
    # Drop bad apples aka molecules which couldn't be embedded.
    return prior.dropna()


def prior_from_ground_truth_matrix():
    num_runs = 10
    all_data_df = pd.read_pickle(TEMPLATING_GROUND_TRUTH)
    save_file = OSDA_PRIOR_FILE
    all_data_df.reset_index(inplace=True)
    prior = osda_prior_helper(all_data_df, num_runs, save_file)
    save_matrix(prior, save_file)


# Depends upon daniel's 78K hypothetical OSDA list (ask him for it)
def precompute_priors_for_780K_Osdas():
    num_runs = 4
    # sample_size is archaic & was just used to test scaling.
    # this is probably a code smell and should be removed if not helpful.
    sample_size = 78616

    input = HYPOTHETICAL_OSDA_ENERGIES
    all_data_df = pd.read_csv(input)
    all_data_df = all_data_df.set_index("inchi")

    inchi_to_smile_conversion = HYPOTHETICAL_OSDA_BOXES
    inchi_to_smile = pd.read_csv(inchi_to_smile_conversion)
    inchi_to_smile = inchi_to_smile.set_index("inchi")[["smiles"]]

    joined_df = all_data_df.join(inchi_to_smile, on="inchi")
    joined_df.rename(columns={"smiles": "SMILES"}, inplace=True)
    all_data_df = joined_df.sample(sample_size)

    save_file = OSDA_HYPOTHETICAL_PRIOR_FILE
    prior = osda_prior_helper(all_data_df, num_runs, save_file)
    # drop all molecules that couldn't be embedded
    prior = prior.loc[(prior != 0).any(axis=1)]
    save_matrix(prior, "prior_" + save_file)

    # We need to reindex all_data_df for later...
    all_data_df = all_data_df.set_index("SMILES")
    all_data_df = all_data_df.reindex(prior.index)
    all_data_df.index = all_data_df.index.rename("SMILES")
    save_matrix(all_data_df, save_file)


# zeo1_osda_smile = "[CH3][P+](C1CCCCC1)(C2CCCCC2)(C3CCCCC3)"
def precompute_oneoff_prior(
    osda_smile="[CH3][P+](C1CCCCC1)(C2CCCCC2)(C3CCCCC3)",
    save_name="tricyclohexylmethylphosphonium_prior.pkl",
    num_runs=10,
):
    properties = average_properties(osda_smile, num_runs)
    series = pd.Series(properties)
    series.name = osda_smile
    prior = pd.DataFrame()
    prior = prior.append(series)
    # save_matrix(prior, save_name)
    return prior.dropna()


# TODO: This method is pretty jank, we should destroy later
# Only call this method if you know what you're sure
def append_oneoff_prior(
    precomputed_file_name="data/precomputed_OSDA_prior_10_with_whims.pkl",
    addendum_file_name="missing_OSDA_2.pkl",
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    other_osda_df = pd.read_pickle(addendum_file_name)
    concatted_prior = pd.concat([precomputed_prior, other_osda_df])
    concatted_prior = concatted_prior[~concatted_prior.index.duplicated(keep="first")]
    concatted_prior.to_pickle(precomputed_file_name)


if __name__ == "__main__":
    # precompute_oneoff_prior("CC[N+]12C[N@]3C[N@@](C1)C[N@](C2)C3", "missing_OSDA_1.pkl", 150)
    # precompute_oneoff_prior(
    #     "C[C@H]1CC[N+](C)(C)[C@@H]2C[C@@H]1C2(C)C", "missing_OSDA_2.pkl"
    # )
    prior_from_ground_truth_matrix()
    # precompute_oneoff_prior()
    # precompute_priors_for_780K_Osdas()
