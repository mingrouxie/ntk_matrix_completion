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
import os
import pathlib
import time


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


# TODO: Conformers, WHIM PCA, GetMorganFingerprint, bertz_ct
# Also TODO: figure out how to get deterministic results or at least less variance...
# Also also TODO: cache this.
def smile_to_property(smile, process_conformers=False, debug=False):
    properties = {}
    m = Chem.MolFromSmiles(smile)
    num_bonds = len(Chem.RemoveAllHs(m).GetBonds())
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
    except:
        # Some molecules just can't be embedded it seems
        return properties
    # WHIMs and GETAWAY are the big ones I think...

    # TODO: these will need to be normalized...
    properties["whims"] = Chem.rdMolDescriptors.CalcWHIM(m2)
    properties["getaway"] = Chem.rdMolDescriptors.CalcGETAWAY(m2)
    # TODO: do we need to do PCA here?
    # TODO: we're going to need to do some weird processing here...
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

    # these may be useless.
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

    # this quantifies the complexity of a molecule?
    properties["bertz_ct"] = Chem.GraphDescriptors.BertzCT(m2)
    # do we want other weird things like HallKierAlpha?
    # I want some measure of flexibility. It seems like they calculated that by taking all the conformers.
    # https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.6b00565?rand=xovj8tmp
    # pdb.set_trace()
    if debug:
        m2 = Chem.RemoveHs(m)
        Draw.MolToFile(m2, "test3.o.png")
    return properties


# TODO: Come back for the WHIMs list of floats can't be taken an average of apparently...
@lru_cache(maxsize=16384)
def average_properties(smile, num_runs=1):
    df = pd.DataFrame([smile_to_property(smile, process_conformers=False) for i in range(num_runs)])
    meaned_df = df.mean(numeric_only=True)
    # Pretty certain it's okay to take the average over WHIMs and GETAWAY...
    # But maybe good to double check...
    # Now let's take care of the columns that are not numeric.
    for col in df.columns:
        if df.dtypes[col] in (np.dtype('float64'), np.dtype('int64')):
            continue
        if isinstance(df[col][0], list):
            meaned_df[col] = np.mean(np.array([np.array(v) for v in df[col].values]), axis=0)
        elif isinstance(df[col][0], dict):
            # For now since only conformers are lists and because all conformer requests are
            # cached and identical, let's just pick one at random.
            meaned_df[col] = df[col].sample()
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
    # Drop molecules which couldn't be embedded.
    return prior.dropna()


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    matrix.to_pickle(savepath)


def prior_from_small_matrix():
    num_runs = 10
    input = "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteNonbindingTensor.pkl"
    all_data_df = pd.read_pickle(input)
    save_file = "precomputed_OSDA_prior_conjugates_part2.pkl"
    all_data_df.reset_index(inplace=True)
    all_data_df = all_data_df.iloc[339:,:]
    prior = osda_prior_helper(all_data_df, num_runs, save_file)
    save_matrix(prior, save_file)


def precompute_priors_for_780K_Osdas():
    num_runs = 4
    sample_size = 78616

    input = "/Users/mr/Documents/Work/MIT/PhD/matrix_completion/ntk_matrix_completion/cmap_imputation/data/daniels_data/211221_energies.csv"
    all_data_df = pd.read_csv(input)
    all_data_df = all_data_df.set_index("inchi")

    inchi_to_smile_conversion = "/Users/mr/Documents/Work/MIT/PhD/matrix_completion/ntk_matrix_completion/cmap_imputation/data/daniels_data/211221_boxes.csv"
    inchi_to_smile = pd.read_csv(inchi_to_smile_conversion)
    inchi_to_smile = inchi_to_smile.set_index("inchi")[["smiles"]]  # .to_dict('index')

    joined_df = all_data_df.join(inchi_to_smile, on="inchi")
    joined_df.rename(columns={"smiles": "SMILES"}, inplace=True)
    all_data_df = joined_df.sample(sample_size)

    save_file = "precomputed_energies_" + str(sample_size) + "by196WithWhims.pkl"
    prior = osda_prior_helper(all_data_df, num_runs, save_file)
    # drop all molecules that couldn't be embedded
    prior = prior.loc[(prior != 0).any(axis=1)]
    save_matrix(prior, "prior_" + save_file)

    # We need to reindex all_data_df for later...
    all_data_df = all_data_df.set_index("SMILES")
    all_data_df = all_data_df.reindex(prior.index)
    all_data_df.index = all_data_df.index.rename("SMILES")
    save_matrix(all_data_df, save_file)


if __name__ == "__main__":
    precompute_priors_for_780K_Osdas()
