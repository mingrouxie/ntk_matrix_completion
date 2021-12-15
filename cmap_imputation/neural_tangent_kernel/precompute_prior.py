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

from sklearn.preprocessing import normalize, OneHotEncoder


# TODO: Conformers, WHIM PCA, GetMorganFingerprint, bertz_ct
# Also TODO: figure out how to get deterministic results or at least less variance...
# Also also TODO: cache this.
def smile_to_property(smile, debug=False):
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
    properties["whims"] = Chem.rdMolDescriptors.CalcWHIM(m2)
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

    if debug:
        Draw.MolToFile(m2, "test3.o.png")
    return properties


# TODO: Come back for the WHIMs list of floats can't be taken an average of apparently...
@lru_cache(maxsize=16384)
def average_properties(smile, num_runs=1):
    df = pd.DataFrame([smile_to_property(smile) for i in range(num_runs)])
    return dict(df.mean())


def osda_prior_helper(all_data_df, num_runs):
    # still need:
    # rog
    # Charge
    # min_vol_conformer_pca_whim, all_conformer_whim
    # WHIM descriptors https://onlinelibrary.wiley.com/doi/abs/10.1002/qsar.200510159
    prior = pd.DataFrame()
    bad_apples = np.array([])
    for smile in tqdm(all_data_df.reset_index().SMILES):
        # Some prior values might be 0.0 if the smiles string couldn't be embedded.
        properties = average_properties(smile, num_runs)
        prior = prior.append(properties, ignore_index=True)
        if len(properties) == 0:
            bad_apples = np.append(bad_apples, smile)
    print(
        " we got this many bad apples ",
        len(bad_apples),
        " check them out: ",
        bad_apples,
    )
    return prior


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    # if os.path.exists(savepath):
    #     overwrite = input(f"A file already exists at path {savepath}, do you want to overwrite? (Y/N): ")
    matrix.to_pickle(savepath)


if __name__ == "__main__":
    print(
        "Modify make_prior in prior.py to add a custom prior! There are a few choices to start."
    )
    num_runs = 1
    input = '/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteNonbindingTensor.pkl'
    all_data_df = pd.read_pickle(input)
    save_file = "precomputed_OSDA_prior.pkl"
    prior = osda_prior_helper(all_data_df, num_runs)
    pdb.set_trace()
    save_matrix(prior, save_file)
