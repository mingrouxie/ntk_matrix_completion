import pandas as pd
import numpy as np

BAD_SUBSTRATES = ['RWY', 'ITN', 'IRY', 'NES', 'SVR', 'UWY', 'CIT-14']

def make_ground_truth(affs_file, failed_file, df_name, mask_name, to_write=True):
    affs_initial = pd.read_pickle(affs_file)
    fail_initial = pd.read_pickle(failed_file)

    affs = affs_initial[["substrate", "ligand", "bindingatoms"]]
    affs = affs.loc[affs.bindingatoms < 0]
    failed_in_affs = affs.loc[affs.bindingatoms >= 0]
    failed_in_affs['bindingatoms'] = np.nan
    fail_df = fail_initial[["ligand", "substrate"]]
    fail_df['bindingatoms'] = np.nan

    df = pd.concat([affs, failed_in_affs, fail_df])
    # TODO: throw this in the database.py? or better to have it here at the endpoint
    df = df.loc[(~df.ligand.str.contains('F'))
                & (~df.ligand.str.contains('Cl'))
                & (~df.substrate.isin(BAD_SUBSTRATES))
                # & (df.ligand_atoms > 10)
                ]

    unique = df.loc[~df.duplicated(subset=['substrate', 'ligand'], keep=False)]
    dupped = df.loc[df.duplicated(subset=['substrate', 'ligand'], keep=False)]
    dupped = dupped.sort_values(
        ['substrate', 'ligand', 'bindingatoms']) 
    dupped = dupped.loc[~dupped.duplicated(
        subset=['substrate', 'ligand'], keep='first')]
    # keep='first' keeps all the indices apart from the first
    # for clarity, we are keeping the entries that do bind
    df = pd.concat([unique, dupped])

    df['exists'] = 1
    breakpoint()
    priors = df.pivot(index='ligand', columns='substrate',
                      values='bindingatoms')
    # entries in priors = NaN if non-binding OR does not exist
    mask = df.pivot(index='ligand', columns='substrate', values="exists")
    # mask is Boolean for data point exists or not

    if to_write:
        priors.to_pickle(df_name)
        mask.to_pickle(mask_name)
    return priors, mask


if __name__ == '__main__':
    make_ground_truth(
        # '/home/mrx/general/zeolite/queries/output_files/20220525/affs/iza_parse_210720_omar_quaternary_affs.pkl',
        # '/home/mrx/general/zeolite/queries/output_files/20220611/failed/iza_parse_210720_omar_quaternary_failed.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_210720_omar_quaternary_truth.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_210720_omar_quaternary_mask.pkl',

        # '/home/mrx/general/zeolite/queries/output_files/20220525/affs/iza_parse_literature_affs.pkl',
        # '/home/mrx/general/zeolite/queries/output_files/20220611/failed/iza_parse_literature_failed.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_literature_truth.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_literature_mask.pkl',

        # '/home/mrx/general/zeolite/queries/output_files/20220525/affs/iza_parse_210720_omar_diquaternary_affs.pkl',
        # '/home/mrx/general/zeolite/queries/output_files/20220611/failed/iza_parse_210720_omar_diquaternary_failed.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_210720_omar_diquaternary_truth.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_210720_omar_diquaternary_mask.pkl',

        # '/home/mrx/general/zeolite/queries/output_files/20220525/affs/ase_db_parse_210720_omar_diquaternary_affs.pkl',
        # '/home/mrx/general/zeolite/queries/output_files/20220611/failed/ase_db_parse_210720_omar_diquaternary_failed.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/ase_db_parse_210720_omar_diquaternary_truth.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/ase_db_parse_210720_omar_diquaternary_mask.pkl',

        # '/home/mrx/general/zeolite/queries/output_files/20220525/affs/ase_db_parse_210720_omar_quaternary_affs.pkl',
        # '/home/mrx/general/zeolite/queries/output_files/20220611/failed/ase_db_parse_210720_omar_quaternary_failed.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/ase_db_parse_210720_omar_quaternary_truth.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/ase_db_parse_210720_omar_quaternary_mask.pkl',

        # '/home/mrx/general/zeolite/queries/output_files/20220525/affs/ase_db_parse_literature_affs.pkl',
        # '/home/mrx/general/zeolite/queries/output_files/20220611/failed/ase_db_parse_literature_failed.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/ase_db_parse_literature_truth.pkl',
        # '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/ase_db_parse_literature_mask.pkl',

        '/home/mrx/general/zeolite/queries/output_files/20220616/affs/iza_parse_literature_affs.pkl',
        '/home/mrx/general/zeolite/queries/output_files/20220616/failed/iza_parse_literature_failed.pkl',
        '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_literature_truth_overlap.pkl',
        '/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_parse_literature_mask_overlap.pkl',


    )