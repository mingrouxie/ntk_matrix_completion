source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

hostname

ROOT=/home/mrx/projects/affinity/ntk_matrix_completion
DATAROOT=/home/mrx/projects/affinity_pool/ntk_matrix_completion

# TRUTH_DIR=$DATAROOT/data/truths/testing_3/new # got deleted, but it was re-saved in testing_7
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 5 --cs 2021_apriori_dskoda --nan_after_nb drop

FWS=(SFS EON JSR MAR MEL SAT OSO BEA MTN DDR SVV TUN BEC SBE RTH AFT SAF CDO SSY AEN AWW SAS EWO IWR AFX FER JOZ MTF UOS KFI SGT ERI UEI POR ZON JRY RHO LTA PON IWW EPI IFY PWW GON ATO MSO NAT SFW SWY ETV SAV CTH FAR TOL JSW GME OWE OSI EAB APD APC LTL BOF FRA VNI AFN ATT ATN AFS ITT MEP RTE STO CON LEV AFI MOZ TER MFS IFR PTY AHT LTF NON IRR AEL AFG PUN EUO UOZ BOG EZT NPT PCS VET BOZ ETR RRO EDI MRE UOE STW SOD ATS AET ITE USI BSV SBT HEU VFI SFF MOR DFT MER CGS LOV JST CZP DAC POS SBN IWS GIS BPH IRN AVL SOV SFO STF SIV SFN LAU OFF ACO ASV MEI SFH MAZ UFI EWS SFE SAO STI IHW SEW AWO PCR SBS MTT ESV SOS AFV SSF BRE SOF UOV FAU SFG NPO ITH LIO ITW CHA MSE MTW CAN AST RUT NSI LOS OBW OKO IFO CSV PWO MWW AEI MRT TON AFY JNT MFI CGF UTL AFR AVE ETL IFW ISV ITR PHI JSN EMT GIU SZR THO SOR CFI ITG DFO STT)
TRUTH_DIR=$DATAROOT/data/truths/230212/iza_apriori_zach
echo 'Zero for NB'
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 5 --nan_after_nb drop --substrate iza_parse --cs 221227_generative_model_subset 2021_apriori_dskoda --substrate_ls "${FWS[@]}"

# 221227_generative_model_subset for ComplexSet is the 221227_generative_model_subset MolSet with the 209 frameworks from 2021_apriori_dskoda ComplexSet