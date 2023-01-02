21 Dec 22
- Renamed `hyp_osda` to `221113_hyp_osda` because it does not contain some of the new or renamed features

16 Dec 22
- 221216 contains priors for science file as well as iza_all

8 Dec 22
- `iza_parse_literature_docked_priors.pkl` is used in the IZC notebook 
- `zeolite_priors_20221118_25419` is currently in use as the newest IZA zeolite prior file

- `hyp_osdas_others_old` (ignore)
copied over to `hyp_osdas`: (153 to 160).pkl
i.e. do not use this folder
I have not yet predicted on these files either

Before 19 Aug 22
Cannot generate for the double hypothetical space because we are talking about a MASSIVE matrix (99.9% empty)
MemoryError: Unable to allocate 861. GiB for an array with shape (442652, 261057) and data type float64
