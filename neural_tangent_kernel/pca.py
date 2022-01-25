import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from package_matrix import get_ground_truth_energy_matrix
from prior import load_vector_priors
import pdb
from path_constants import HYPOTHETICAL_OSDA_ENERGIES, OSDA_HYPOTHETICAL_PRIOR_FILE

ground_truth, binary_data = get_ground_truth_energy_matrix()
daniel_energies = pd.read_csv(HYPOTHETICAL_OSDA_ENERGIES)
precomputed_priors = pd.read_pickle(OSDA_HYPOTHETICAL_PRIOR_FILE)
daniel_energies = daniel_energies.reindex(precomputed_priors.index)
daniel_energies = daniel_energies.drop(
    set.intersection(set(ground_truth.index), set(daniel_energies.index))
)
all_data = pd.concat([ground_truth, daniel_energies])
getaway_df = load_vector_priors(
    target_index=all_data.index,
    vector_feature="getaway",
    identity_weight=0,
    normalize=False,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
)
getaway_df = getaway_df.dropna()
x = np.nan_to_num(getaway_df.values)
# Oooo, it already normalizes down all of the columns... not just altogether...
normalized_altogether = StandardScaler().fit_transform(x)


standardized_getaway_df = pd.DataFrame(
    data=normalized_altogether, index=getaway_df.index
)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(
    data=principalComponents, columns=["principal component 1", "principal component 2"]
)
print(principalDf.head(5))
print("what's the explained variance? ", pca.explained_variance_ratio_)


indicator = pd.DataFrame(
    data=np.vstack(
        (np.ones((ground_truth.shape[0], 1)), np.zeros((daniel_energies.shape[0], 1)))
    )
)
finalDf = pd.concat([principalDf, indicator], axis=1)
finalDf.index = all_data.index

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1", fontsize=15)
ax.set_ylabel("Principal Component 2", fontsize=15)
ax.set_title("PCA of GETAWAY Descriptors", fontsize=20)

targets = [0, 1]
colors = ["r", "b"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(
        finalDf.loc[indicesToKeep, "principal component 1"],
        finalDf.loc[indicesToKeep, "principal component 2"],
        c=color,
        s=70, 
        alpha=0.1
    )
ax.legend(['Hypothetical', 'Ground Truth'])
ax.grid()
plt.show()
plt.savefig('pca.png')


finalDf.loc[finalDf['principal component 2'] > 20]
finalDf.loc[finalDf['principal component 2'] == 20.267503]

# C[N@@+]1(Cc2ccccc2)CCCC[C@@H]1CCC[C@H]1CCCC[N@@...             -15.757894              20.267503  1.0
# C[N@@+]1(Cc2ccccc2)CCCC[C@H]1CCC[C@@H]1CCCC[N@+...             -14.746956              20.625072  1.0
# C=CC[N@+]1(CCC)CCc2c(C)cccc2C1                                 -15.698148              20.281823  0.0
# C=C[C@H](C)[N@+]1(CCC=C(C)C)CCCC[C@H]1C                        -14.697408              20.135789  0.0




from precompute_osda_priors import smile_to_property
smile_to_property(" C[N@@+]1(Cc2ccccc2)CCCC[C@@H]1CCC[C@H]1CCCC[N@@+]1(C)Cc1ccccc1", debug=True)
pdb.set_trace()
print('hi yitong')
