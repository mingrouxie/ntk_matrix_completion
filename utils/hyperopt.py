import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

from sklearn.model_selection import check_cv, cross_val_score
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from copy import deepcopy
from hyperopt.pyll import scope
from xgboost import cv
from sklearn.model_selection import train_test_split
from models.neural_tangent_kernel.ntk import SplitType

from ntk_matrix_completion.utils.random_seeds import (
    HYPPARAM_SEED,
    MODEL_SEED,
    HYPEROPT_SEED,
)
from ntk_matrix_completion.configs.xgb_hp import get_hyperopt_xgb_space
from ntk_matrix_completion.models.neural_tangent_kernel.ntk import (
    create_iterator,
)
from utils.loss import masked_rmse
from utils.path_constants import XGBOOST_OUTPUT_DIR

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

DEBUG = True


def get_data():
    import os

    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Any results you write to the current directory are saved as output.
    data = "/kaggle/input/wholesale-customers-data-set/Wholesale customers data.csv"
    df = pd.read_csv(data)
    X = df.drop("Channel", axis=1)
    y = df["Channel"]  # convert labels into binary values
    y[y == 2] = 0
    y[y == 1] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    return X_train, X_test, y_train, y_test


def hp_obj_old(space):
    # get data
    X_train, X_test, y_train, y_test = get_data()

    # set up model with parameters from space
    clf = xgb.XGBClassifier(
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
    )

    # fit model to training data
    evaluation = [(X_train, y_train), (X_test, y_test)]
    clf.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        eval_metric="auc",
        early_stopping_rounds=10,
        verbose=False,
    )

    # predict on test data
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred > 0.5)
    print("SCORE:", accuracy)

    # return test accuracy
    return {"loss": -accuracy, "status": STATUS_OK}


class HyperoptSearchCV:
    # TODO: check space same as randomizedsearchcv
    def __init__(self, X, y, mask, params=None, cv=5, fixed_params=None, output=XGBOOST_OUTPUT_DIR) -> None:
        """
        Inputs:

        X: model input
        y: model labels
        mask: same shape as y, 1 for binding and 0 for non-binding. To not use this, simply set all entries=1
        params: search space for hyperparameters
        cv: generator or integer (default: 5)
        fixed_params: hyperparameters that are fixed. Overrides params.
        output: Output directory
        """
        if not params:
            self.space = get_hyperopt_xgb_space()
        else:
            self.space = params
        self.X = X
        self.y = y
        self.cv = cv  # where do I even use this??????
        self.fixed_params = fixed_params
        self.mask = mask
        self.output = output

    def model_eval(self, params):
        # https://www.kaggle.com/code/ilialar/hyperparameters-tunning-with-hyperopt/notebook
        """
        This method carries out cross-validation, with the performance metric computed
        only for entries whose corresponding value in self.mask is 1.
        This method is used as the objective function in hyperopt.fmin.

        Inputs:
        params: set of hyperparameters
        metrics: (str) metric for evaluating performance

        Returns:
        Dictionary containing `loss`, `status` and `loss_variance` as keys
        """
        params.update(**self.fixed_params)  # overwrites
        model = xgb.XGBRegressor(**params)
        # print(
        #     f"[HyperoptSearchCV] check, is params changing?: {params['learning_rate']}"
        # )  # params)

        iterator = create_iterator(
            split_type=SplitType.OSDA_ISOMER_SPLITS,
            all_data=self.X.reset_index().set_index("SMILES"),
            metrics_mask=self.mask,
            k_folds=5,
            seed=HYPEROPT_SEED,
        )

        results = self.cross_val(model, cv_generator=iterator)
        return results

    def cross_val(self, model, cv_generator, metrics="masked_rmse"):
        train_scores = []
        test_scores = []
        for train, test, test_mask_chunk in cv_generator:
            X_train = self.X.loc[train.index]
            X_test = self.X.loc[test.index]
            y_train = self.y.loc[train.index]["Binding (SiO2)"].values
            y_test = self.y.loc[test.index]["Binding (SiO2)"].values
            mask_train = self.mask.loc[train.index]
            mask_train = mask_train.set_index([mask_train.index, "Zeolite"]).values
            mask_test = self.mask.loc[test.index]
            mask_test = mask_test.set_index([mask_test.index, "Zeolite"]).values

            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            if metrics == "masked_rmse":
                train_scores.append(
                    masked_rmse(y_train, pred_train, np.squeeze(mask_train))
                )
                test_scores.append(
                    masked_rmse(y_test, pred_test, np.squeeze(mask_test))
                )
            else:
                raise NotImplementedError(
                    f"[cross_val] metrics {metrics} not implemented"
                )

        # print(
        #     f"[cross_val] Train, test scores: {np.mean(train_scores)}, {np.mean(test_scores)}"
        # )
        return {
            "loss": np.mean(test_scores),
            "status": STATUS_OK,
            "loss_variance": np.std(test_scores),
            "train_loss": np.mean(train_scores),
            "train_loss_variance": np.std(train_scores)
        }

    def fit(self, debug=False):
        if debug:
            max_evals = 50
        else:
            max_evals = 500
        self.trials = Trials()
        print(f"[HyperoptSearchCV] fmin seed is {HyperoptSearchCV}")
        self.best_params_ = fmin(
            fn=self.model_eval,  # objective function
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,  # Number of hyperparameter settings to try (the number of models to fit). Online values range from 50 to 1000
            trials=self.trials,
            verbose=True,
            rstate=np.random.default_rng(HYPEROPT_SEED),
        )
        print(f"[HyperoptSearchCV] Best parameters are {self.best_params_}")

        # Save Trials object which contains information about each step in fmin
        pickle.dump(
            self.trials, open(os.path.join(self.output, "trials.pkl"), "wb")
        )

    @property
    def cv_results_(self):  # dict of numpy (masked) ndarrays in sklearn CV methods
        return self.trials.losses()

    def hyperoptoutput2param(self, best):
        """Change hyperopt output to dictionary with values. Taken from 10.C51 PS2"""
        for key in best.keys():
            if key in self.space.keys():
                best[key] = self.space[key][best[key]]
        return best


if __name__ == "__main__":
    search = HyperoptSearchCV()
    search.fit(debug=DEBUG)
    breakpoint()
