import time
import pandas as pd
import numpy as np
import os
import json
import argparse
import logging
from ast import literal_eval
import yaml
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader

# from ntk_matrix_completion.utils.logger import setup_logger
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader

from ntk_matrix_completion.utils.early_stopping import EarlyStopper
from ntk_matrix_completion.models.multitask.multitask_nn import MULTITASK_MODELS
from ntk_matrix_completion.utils.loss import multitask_loss
from ntk_matrix_completion.features.prior import make_prior
from ntk_matrix_completion.utils.package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
)
from ntk_matrix_completion.utils.random_seeds import (
    HYPPARAM_SEED,
    ISOMER_SEED,
    MODEL_SEED,
)
from ntk_matrix_completion.utils.utilities import (
    cluster_isomers,
    get_isomer_chunks,
    scale_data,
    report_best_scores,
    SplitType,
    MultiTaskTensorDataset
)

from sklearn.model_selection import train_test_split

# TODO: truth in main() but y in train() etc.? 

def train_notes(kwargs):
    '''for 1 epoch'''
    # TODO: working notes
    # see sam's code for structure reference
    # see xgb on how to retrieve the data zzz

    # hyperparam
    # run sigopt
    if kwargs["tune"]:
        # tune(#truth, #prior, kwargs)
        pass

    # feature selection based on train? 

    # build model

    # train model

    # evaluation

    # remember to unscale targets

    # compute metrics

    # export results

    # save model

    # class MultipleOptimizer(object):
    #     def __init__(*op):
    #         self.optimizers = op

    #     def zero_grad(self):
    #         for op in self.optimizers:
    #             op.zero_grad()

    #     def step(self):
    #         for op in self.optimizers:
    #             op.step()


    # opt = MultipleOptimizer(optimizer1(params1, lr=lr1), 
    #                         optimizer2(params2, lr=lr2))

    # loss.backward()
    # opt.zero_grad()
    # opt.step()

    return


def train(model, dataloader, optimizers, device):
    
    '''
    A function to train on the entire dataset for one epoch.
    
    Args: 
        model (torch.nn.Module): multitask model 
        dataloader (torch.utils.data.Dataloader): DataLoader object for the train data
        optimizers (list of torch.optim.Optimizer): list of Optimizer objects to interface gradient calculation and optimization. Hardcoded as [classification_optimizer, regression_optimizer] 
        device (str): Your device
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''
    batch_loss = []
    model.train()
    for batch in dataloader:    
        X, y, mask, _ = batch
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)
        
        # TODO: where to separate NB and B? Do we need to separate it?? 
        y_preds = model(X) 
        load_loss, energy_loss = multitask_loss(y, y_preds, mask)
        batch_loss.append((load_loss.item(), energy_loss.item()))
        # print("[train] indiv loss in a batch", (load_loss.item(), energy_loss.item()))

        # TODO: checked ordering for multitasknnsep and multitasknncorr! 
        optimizers[0].zero_grad()
        load_loss.backward()
        optimizers[1].zero_grad()
        energy_loss.backward()

        optimizers[0].step()
        optimizers[1].step()

    return np.array(batch_loss).mean()


def validate(model, dataloader, device):
    
    '''
    A function to validate on the validation dataset for one epoch.
    
    Args: 
        model (torch.nn.Module): multitask model 
        dataloader (torch.utils.data.Dataloader): DataLoader object for the validation data
        device (str): Your device
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''
    val_loss = []
    model.eval() 

    with torch.no_grad():    
        for batch in dataloader:
            X, y, mask, _ = batch
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            y_pred = model(X)
            load_loss, energy_loss = multitask_loss(y, y_pred, mask)
            val_loss.append((load_loss.item(), energy_loss.item())) 

    return np.array(val_loss).mean()


def evaluate(model, dataloader, device):

    '''
    A function to return the classification probabilities and true ys (for evaluation). 
    
    Args: 
        model (torch.nn.Module): multitask model 
        dataloader (torch.utils.data.Dataloader): DataLoader object for the train data
        device (str): Your device
        
    Returns: 
        (np.array, np.array, np.array): true ys, predicted loads, predicted energies
    '''
    y_preds_all = [[],[]]
    ys = []
    masks = []
    indices = [[], []] # molecule-zeolite pair

    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            X, y, mask, idx = batch
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            y_preds = model(X) 
            # TODO: hardcoded
            y_preds_all[0].extend(y_preds[0])
            y_preds_all[1].extend(y_preds[1])
            ys.extend(y)
            masks.extend(mask)
            indices[0].extend(idx[0]) 
            indices[1].extend(idx[1]) 

    return ys, y_preds_all, masks, indices


def main(kwargs):
    start_time = time.time() # seconds
#### TODO: dedup because code is almost the same as in xgb.py START

    # get ys
    if kwargs["energy_type"] == Energy_Type.BINDING:
        # truth = pd.read_csv(BINDING_CSV) # TODO: debug: binding values only
        truth = pd.read_csv(kwargs["truth"])  # B with NB values
    else:
        print("[MT] Please work only with binding energies")
        breakpoint()

    if kwargs["sieved_file"]:
        sieved_priors_index = pd.read_pickle(kwargs["sieved_file"]).index
        sieved_priors_index.name = "SMILES"
        truth = truth[truth["SMILES"].isin(sieved_priors_index)]

    truth = truth.set_index(["SMILES", "Zeolite"])

    # ys are binding energy and normalized loading TODO: energy is hardcoded hm

    print(f"[MT] Filling {truth[kwargs['load_label']].isna().values.sum()} nan points with 0")    
    truth[kwargs["load_label"]] = truth[kwargs["load_label"]].fillna(0)

    truth = truth[["Binding (SiO2)", *kwargs["load_label"]]]

    mask = pd.read_csv(kwargs["mask"])

    # get features
    print("[MT] prior_method used is", kwargs["prior_method"])
    prior = make_prior(
        test=None,
        train=None,
        method=kwargs["prior_method"],
        normalization_factor=0,
        all_data=truth,
        stack_combined_priors=False,
        osda_prior_file=kwargs["osda_prior_file"],
        zeolite_prior_file=kwargs["zeolite_prior_file"],
        osda_prior_map=kwargs["osda_prior_map"],
        zeolite_prior_map=kwargs["zeolite_prior_map"],
        other_prior_to_concat=kwargs["other_prior_to_concat"],
    )

    # TODO: THIS IF THREAD IS RATHER UNKEMPT. WHEN WE GENERALIZE TO ZEOLITES....
    if kwargs["prior_method"] == "CustomOSDAVector":
        X = prior
        print(f"[MT] Prior of shape {prior.shape}")
    elif kwargs["prior_method"] == "CustomOSDAandZeoliteAsRows":
        X_osda_handcrafted_prior = prior[0]
        X_osda_getaway_prior = prior[1]
        X_zeolite_prior = prior[2]

        print(
            f"[MT] Check prior shapes:",
            X_osda_handcrafted_prior.shape,
            X_osda_getaway_prior.shape,
            X_zeolite_prior.shape,
        )

        ### what to do with the retrieved X
        print("[MT] Prior treatment is", kwargs["prior_treatment"])
        if kwargs["prior_treatment"] == 1:
            X = X_osda_handcrafted_prior
        elif kwargs["prior_treatment"] == 2:
            X = np.concatenate([X_osda_handcrafted_prior, X_osda_getaway_prior], axis=1)
        elif kwargs["prior_treatment"] == 3:
            X = np.concatenate([X_osda_handcrafted_prior, X_zeolite_prior], axis=1)
        elif kwargs["prior_treatment"] == 4:
            X = np.concatenate([X_osda_getaway_prior, X_zeolite_prior], axis=1)
        elif kwargs["prior_treatment"] == 5:
            X = np.concatenate(
                [X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior],
                axis=1,
            )
        elif kwargs["prior_treatment"] == 6:
            X = X_zeolite_prior
        else:
            # if kwargs["stack_combined_X"] == "all":
            #     X = np.concatenate(
            #         [X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior],
            #         axis=1,
            #     )
            # elif kwargs["stack_combined_X"] == "osda":
            #     X = np.concatenate([X_osda_handcrafted_prior, X_osda_getaway_prior], axis=1)
            # elif kwargs["stack_combined_X"] == "zeolite":
            #     X = X_zeolite_prior
            # else:
            print(f"[MT] What do you want to do with the X??")
            breakpoint()
    else:
        print(f"[MT] prior_method {kwargs['prior_method']} not implemented")
        breakpoint()

    X = pd.DataFrame(X, index=truth.index)
    print("[MT] Final prior X shape:", X.shape)

    # split data
    truth = truth.reset_index("Zeolite")

    if kwargs["split_type"] == SplitType.OSDA_ISOMER_SPLITS:
        # get train_test_split by isomers
        clustered_isomers = pd.Series(cluster_isomers(truth.index).values())
        clustered_isomers = clustered_isomers.sample(frac=1, random_state=ISOMER_SEED)
    else:
        print("[MT] What data splits do you want?")
        breakpoint()

    clustered_isomers_train, clustered_isomers_test = train_test_split(
        clustered_isomers, test_size=0.1, shuffle=False, random_state=ISOMER_SEED
    )
    smiles_train = sorted(list(set().union(*clustered_isomers_train)))
    smiles_test = sorted(list(set().union(*clustered_isomers_test)))

    truth_train = truth.loc[smiles_train].reset_index().set_index(["SMILES", "Zeolite"])
    truth_test = truth.loc[smiles_test].reset_index().set_index(["SMILES", "Zeolite"])

    # scale ground truth if specified
    truth_train_scaled = pd.DataFrame(index=truth_train.index)
    truth_test_scaled = pd.DataFrame(index=truth_test.index)
    if kwargs["energy_scaler"]:
        truth_train_scaled['Binding (SiO2)'], truth_test_scaled['Binding (SiO2)'], kwargs["energy_scaler_info"] = scale_data(
            kwargs["energy_scaler"], 
            pd.DataFrame(truth_train['Binding (SiO2)']), 
            pd.DataFrame(truth_test['Binding (SiO2)']), 
            kwargs['output'], # save scaling info
            "truth_energy" # scaling info filename
            )
    else:
        truth_train_scaled['Binding (SiO2)'] = truth_train['Binding (SiO2)']
        truth_test_scaled['Binding (SiO2)'] = truth_test['Binding (SiO2)']

    if kwargs ["load_scaler"]:
        truth_train_scaled[kwargs["load_label"]], truth_test_scaled[kwargs["load_label"]], kwargs["load_scaler_info"] = scale_data(
            kwargs["energy_scaler"], 
            pd.DataFrame(truth_train[kwargs["load_label"]]), 
            pd.DataFrame(truth_test[kwargs["load_label"]]), 
            kwargs['output'], # save scaling info
            "truth_load" # scaling info filename
            )
    else:
        truth_train_scaled[kwargs["load_label"]] = truth_train[kwargs["load_label"]]
        truth_test_scaled[kwargs["load_label"]] = truth_test[kwargs["load_label"]]

    # save scaled truths
    truth_train_scaled.to_pickle(
        os.path.join(kwargs["output"], "truth_train_scaled.pkl")
    )
    truth_test_scaled.to_pickle(os.path.join(kwargs["output"], "truth_test_scaled.pkl"))

    # split inputs
    X_train = X.loc[smiles_train]
    X_test = X.loc[smiles_test]

    # scale inputs
    X_train_scaled, X_test_scaled, input_scaler_info = scale_data(
        kwargs["input_scaler"], X_train, X_test, kwargs["output"], "input"
    )
    # TODO: print("DEBUG: Check X and mask have been created properly")
    # breakpoint()
    X_train_scaled.to_pickle(os.path.join(kwargs["output"], "X_train_scaled.pkl"))
    X_test_scaled.to_pickle(os.path.join(kwargs["output"], "X_test_scaled.pkl"))

    # split mask;
    mask_train = mask.set_index("SMILES").loc[smiles_train][["Zeolite", "exists"]]
    mask_train.to_pickle(os.path.join(kwargs["output"], "mask_train.pkl"))
    mask_test = mask.set_index("SMILES").loc[smiles_test][["Zeolite", "exists"]]
    mask_test.to_pickle(os.path.join(kwargs["output"], "mask_test.pkl"))

#### TODO: dedup because code is almost the same as in xgb.py END
    
    # save indices
    mask_train = mask_train.reset_index().set_index(['SMILES', 'Zeolite'])
    mask_test = mask_test.reset_index().set_index(['SMILES', 'Zeolite'])
    idx_train = mask_train.index.to_list()
    idx_test = mask_test.index.to_list()

    # make it torch friendly
    # idx_train = torch.tensor(idx_train, device=kwargs['device'])
    # idx_test = torch.tensor(idx_test, device=kwargs['device'])
    X_train_scaled = torch.tensor(X_train_scaled.values, device=kwargs['device']).float()
    truth_train_scaled = torch.tensor(truth_train_scaled.values, device=kwargs['device']).float()
    mask_train = torch.tensor(mask_train.values, device=kwargs['device']).float()
    X_test_scaled = torch.tensor(X_test_scaled.values, device=kwargs['device']).float()
    truth_test_scaled = torch.tensor(truth_test_scaled.values, device=kwargs['device']).float()
    mask_test = torch.tensor(mask_test.values, device=kwargs['device']).float()

    # get datasets and dataloaders
    train_dataset = MultiTaskTensorDataset(X_train_scaled, truth_train_scaled, mask_train, idx_train)
    test_dataset = MultiTaskTensorDataset(X_test_scaled, truth_test_scaled, mask_test, idx_test)

    # breakpoint()
    # # TODO: debug check
    # try:
    #     assert set(train_dataset).isdisjoint(set(test_dataset)), "Train and test not disjoint"
    # except TypeError:
    #     pass

    train_dataloader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=False) # TODO: no shuffle to preserve isomer ordering as much as possible during CV?
    test_dataloader = DataLoader(test_dataset, batch_size=kwargs['batch_size'], shuffle=False) # TODO: no shuffle to preserve isomer ordering as much as possible during CV?

    prep_time = time.time() - start_time
    print("Time to prepare labels and input", "{:.2f}".format(prep_time/60), "mins")

    # get model
    model = kwargs['model'](
        l_sizes=kwargs['l_sizes'], 
        class_op_size=len(kwargs['load_label']),
        batch_norm=kwargs['batch_norm'],
        softmax=kwargs['softmax']
        )
    model.to(kwargs['device'])
    print("[MT] model:\n")
    print(model)

    # get optimizers
    cla_params = model.classifier.parameters()
    reg_params = model.regressor.parameters()
    if kwargs["optimizer"]["cla_opt"] == 'adam': #TODO: hardcoded for now
        cla_opt = torch.optim.Adam(params=cla_params, lr=1e-2) # TODO: lr
    if kwargs["optimizer"]["reg_opt"] == 'adam':
        reg_opt = torch.optim.Adam(params=reg_params, lr=1e-2)
    optimizers = [cla_opt, reg_opt]

    # get scheduler
    # lr scheduler to prevent overfitting or overshooting a loss minimum (from PS3)
    # TODO: disable first
    # TODO: what kind of schedulers are there
    # if kwargs['scheduler']:
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5)

    # early stopping
    early_stopper = EarlyStopper(patience=kwargs["patience"], min_delta=kwargs["min_delta"])

    # train model
    epoch_losses = []
    val_losses = []
    print("\n[MT] Training model")
    for epoch in range(kwargs['epochs']):
        epoch_loss = train(model, train_dataloader, optimizers, kwargs['device'])
        val_loss = validate(model, train_dataloader, kwargs['device'])
        epoch_losses.append(epoch_loss)
        val_losses.append(val_loss)
        # if kwargs['scheduler']:
        #     scheduler.step(val_loss)
        print("Epoch", epoch, "{:.4f}".format(epoch_loss), "{:.4f}".format(val_loss))
        if early_stopper.early_stop(val_loss):  
            print("Early stopping, val_loss:", val_loss)
            break

    # save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'cla_opt_state_dict': optimizers[0].state_dict(),
        'reg_opt_state_dict': optimizers[1].state_dict(),
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
    }, os.path.join(kwargs['output'], 'model.pt'))
    train_time = time.time() - start_time - prep_time
    print("Time to train:", "{:.2f}".format(train_time/60), "mins")

    # evaluate on entire train/test set
    def evaluate_whole(dataloader, label="test"):
        ys, y_preds, masks, indices = evaluate(model, dataloader, kwargs['device'])
        ys = torch.stack(ys)
        # TODO: warning here: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
        y_preds = [torch.stack(y_preds[0]), torch.stack(y_preds[1])]
        masks = torch.stack(masks)
        load_loss, energy_loss = multitask_loss(ys, y_preds, masks)
        print(f"main: {label} set load_loss:", "{:.4f}".format(load_loss.item()), "; energy_loss:", "{:.4f}".format(energy_loss.item()))

        # save predictions
        test_mask = pd.DataFrame(masks.cpu().numpy())
        test_mask.columns = ['exists']
        test_mask.to_csv(os.path.join(kwargs["output"], f'pred_{label}_mask.csv'))
        y_preds = pd.DataFrame(torch.cat([y_preds[1].cpu(), y_preds[0].cpu()], dim=1))
        ys = pd.DataFrame(ys.cpu().numpy())


        y_preds.columns = ["Binding (SiO2)", *kwargs["load_label"]]
        ys.columns = ["Binding (SiO2)", *kwargs["load_label"]]

        y_preds.to_csv(os.path.join(kwargs["output"], f'pred_{label}_y_preds.csv'))
        ys.to_csv(os.path.join(kwargs["output"], f'pred_{label}_ys.csv'))

        indices = pd.DataFrame(indices).T
        indices.columns = ['SMILES', 'Zeolite']
        indices.to_csv(os.path.join(kwargs['output'], f'pred_{label}_indices.csv'))

    evaluate_whole(train_dataloader, label='train')
    evaluate_whole(test_dataloader, label='test')

    print("[MT] Output folder is", kwargs["output"])
    print("Total time:", "{:.2f}".format((time.time() - start_time) / 60), "mins")

def get_defaults():
    '''Ensure backward compatibility with some of the old run files when new arguments are added. These get overriden by the config file.'''
    # TODO: please throw into utils or somewhere
    kwargs = {
        'batch_norm': False,
        'softmax': True
    }
    return kwargs


def preprocess(args):
    config_file = args.__dict__['config']
    kwargs = get_defaults()
    with open(config_file, "rb") as file:
        kwargs_config = yaml.load(file, Loader=yaml.Loader)
        kwargs.update(kwargs_config)
        kwargs['config'] = config_file

    if os.path.isdir(kwargs["output"]):
        now = "_%d%d%d_%d%d%d" % (
            datetime.now().year,
            datetime.now().month,
            datetime.now().day,
            datetime.now().hour,
            datetime.now().minute,
            datetime.now().second,
        )
        kwargs["output"] = kwargs["output"] + now
    print("[MT] Output folder is", kwargs["output"])
    os.makedirs(kwargs["output"], exist_ok=True)
    # setup_logger(kwargs["output"], log_name="multitask_train.log", debug=kwargs["debug"])
    # pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # check device
    try:
        kwargs['device'] = torch.cuda.current_device()
    except RuntimeError:
        kwargs['device'] = 'cpu'
    print(("[preprocess] kwargs device:", kwargs['device']), 'of type', type(kwargs['device']))

    # transform some inputs
    kwargs['model'] = MULTITASK_MODELS[kwargs['model']]
    kwargs['l_sizes'] = literal_eval(kwargs['l_sizes'])
    kwargs["energy_type"] = Energy_Type(kwargs["energy_type"])
    kwargs["split_type"] = SplitType(kwargs["split_type"])
    # TODO: 22 and 49 are hardcoded
    if kwargs["load_type"] == "load":
        kwargs["load_label"] = [f"load_{i}" for i in range(0,22)]
    elif kwargs["load_type"] == "load_norm":
        kwargs["load_label"] = [f"load_norm_{i}" for i in range(0,49)]
    elif kwargs["load_type"] == "single":
        kwargs["load_label"] = ["Loading"]

    if kwargs.get("split_type", None) == "naive":
        kwargs["split_type"] = SplitType.NAIVE_SPLITS
    elif kwargs.get("split_type", None) == "zeolite":
        kwargs["split_type"] = SplitType.ZEOLITE_SPLITS
    elif kwargs.get("split_type", None) == "osda":
        kwargs["split_type"] = SplitType.OSDA_ISOMER_SPLITS

    # dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    with open(Path(kwargs["output"]) / "args.yaml", "w") as fp:
        fp.write(yaml_args)
    with open(Path(kwargs["config"].split(".")[0] + "_args.yaml"), "w") as fp:
        fp.write(yaml_args)

    print("Output folder:", kwargs["output"])
    print(f"Args:\n{yaml_args}")
    return kwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multitask")
    parser.add_argument("--config", help="Config file", required=True)
    args = parser.parse_args()
    kwargs = preprocess(args)
    main(kwargs)
