import numpy as np

class EarlyStopper:
    '''From https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    patience: threshold for counter
    min_delta: positive delta
    counter: keep track of epochs with loss within a min_delta of the previous losses
    min_validation_loss: updated with the lowest loss
    '''
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta): # if val_loss is within this delta then counter does not increase, allows for fluctuations if specified
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# early_stopper = EarlyStopper(patience=3, min_delta=10)
# for epoch in np.arange(n_epochs):
#     train_loss = train_one_epoch(model, train_loader)
#     validation_loss = validate_one_epoch(model, validation_loader)
#     if early_stopper.early_stop(validation_loss):             
#         break