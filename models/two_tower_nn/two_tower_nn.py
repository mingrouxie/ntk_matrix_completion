import sys
import pathlib
import pdb
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass


from ntk_matrix_completion.utils.analysis_utilities import (
    plot_top_k_curves,
    calculate_metrics,
)
from ntk_matrix_completion.models.neural_tangent_kernel.ntk import (
    run_ntk,
    skinny_ntk_sampled_not_sliced,
)
from ntk_matrix_completion.utils.package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
    make_skinny,
    unmake_skinny,
    package_dataloader,
)
from ntk_matrix_completion.utils.utilities import save_matrix, plot_matrix
from ntk_matrix_completion.utils.path_constants import (
    TEN_FOLD_CROSS_VALIDATION_ENERGIES,
)

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    # NOTE: nothing fancy like layer_norm or dropout..
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i <= final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


@dataclass
class TwoTowerNN:
    zeolite_encoder: MLP
    osda_encoder: MLP


# TODO(Yitong): BIG TODO. Can we choose this loss function to be more sensitive
# to the lower energies? MSE is not biased in that way
# TODO: do we want to try L2 normalization before dot product?
def loss(zeolite_encoding, osda_encoding, true, loss=nn.MSELoss()):
    assert zeolite_encoding.shape == osda_encoding.shape and len(
        zeolite_encoding
    ) == len(true)
    # Taking the matrix multiplication of all osda & zeo encodings is just for hardware
    # speed up. We really only care about the diagonal of the product matrix (i.e., dot products
    # of each zeolite and each OSDA)
    products = torch.matmul(osda_encoding, zeolite_encoding.T)
    pred = torch.diagonal(products)
    return loss(true, pred)


def run(
    train_loader,
    model: TwoTowerNN,
    device,
    epoch,
    optimizer,
    writer,
    loss_func=loss,
    purpose="train",
    debug=True,
):
    if purpose == "train":
        model.zeolite_encoder.train()
        model.osda_encoder.train()

    for batch_idx, (osda_data, zeolite_data, labels) in enumerate(train_loader):
        # This is probably redundant.
        osda_data, zeolite_data, labels = (
            osda_data.to(device),
            zeolite_data.to(device),
            labels.to(device),
        )
        if purpose == "train":
            optimizer.zero_grad()
        zeolite_encoding = model.zeolite_encoder(zeolite_data)
        osda_encoding = model.osda_encoder(osda_data)
        loss = loss_func(zeolite_encoding, osda_encoding, labels)
        if purpose == "train":
            loss.backward()
            optimizer.step()
        if batch_idx % 20 == 0 and debug:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch,
                    batch_idx,
                    loss,
                )
            )
    writer.add_scalar("Loss/{}".format(purpose), loss, epoch)


def main():
    # https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BIG TODO: Implement the isomeric test / train split in the NN package loader too!
    train_dataset, _test_dataset, train_loader, test_loader = package_dataloader(device)
    # TODO: really make the zeolite input embeddings larger...
    # OSDA input embedding dim right now is 289... 273 from GETAWAY and 16 handcrafted descriptors
    osda_dim = [train_dataset[0][0].shape[0], 128, 64, 32]
    # zeolite input embedding dim right now is 15...
    zeolite_dim = [train_dataset[0][1].shape[0], 128, 64, 32]

    model = TwoTowerNN(
        MLP(zeolite_dim).to(device),
        MLP(osda_dim).to(device),
    )
    # Never used tensorboard before.. is this useful?
    osda_data_for_tensorboard, zeolite_data_for_tensorboard, _labels = next(
        iter(train_loader)
    )
    writer.add_graph(model.osda_encoder, osda_data_for_tensorboard)
    writer.add_graph(model.zeolite_encoder, zeolite_data_for_tensorboard)
    optimizer = torch.optim.Adam(
        list(model.zeolite_encoder.parameters())
        + list(model.osda_encoder.parameters()),
        lr=0.005,
    )
    num_epochs = 5
    for epoch in range(num_epochs):
        run(
            train_loader,
            model,
            device,
            epoch,
            optimizer,
            writer,
        )
        run(test_loader, model, device, epoch, optimizer, writer, purpose="test")
    print("hello yitong")
    writer.close()


if __name__ == "__main__":
    main()
