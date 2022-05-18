from typing import List, Tuple, Any, Dict, Union, Optional
from tqdm import tqdm
import numpy as np
import xarray as xr
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd

import sys

sys.path.append("/home/tommy/ConceptFormation/neuralhydrology")
from neuralhydrology.modelzoo.basemodel import BaseModel
from scripts.cell_state.cell_state_dataset import (
    CellStateDataset,
    get_train_test_dataset,
    train_validation_split,
)
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class LinearModel(nn.Module):
    def __init__(self, D_in: int, dropout: float = 0.0, **kwargs):
        super(LinearModel, self).__init__(**kwargs)

        #  number of weights == number of dimensions in cell state vector (cfg.hidden_size)
        self.D_in = D_in
        self.model = torch.nn.Sequential(torch.nn.Linear(self.D_in, 1, bias=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data: Dict[str, torch.Tensor]):
        return self.model(self.dropout(data["x_d"].squeeze()))


class NonLinearModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: str = "tanh",
        dropout: float = 0.0,
        **kwargs,
    ):
        super(NonLinearModel, self).__init__(**kwargs)

        self.input_size = input_size
        self.output_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes[:-1]
        self.dropout = dropout

        # create network
        self.activation = self._get_activation(name=activation)
        self._create_model()

    def _create_model(self):
        layers = []

        for ix, hidden_size in enumerate(self.hidden_sizes):
            if ix == 0:
                #  first layer is input_size -> hidden_size[0]
                layers.append(nn.Linear(self.input_size, hidden_size))
            else:
                #  nth layer is previous hidden_size -> current hidden_size
                layers.append(nn.Linear(self.hidden_sizes[ix - 1], hidden_size))

            layers.append(self.activation)
            layers.append(nn.Dropout(p=self.dropout))

        #  final layer is hidden_size[-2] -> hidden_size[-1]
        layers.append(nn.Linear(hidden_size, self.output_size))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, data: Dict[str, torch.Tensor]):
        return self.model(data["x_d"].squeeze())

    def _get_activation(self, name: str) -> nn.Module:
        if name.lower() == "tanh":
            activation = nn.Tanh()
        elif name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif name.lower() == "linear":
            activation = nn.Identity()
        elif name.lower() == "relu":
            activation = nn.ReLU()
        else:
            raise NotImplementedError(
                f"{name} currently not supported as activation in this class"
            )
        return activation

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        for layer in self.net:
            if isinstance(layer, nn.modules.linear.Linear):
                n_in = layer.weight.shape[1]
                gain = np.sqrt(3 / n_in)
                nn.init.uniform_(layer.weight, -gain, gain)
                nn.init.constant_(layer.bias, val=0)


def train_model(
    model: nn.Module,
    train_dataset: CellStateDataset,
    learning_rate: float = 1e-2,
    n_epochs: int = 5,
    l2_penalty: float = 0,
    val_split: bool = False,
    desc: str = "Training Epoch",
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[Any, List[float], List[float]]:
    #  GET loss function
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # GET optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_penalty
    )

    #  TRAIN
    train_losses_ALL = []
    val_losses_ALL = []
    for epoch in tqdm(range(n_epochs), desc=desc):
        train_losses = []
        val_losses = []

        #  new train-validation split each epoch
        if val_split:
            #  create a unique test, val set (random) for each ...
            train_sampler, val_sampler = train_validation_split(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

        for data in train_loader:
            y_pred = model(data)
            y = data["y"][
                :,
            ]
            loss = loss_fn(y_pred, y)

            # train/update the weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())

        # VALIDATE
        if val_split:
            model.eval()
            with torch.no_grad():
                for (basin, time), data in val_loader:
                    X, y = data
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                    val_losses.append(loss.detach().cpu().numpy())

            #  save the epoch-mean losses
            val_losses_ALL = np.mean(val_losses)

        train_losses_ALL.append(np.mean(train_losses))

    return model, train_losses_ALL, val_losses_ALL


#  ALL Training Process
def train_model_loop(
    input_data: xr.Dataset,
    target_data: xr.DataArray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_test: bool = True,
    train_val: bool = False,
    return_loaders: bool = True,
    desc: str = "Training Epoch",
    dropout: float = 0.0,
    device: str = "cpu",
    l2_penalty: float = 0,
    num_workers: int = 4,
) -> Tuple[List[float], BaseModel, Optional[Tuple[DataLoader]]]:
    #  1. create dataset (input, target)
    dataset = CellStateDataset(
        input_data=input_data,
        target_data=target_data,
        device=device,
        start_date=start_date,
        end_date=end_date,
    )
    print("Data Loaded")

    #  2. create train-test split
    if train_test:
        #  build the train, test, validation
        train_dataset, test_dataset = get_train_test_dataset(dataset)
        test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=num_workers
        )
    else:
        train_dataset = dataset
        test_dataset = dataset
        test_loader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=num_workers
        )
    print("Train-Test-Val Split")

    #  3. initialise the model
    model = LinearModel(D_in=dataset.dimensions, dropout=dropout)
    model = model.to(device)

    # 4. Run training loop (iterate over batches)
    print("Start Training")
    model, train_losses, _ = train_model(
        model,
        train_dataset,
        learning_rate=1e-3,
        n_epochs=20,
        val_split=train_val,
        desc=desc,
        l2_penalty=l2_penalty,
    )

    # 5. Save outputs (losses: List[float], model: BaseModel, dataloader: DataLoader)
    if return_loaders:
        return train_losses, model, test_loader
    else:
        return train_losses, model, None


def to_xarray(predictions: Dict[str, List]) -> xr.Dataset:
    return pd.DataFrame(predictions).set_index(["time", "station_id"]).to_xarray()


def calculate_predictions(model: BaseModel, loader: DataLoader) -> xr.Dataset:
    predictions = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for data in loader:
            X, y = data["x_d"], data["y"]
            basin, time = data["meta"]["basin"], data["meta"]["time"]
            y_hat = model(data)

            #  Coords / Dimensions
            predictions["time"].extend(pd.to_datetime(time))
            predictions["station_id"].extend(basin)

            # Variables
            predictions["y_hat"].extend(y_hat.detach().cpu().numpy().flatten())
            predictions["y"].extend(y.detach().cpu().numpy().flatten())

    return to_xarray(predictions)


if __name__ == "__main__":
    from pathlib import Path
    from scripts.cell_state.analysis import get_all_models_weights

    data_dir = Path("/datadrive/data")
    run_dir = data_dir / "runs/complexity_AZURE/hs_064_0306_205514"

    # load in config
    cfg = Config(run_dir / "config.yml")
    cfg.run_dir = run_dir

    #  load in input data
    input_data = xr.open_dataset(data_dir / "SOIL_MOISTURE/norm_cs_data.nc")
    #  load in target data (Soil Moisture)
    target_data = xr.open_dataset(data_dir / "SOIL_MOISTURE/interpolated_esa_cci_sm.nc")

    start_date = pd.to_datetime(input_data.time.min().values)
    end_date = pd.to_datetime(input_data.time.max().values)

    losses_list = []
    models = []
    test_loaders = []

    train_test = True
    train_val = False
    num_workers = 4

    data_vars = [v for v in target_data.data_vars]
    target_features = data_vars if len(data_vars) > 1 else ["sm"]
    for feature in target_features:
        print(f"-- Training Model for {feature} --")
        train_losses, model, test_loader = train_model_loop(
            input_data=input_data,
            target_data=target_data[feature],  #  needs to be xr.DataArray
            train_test=train_test,
            train_val=train_val,
            return_loaders=True,
            start_date=start_date,
            end_date=end_date,
            num_workers=num_workers,
            l2_penalty=2,
        )

        # store outputs of training process
        losses_list.append(train_losses)
        models.append(model)
        test_loaders.append(test_loader)

    #  run forward pass and convert to xarray object
    print("-- Running Test-set Predictions --")
    preds = calculate_predictions(model, test_loader)

    # extract weights and biases
    print("-- Extracting weights and biases --")
    ws, bs = get_all_models_weights(models)

    #  calculate raw correlations (cell state and values)
    assert False, "No need to reload the CellStateDataset"
