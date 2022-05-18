from typing import Tuple, Optional, List
from collections import defaultdict
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.cell_state.cell_state_model import LinearModel
from scripts.cell_state.timeseries_dataset import TimeSeriesDataset
from scripts.cell_state.timeseries_model import _round_time_to_hour
from scripts.cell_state.cell_state_model import to_xarray


class PytorchProbe:
    def __init__(
        self, 
        input_features: int,
        l1_ratio: Optional[float] = 0.15,
        learning_rate: float = 0.001,
    ):
        self.input_features = input_features
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio

        #  initalise model
        self.init_model()
        self.init_optimizer()
        self.init_loss()

    def init_model(self):
        self.model = LinearModel(D_in=self.input_features)

    def init_optimizer(self):
        # GET optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0
        )

    def init_loss(self):
        #  GET loss
        self.loss_fn = nn.MSELoss()

    def load_data(self):
        self._create_datasets(self)

    def _create_datasets(
        self,
        target_ds: xr.Dataset,
        input_ds: xr.Dataset,
        target_variable: str,
        input_variables: List[str] = [f"dim{i}" for i in np.arange(64)],
        train_start_date: pd.Timestamp = pd.to_datetime("1998-01-01"),
        train_end_date: pd.Timestamp = pd.to_datetime("2006-09-30"),
        test_start_date: pd.Timestamp = pd.to_datetime("2006-10-01"),
        test_end_date: pd.Timestamp = pd.to_datetime("2009-10-01"),
        seq_length: int = 1,
        basin_dim: str = "station_id",
        time_dim: str = "time",
    ) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
        # train test split
        target_data = target_ds.sel(time=slice(train_start_date, train_end_date))
        input_data = input_ds.sel(time=slice(train_start_date, train_end_date))

        test_target_data = target_ds.sel(time=slice(test_start_date, test_end_date))
        test_input_data = input_ds.sel(time=slice(test_start_date, test_end_date))

        #  create datasets
        train_dataset = TimeSeriesDataset(
            input_data=input_data,
            target_data=target_data,
            target_variable=target_variable,
            input_variables=input_variables,
            seq_length=seq_length,
            basin_dim=basin_dim,
            time_dim=time_dim,
            desc="Creating Train Samples",
        )

        test_dataset = TimeSeriesDataset(
            input_data=test_input_data,
            target_data=test_target_data,
            target_variable=target_variable,
            input_variables=input_variables,
            seq_length=seq_length,
            basin_dim=basin_dim,
            time_dim=time_dim,
            desc="Creating Test Samples",
        )

        return train_dataset, test_dataset

    def init_dataloaders(self, num_workers: int = 0, batch_size: int = 1000):
        train_dataset, test_dataset = self._create_datasets()
        #  create dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )

    def train(self, n_epochs: int = 10):
        self.epoch_losses = []
        mean_epoch_loss = 9999
        for epoch in range(n_epochs):
            losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for data in pbar:
                pbar.set_postfix_str(f"{mean_epoch_loss:.2f}")
                y_pred = self.model(data)
                y = data["y"].squeeze(1)

                #  calculate loss
                loss = self.loss_fn(y_pred, y)

                #  add regularisation terms
                if self.l1_ratio is not None:
                    #  l1 loss-penalty (1st order magnitude, vector of weights)
                    loss = loss + (
                        self.l1_ratio
                        * torch.norm(
                            torch.cat(
                                [param.view(-1) for param in self.model.parameters()]
                            ),
                            p=1,
                        )
                    )
                    #  l2 loss-penalty (2nd order magnitude, vector of weights)
                    loss = loss + (
                        (1 - self.l1_ratio)
                        * torch.square(
                            torch.norm(
                                torch.cat(
                                    [
                                        param.view(-1)
                                        for param in self.model.parameters()
                                    ]
                                ),
                                p=2,
                            )
                        )
                    )

                losses.append(loss.detach().numpy())

                # train/update the weight
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            mean_epoch_loss = float(np.mean(losses))
            self.epoch_losses.append(mean_epoch_loss)

    def evaluate(self) -> xr.Dataset:
        predictions = defaultdict(list)

        with torch.no_grad():
            for data in tqdm(self.test_loader, "Evaluation Forward Pass"):
                y_hat = self.model(data).squeeze()
                y = data["y"].squeeze()
                basin, time = (
                    data["meta"]["spatial_unit"].numpy(),
                    data["meta"]["time"].numpy(),
                )

                #  Coords / Dimensions
                predictions["time"].extend(
                    _round_time_to_hour(
                        pd.to_datetime([t[0] for t in time.astype("datetime64[ns]")])
                    )
                )
                predictions["station_id"].extend(basin)

                # Variables
                predictions["y_hat"].extend(y_hat.detach().cpu().numpy().flatten())
                predictions["y"].extend(y.detach().cpu().numpy().flatten())

        return to_xarray(predictions)


if __name__ == "__main__":
    pass
