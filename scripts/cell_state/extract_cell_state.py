import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, DefaultDict
from collections import defaultdict
from torch import Tensor
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from copy import deepcopy

import sys

sys.path.append("/home/tommy/ConceptFormation/neuralhydrology")
from neuralhydrology.evaluation.tester import RegressionTester
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.datautils.utils import load_basin_file
from neuralhydrology.utils.config import Config
from scripts.read_model import get_model, _load_weights


def subset_data_dicts(
    predictions: Dict[str, Tensor], predict_last_n: int = 1
) -> Dict[str, Tensor]:
    #  get only the final prediction
    subset_preds = {}
    for key in predictions.keys():
        subset_preds[key] = predictions[key][:, -predict_last_n:, :]
    return subset_preds


def convert_to_xarray(
    cfg: Config, output: DefaultDict[str, Tensor], key: str, basin: str
) -> xr.Dataset:
    data = output[key].detach().cpu().numpy()
    date_range = pd.date_range(
        start=cfg.test_start_date, end=cfg.test_end_date, freq="1D"
    )

    return xr.Dataset(
        {key: (("date", "station_id", "dimension"), data)},
        coords={
            "date": date_range,
            "station_id": [basin],
            "dimension": np.arange(data.shape[-1]),
        },
    )

def _extract_cell_state_from_model_forward_pass_train_data(
    cfg: Config, run_dir: Path, model: BaseModel
) -> List[xr.Dataset]:
    # overwrite the config file to use the train dataset (in time and space)
    new_cfg = deepcopy(cfg)
    new_cfg._cfg["test_start_date"] = cfg.train_start_date
    new_cfg._cfg["test_end_date"] = cfg.train_end_date
    new_cfg._cfg["test_basin_file"] = cfg.train_basin_file

    all_cn_xr = _extract_cell_state_from_model_forward_pass_test_data(cfg=new_cfg, run_dir=run_dir, model=model)
    return all_cn_xr


def _extract_cell_state_from_model_forward_pass_test_data(
    cfg: Config, run_dir: Path, model: BaseModel
) -> List[xr.Dataset]:
    #  get dataloader / dataset
    tester = RegressionTester(cfg, run_dir=run_dir)

    #  do a forward pass and store the output c_n for each basin
    basins = load_basin_file(cfg.test_basin_file)
    pbar = tqdm(basins, desc="Extracting cell state for basin")
    all_cn_xr = []
    for basin in pbar:
        output = defaultdict(Tensor)
        ds = tester._get_dataset(basin)
        loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)
        with torch.no_grad():
            for data in loader:

                for key in data:
                    data[key] = data[key].to(cfg.device)
                predictions = model(data)

                # get only the final indices!
                subset_preds = subset_data_dicts(
                    predictions, predict_last_n=cfg.predict_last_n
                )
                subset_preds["y"] = data["y"][:, -cfg.predict_last_n :, :]

                #  concatenate these tensors by time (index=0)
                #  NOTE: should be the same size as the test date_range
                for key in [k for k in subset_preds.keys()]:
                    output[key] = torch.cat(
                        (output[key], subset_preds[key].detach().cpu()), 0
                    )

        all_cn_xr.append(convert_to_xarray(cfg, output, key="c_n", basin=basin))

    return all_cn_xr


def get_cell_states(cfg: Config, run_dir: Path, subset: str = "test") -> xr.Dataset:
    #  get model
    model = get_model(cfg).to(cfg.device)
    _load_weights(model, cfg)

    #  get cell state xarray objects
    if subset == "train":
        all_cn_xr = _extract_cell_state_from_model_forward_pass_train_data(cfg, run_dir, model)
    elif subset == "test":
        all_cn_xr = _extract_cell_state_from_model_forward_pass_test_data(cfg, run_dir, model)
    else:
        assert False, f"Expect one of [train test] got: {subset}"
    return xr.concat(all_cn_xr, dim="station_id")


def ensemble_dir_get_all_cell_states(ensemble_dir: Path, subset: str = "test") -> None:
    run_dirs = [d for d in ensemble_dir.iterdir() if (d.is_dir())]
    pbar = tqdm(run_dirs, desc="Extracting cell state for ensemble members")
    for run_dir in pbar:
        if (out_dir / "cell_states.nc").exists():
            pbar.set_postfix_str(run_dir.name)
            # load the config file
            cfg = Config(run_dir / "config.yml") 

            # create the cell state xr object
            cn = get_cell_states(cfg, run_dir, subset=subset)

            # write to netcdf file
            out_dir = run_dir / "cell_states"
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)

            cn.to_netcdf(out_dir / "cell_states.nc")
        else:
            continue


if __name__ == "__main__":
    run_dir = Path("/datadrive/data/runs/complexity_AZURE/hs_064_0306_205514")

    # load config
    cfg = Config(run_dir / "config.yml")
    cfg.run_dir = run_dir

    #  get cell states
    subset = "train"
    cn = get_cell_states(cfg, run_dir, subset=subset)

    out_dir = run_dir / "cell_states"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    cn.to_netcdf(out_dir / f"{subset}_cell_states.nc")
