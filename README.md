# Hydrological Concept Formation inside Long Short-Term Memory (LSTM) networks
Code for our paper [Hydrological Concept Formation inside Long Short-Term Memory (LSTM) networks](https://hess.copernicus.org/preprints/hess-2021-566/#discussion). 

The results that this code produces can be found on [zenodo here](https://zenodo.org/record/5600851#.YoU4SxPMJHk). 

**neuralhydrology**: [deep learning codebase copied at version for paper](https://neuralhydrology.readthedocs.io/en/latest/) that includes the LSTM used in this paper

# Notebooks
There are a lot of experimental notebooks. The important notebooks are:
1) The main body of Results: `notebooks/19a_ERA5_land_probes.ipynb`
1) Appendix B: Probing the ESA CCI Soil Moisture `notebooks/19b_ESA_CCI_probes.ipynb`
1) Appendix C: Investigating the Catchment Specific Probe Offsets `notebooks/19c_FIX_Probe_Offsets.ipynb`

# Scripts
**cell_state**: functions used in the notebooks for training linear probes
1) `scripts/cell_state/extract_cell_state.py` -- from a neuralhydrology trained model path, save the cell_states associated with that model for each basin
1) `scripts/cell_state/sklearn_models.py` -- fit, predict and evaluate models from sklearn ([`init_linear_model`, `init_nonlinear_model`]) 
1) `scripts/cell_state/analysis.py` -- `get_model_weights` from linear regression models,`load_probe_components` get probe results from how they're saved 
1) `scripts/cell_state/timeseries_dataset.py` -- `TimeSeriesDataset` object for training probes
1) `scripts/cell_state/timeseries_model.py` -- train pytorch linear model for soil moisture // snow

Marginalia:
1) `scripts/cell_state/jdb_fitting.py`
1) `scripts/cell_state/baseline.py` -- empty :( (but look at `notebooks/07_baseline_cell_state.ipynb` for the random noise baselines)
1) `scripts/cell_state/cell_state_dataset.py` -- create `CellStateDataset` for cell state and target data (for training pytorch models)
1) `scripts/cell_state/normalize.py` -- helper functions for normalizing cell states
1) `scripts/cell_state/utils.py` -- helper functions for training a series of models `train_and_evaluate_models`
1) `scripts/cell_state/cell_state_model.py` -- `LinearModel` used by `pytorch_probe.py`
1) `scripts/cell_state/pytorch_probe.py` -- `PytorchProbe`

**probe_data**: the data used for training the probes (the probe targets from ERA5Land)
1) `scripts/probe_data/get_era5_sm.py` -- Download the SM and SWE data from ERA5 Land via cds api
2) `scripts/probe_data/era5_hourly_to_daily.py` -- Resample Data to daily
3) `scripts/probe_data/join_daily_to_onefile.py` -- Merge all of the data into one netcdf file
4) `scripts/clip_netcdf_to_shapefile.py` -- Convert into basin timeseries (chop out regions from .shp)

**extra_features**: `join_camels_to_era5land` - create one big netcdf file with the original camels data and the era5_land variables (`swvl{1,2,3,4}` - snow water volume level {1,2,3,4} & `sd` - snow depth)

**geospatial**: geospatial helper functions for plotting location data from camelsGB `plot_spatial_location`

**plots**: plot nice hydrographs with `plot_hydrograph`

**read_model**: load model weights from a neuralhydrology.Model with `_load_weights`

**read_nh_results**: helper functions for reading result files created by `neuralhydrology`

**static_correlations**: plotting functions for getting relationship between statics and nse scores

**utils** `get_data_dir` // `move_directory` // `corr_df_ready_for_plotting`

__Marginalia__:
**cwatm_data**: functions for getting cwatm data into camels dataset format 
1) `scripts/cwatm_data/cwatm_to_camels_dataset.py` -- functions for building TARGET, INPUT, HIDDEN datasets
2) `scripts/cwatm_data/ldd_basins.py` -- river drainage basins from drainage direction (LDD - local drainage direction) [for cwatm](https://cwatm.iiasa.ac.at/data.html) using `pyflwdir`
3) `scripts/cwatm_data/masked_mean_cwatm_data.py` -- build mean data over catchment areas for TARGET, INPUT, HIDDEN 

**dataset_to_camelsgb_format** = functions for creating CAMELSGB dataset for CWatM Data

**integrated_gradients**: code for calculating integrated gradients (runs slowly)

# Notes about this repository:
Pickle files generated using Xarray 0.20.1 