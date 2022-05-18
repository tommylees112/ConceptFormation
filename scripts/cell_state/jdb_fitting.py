import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from pathlib import Path
from scripts.cell_state.utils import create_train_test_default_dict_for_all_target_vars
from scripts.cell_state.utils import train_and_evaluate_models


def reload_all_train_test(data_dir: Path, filepath: str = "all_train_test.pkl", reload: bool = True):
    RELOAD = reload

    if not (data_dir / filepath).exists() or not RELOAD:
        all_train_test = create_train_test_default_dict_for_all_target_vars(
            train_cs=train_cs,
            test_cs=test_cs,
            train_target_ds=train_target_ds,
            test_target_ds=test_target_ds,
            input_variables=input_variables,
        )
        with (data_dir / filepath).open("wb") as f:
            pickle.dump(all_train_test, f)
    else:
        all_train_test = pickle.load((data_dir / filepath).open("rb"))

    return all_train_test


if __name__ == "__main__":
    data_dir = Path("/DataDrive200/data")
    all_train_test = reload_all_train_test(data_dir)
    all_models_preds = train_and_evaluate_models(all_train_test, sgd=True)

    # timeseries plotting
    time = slice("1998", "2008")
    pixels = [52010, 15021]

    target_vars = [f"swvl{ix+1}" for ix in range(4)]
    for px in pixels:
        f, axs = plt.subplots(2, 2, figsize=(12, 4), sharex=True)
        for ix, target_var in enumerate(target_vars):
            ax = axs[np.unravel_index(ix, (2, 2))]
            preds = all_models_preds[target_var]["preds"]
            data = preds.sel(station_id=px, time=time)

            ax.plot(data.time, data.obs, color="k", ls="--", alpha=0.3, label="Observed")
            ax.plot(data.time, data.sim, color=f"C{ix}", ls="-", alpha=0.6, label="Simulated")
            ax.set_title(f"{target_var}")
            if ix == 0:
                ax.legend()
            sns.despine()
        f.suptitle(px)

    # histogram plotting 
    for metric in ["Pearson-r", "RMSE", "NSE"]:

            f, ax = plt.subplots(figsize=(12, 4))

            #  metric = "Pearson-r"
            colors = sns.color_palette("viridis", n_colors=len(target_vars))
            for ix, target_var in enumerate(target_vars):
                errors = all_models_preds[target_var]["errors"]
                nse = errors[metric]

                ax.hist(nse.where(nse > -1, -1), bins=100, density=True, label=f"{target_var}: {nse.median().values:.2f}", alpha=0.6, color=colors[ix]);
                ax.axvline(nse.median(), color=colors[ix], ls="--", alpha=0.5)
            ax.set_xlabel(metric if metric != "Pearson-r" else "Pearson Correlation")
            ax.set_xlim(0.25, 1) if metric in ["Pearson-r"] else None
            ax.set_xlim(-1, 1) if metric in ["NSE"] else None
            ax.legend()
            sns.despine()