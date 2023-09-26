from bubblewrap import Bubblewrap, BWRun, AnimationManager, SymmetricNoisyRegressor
from bubblewrap.regressions import NearestNeighborRegressor
from bubblewrap.default_parameters import default_jpca_dataset_parameters
from bubblewrap.input_sources.data_sources import NumpyDataSource, PairWrapperSource
import numpy as np


def inner_evaluate(parameters, scrap=0):
    obs, beh = NumpyDataSource.get_from_saved_npz("jpca_reduced_sc.npz", time_offsets=(2,))
    ds = PairWrapperSource(obs, beh)
    ds.shorten(scrap)

    bw = Bubblewrap(dim=ds.output_shape[0], **parameters)

    reg = NearestNeighborRegressor(input_d=bw.N, output_d=1)

    br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, show_tqdm=True,
               output_directory="/home/jgould/Documents/Bubblewrap/generated/bubblewrap_runs/")

    br.run(limit=1400, save=False)

    steps = 2
    last = br.prediction_history[steps][-300:]
    last_e = br.entropy_history[steps][-300:]

    bp_last = np.array(br.behavior_error_history[steps][-300:]) ** 2
    mse = np.mean(bp_last)

    # if np.any(np.isnan(bp_last)):
    #     mse = np.quantile(a=bp_last[np.isfinite(bp_last)], q=.9)
    # mse = min(mse, 1e5) + (mse % 10)

    return {
        "bw_log_pred_pr": np.mean(last),
        "runtime": br.runtime,
        "regression_mse": mse,
        "entropy":np.mean(last_e)
    }


def evaluate(parameters):
    results = {}
    for to_scrap in [0, 25, 50, 100, 150, 175]:
        inner_parameters = dict(default_jpca_dataset_parameters, **parameters)
        for key, value in inner_evaluate(inner_parameters, scrap=to_scrap).items():
            results[key] = results.get(key, []) + [value]
    for key, value in results.items():
        results[key] = (np.mean(value), np.std(value) / np.sqrt(len(value)))
    return results

inner_evaluate({})


from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
import botorch
from ax.storage.registry_bundle import RegistryBundle


def make_generation_strategy():
    gs = GenerationStrategy(
        steps=[
            # Quasi-random initialization step
            GenerationStep(
                model=Models.SOBOL,
                num_trials=10,  # How many trials should be produced from this generation step
            ),
            # Bayesian optimization step using the custom acquisition function
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
                model_kwargs={
                    # "surrogate": Surrogate(botorch.models.SingleTaskGP),
                    # "botorch_acqf_class": botorch.acquisition.ExpectedImprovement
                },
            ),
        ]
    )
    return gs




def make_ax_client(generation_strategy):
    ax_client = AxClient(generation_strategy=generation_strategy)

    ax_client.create_experiment(
        name="bw_reg_2step_mse",
        parameters=[
            {
                "name": "num",
                "type": "range",
                "bounds": [5, 2000],
                "value_type": 'int',
                "log_scale": True,
            },
            {
                "name": "B_thresh",
                "type": "range",
                "bounds": [-30.0, 0.0],
                "value_type": 'float',
            },
            {
                "name": "eps",
                "type": "range",
                "bounds": [1e-10, 1],
                "value_type": 'float',
                "log_scale": True,
            },

        ],
        objectives={
            "regression_mse": ObjectiveProperties(minimize=True),
        },
        tracking_metric_names=[
            "runtime",
            "entropy",
            "bw_log_pred_pr"
        ]
    )

    return ax_client


def manually_add_old_trials(ax_client, fname):
    old_ax_client = AxClient.load_from_json_file(fname)
    for key, value in old_ax_client.experiment.trials.items():
        df = list(old_ax_client.experiment.data_by_trial[key].values())[0].df
        dd = {}
        for i in df.index:
            dd[df.loc[i, "metric_name"]] = (df.loc[i, "mean"], df.loc[i, "sem"])
        p = value.arm.parameters
        try:
            _, idx = ax_client.attach_trial(p)
        except ValueError:
            continue
        ax_client.complete_trial(idx, dd)


def main():
    gs = make_generation_strategy()
    ax_client = make_ax_client(gs)

    # manually_add_old_trials(ax_client, f"/home/jgould/Documents/Bubblewrap/generated/optim/ax_{ax_client.experiment.name}_snapshot.json")

    for i in range(250):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
        if i % 5 == 1:
            ax_client.save_to_json_file(
                f"/home/jgould/Documents/Bubblewrap/generated/optim/ax_{ax_client.experiment.name}_snapshot.json")


if __name__ == '__main__':
    # IMPORTANT: bubblewrap has to run once before Ax gets imported or things break
    main()
