from bubblewrap import Bubblewrap, BWRun, AnimationManager, SymmetricNoisyRegressor
from bubblewrap.default_parameters import default_jpca_dataset_parameters
from bubblewrap.regressions import NearestNeighborRegressor
from bubblewrap.input_sources.data_sources import NumpyDataSource, PairWrapperSource
import numpy as np
import time
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models


def inner_evaluate(parameters, scrap=0):
    obs, beh = NumpyDataSource.get_from_saved_npz("jpca_reduced_sc.npz", time_offsets=(0, 1, 5))
    ds = PairWrapperSource(obs, beh)
    ds.shorten(scrap)

    bw = Bubblewrap(dim=ds.output_shape[0], **parameters)

    reg = NearestNeighborRegressor(input_d=bw.N, output_d=1, maxlen=600)

    br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, show_tqdm=True,
               output_directory="/home/jgould/Documents/Bubblewrap/generated/bubblewrap_runs/")

    start = time.time()
    br.run(limit=1400, save=False)
    runtime = time.time() - start
    last = br.prediction_history[1][-300:]
    return {
        "bw_pred_error": np.mean(last),
        "runtime": runtime,
        "regression_mse": np.nanmean(np.array(br.behavior_error_history[1][-300:]) ** 2)
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


def make_generation_strategy():
    gs = GenerationStrategy(
        steps=[
            # Quasi-random initialization step
            GenerationStep(
                model=Models.SOBOL,
                num_trials=2,  # How many trials should be produced from this generation step
            ),
            # Bayesian optimization step using the custom acquisition function
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
                # model_kwargs={
                #     "surrogate": Surrogate(SingleTaskGP),
                # },
            ),
        ]
    )
    return gs


def make_ax_client(generation_strategy):
    ax_client = AxClient(generation_strategy=generation_strategy)

    ax_client.create_experiment(
        name="bw_test_experiment",
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
                "name": "lam",
                "type": "range",
                "bounds": [1e-10, 1],
                "value_type": 'float',
                "log_scale": True,
            },
            {
                "name": "nu",
                "type": "range",
                "bounds": [1e-10, 1],
                "value_type": 'float',
                "log_scale": True,
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
            "bw_pred_error": ObjectiveProperties(minimize=False),
        },
        tracking_metric_names=[
            "runtime",
            "regression_mse"
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
        _, idx = ax_client.attach_trial(p)
        ax_client.complete_trial(idx, dd)


def main():
    gs = make_generation_strategy()
    ax_client = make_ax_client(gs)

    # manually_add_old_trials(ax_client, "old_ax.json")

    for i in range(250):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
        if i % 10 == 1:
            ax_client.save_to_json_file(
                f"/home/jgould/Bubblewrap/generated/optim/ax_{ax_client.experiment.name}_snapshot.json")


if __name__ == '__main__':
    # IMPORTANT: bubblewrap has to run once before Ax gets started or things break
    evaluate({"num": 5, "B_thresh": 0})
    main()
