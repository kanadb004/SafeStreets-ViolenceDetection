from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pytest
import tf_keras
import yaml

from safestreets.models.factory import build_model
from safestreets.training.tune import (
    LSTM_UNITS_CHOICES,
    OptunaPruningCallback,
    WinnerConfig,
    best_winner_config,
    resolve_study,
    run_trials,
    write_best_model_yaml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.sprint3
def test_sample_params_within_search_space() -> None:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    from safestreets.training.tune import _sample_params

    params = _sample_params(trial)
    assert 1e-5 <= params["learning_rate"] <= 1e-2
    assert params["lstm_units"] in LSTM_UNITS_CHOICES
    assert 0.2 <= params["dropout"] <= 0.6
    assert params["batch_size"] in (16, 32, 64)


@pytest.mark.sprint3
def test_pruning_callback_raises_trial_pruned() -> None:
    """A trial whose pruner says `should_prune()` must abort `model.fit` via
    `optuna.TrialPruned`, not silently continue."""
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0),
    )
    # Seed one strong completed trial so the pruner has something to compare against.
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=0.99,
            params={},
            distributions={},
            intermediate_values={0: 0.99},
        )
    )
    trial = study.ask()
    callback = OptunaPruningCallback(trial, monitor="val_auc")

    model = tf_keras.Sequential([tf_keras.layers.Dense(1, activation="sigmoid", input_shape=(2,))])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[])
    x = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros((4, 1), dtype=np.float32)

    with pytest.raises(optuna.TrialPruned):
        model.fit(
            x,
            y,
            epochs=3,
            verbose=0,
            callbacks=[tf_keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: callback.on_epoch_end(epoch, {"val_auc": 0.01})
            )],
        )


def _trivial_objective(trial: optuna.Trial) -> float:
    """A cheap, non-TF stand-in for the real training objective: `run_study`
    delegates the resume/trial-count decision entirely to `resolve_study` and
    `run_trials`, so exercising that logic does not need a real model fit
    (and avoids this machine's TF-Metal plugin, which is fragile under many
    back-to-back Keras graphs in one process, per docs/decisions/ADR-004-hpo.md).
    """
    return trial.suggest_float("x", 0.0, 1.0)


@pytest.mark.sprint3
def test_run_study_resumes_trial_count(tmp_path: Path) -> None:
    storage_path = tmp_path / "optuna" / "study.db"
    sampler = optuna.samplers.TPESampler(seed=1265)
    pruner = optuna.pruners.MedianPruner()

    study1 = resolve_study(
        storage_path, "test-resume", resume=False, sampler=sampler, pruner=pruner
    )
    run_trials(study1, _trivial_objective, n_trials=1)
    assert len(study1.trials) == 1

    study2 = resolve_study(
        storage_path, "test-resume", resume=True, sampler=sampler, pruner=pruner
    )
    run_trials(study2, _trivial_objective, n_trials=2)
    assert len(study2.trials) == 2


@pytest.mark.sprint3
def test_run_study_without_resume_rejects_existing_study(tmp_path: Path) -> None:
    storage_path = tmp_path / "optuna" / "study.db"
    sampler = optuna.samplers.TPESampler(seed=1265)
    pruner = optuna.pruners.MedianPruner()

    study = resolve_study(
        storage_path, "test-noresume", resume=False, sampler=sampler, pruner=pruner
    )
    run_trials(study, _trivial_objective, n_trials=1)
    with pytest.raises(ValueError):
        resolve_study(
            storage_path, "test-noresume", resume=False, sampler=sampler, pruner=pruner
        )


@pytest.mark.sprint3
def test_write_best_model_yaml_reproduces_architecture(tmp_path: Path) -> None:
    winner = WinnerConfig(
        units=256,
        dropout=0.35,
        learning_rate=5e-4,
        batch_size=32,
        best_epoch=10,
        best_val_auc=0.91,
        trial_number=7,
    )
    out_path = tmp_path / "model.best.yaml"
    write_best_model_yaml(winner, REPO_ROOT / "configs" / "model.yaml", out_path)

    with open(out_path) as f:
        best_cfg = yaml.safe_load(f)

    assert best_cfg["arch"] == "lstm_head"
    assert best_cfg["lstm_head"]["units"] == 256
    assert best_cfg["lstm_head"]["dropout"] == pytest.approx(0.35)
    assert best_cfg["lstm_head"]["learning_rate"] == pytest.approx(5e-4)
    assert best_cfg["tuned_batch_size"] == 32

    model = build_model(best_cfg, pos_weight=1.0)
    lstm_layers = [layer for layer in model.layers if isinstance(layer, tf_keras.layers.LSTM)]
    assert lstm_layers[0].units == 256


@pytest.mark.sprint3
def test_best_winner_config_reads_trial_params_and_user_attrs() -> None:
    study = optuna.create_study(direction="maximize")
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=0.87,
        params={
            "learning_rate": 0.0012,
            "lstm_units": 128,
            "dropout": 0.4,
            "batch_size": 16,
        },
        distributions={
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "lstm_units": optuna.distributions.CategoricalDistribution(list(LSTM_UNITS_CHOICES)),
            "dropout": optuna.distributions.FloatDistribution(0.2, 0.6),
            "batch_size": optuna.distributions.CategoricalDistribution([16, 32, 64]),
        },
        user_attrs={"best_epoch": 12},
    )
    study.add_trial(trial)

    winner = best_winner_config(study)
    assert winner.units == 128
    assert winner.batch_size == 16
    assert winner.best_epoch == 12
    assert winner.best_val_auc == pytest.approx(0.87)
