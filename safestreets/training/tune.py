"""Optuna TPE search over the `lstm_head`'s four hyperparameters named in
report §4.4: `learning_rate`, `lstm_units`, `dropout`, `batch_size`. Single
objective, `val_auc`, maximised. See docs/SPRINT_PLAN.md Sprint 3 and
docs/decisions/ADR-004-hpo.md.
"""

from __future__ import annotations

# isort: off
import safestreets.config  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TF import below)

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import tf_keras
import yaml

from safestreets.evaluation.cross_dataset import build_matrix
from safestreets.evaluation.evaluate import load_checkpoint, score_split
from safestreets.evaluation.metrics import compute_metrics
from safestreets.models.heads import LSTMHeadConfig, build_lstm_head
from safestreets.training.losses import compute_pos_weight, weighted_bce
from safestreets.training.train import TrainConfig, load_feature_split

# isort: on

STUDY_NAME = "safestreets-sprint3-lstm-head"
TUNE_EPOCHS = 25
LSTM_UNITS_CHOICES = (64, 128, 256, 512)
BATCH_SIZE_CHOICES = (16, 32, 64)


class OptunaPruningCallback(tf_keras.callbacks.Callback):
    """Reports `monitor` to the trial each epoch and raises `TrialPruned` the
    moment Optuna's pruner says this trial is unpromising.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_auc") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def _sample_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "lstm_units": trial.suggest_categorical("lstm_units", list(LSTM_UNITS_CHOICES)),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "batch_size": trial.suggest_categorical("batch_size", list(BATCH_SIZE_CHOICES)),
    }


def build_objective(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: float,
    seed: int,
    epochs: int = TUNE_EPOCHS,
):
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        tf_keras.utils.set_random_seed(seed)
        cfg = LSTMHeadConfig(
            units=params["lstm_units"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )
        model = build_lstm_head(cfg)
        model.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss=weighted_bce(pos_weight),
            metrics=[tf_keras.metrics.AUC(name="auc")],
        )
        history = model.fit(
            x_train,
            y_train.astype(np.float32),
            validation_data=(x_val, y_val.astype(np.float32)),
            batch_size=params["batch_size"],
            epochs=epochs,
            callbacks=[OptunaPruningCallback(trial, monitor="val_auc")],
            verbose=0,
        )
        val_auc = [float(v) for v in history.history["val_auc"]]
        best_epoch = int(np.argmax(val_auc)) + 1
        trial.set_user_attr("best_epoch", best_epoch)
        return max(val_auc)

    return objective


def resolve_study(
    storage_path: Path | str,
    study_name: str,
    resume: bool,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
) -> optuna.Study:
    """Creates, or resumes, a SQLite-backed study. Without `resume`, refuses to
    silently continue a study that already exists on disk under this name;
    with `resume=True`, continues it (`optuna.create_study(load_if_exists=True)`
    is itself a no-op resume once the study row exists, so the guard above it
    is what actually distinguishes "fresh run" from "--resume" for the caller).
    Split out from `run_study` so the resume/trial-count behavior is testable
    without a real training objective, per the Sprint 3 DoD.
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    if not resume and storage_path.exists():
        existing = optuna.study.get_all_study_summaries(storage)
        if any(s.study_name == study_name for s in existing):
            raise ValueError(
                f"study {study_name!r} already exists at {storage_path}; pass resume=True "
                "(scripts/tune.py --resume) to continue it, or delete the file to start over."
            )

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def run_trials(
    study: optuna.Study, objective, n_trials: int, timeout: int | None = None
) -> optuna.Study:
    """Runs however many trials are needed to bring `study` up to `n_trials`
    total, zero if it is already there (the resume case).
    """
    remaining = max(n_trials - len(study.trials), 0)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=timeout)
    return study


def run_study(
    train_cfg: TrainConfig,
    storage_path: Path | str,
    n_trials: int = 30,
    timeout: int | None = None,
    seed: int = 1265,
    study_name: str = STUDY_NAME,
    resume: bool = False,
    epochs: int = TUNE_EPOCHS,
) -> optuna.Study:
    """Creates (or resumes, if `resume`) a SQLite-backed study and runs enough
    trials to reach `n_trials` total. Re-running with the same storage and
    `resume=True` after an interruption continues the trial count instead of
    restarting it, per the Sprint 3 DoD.
    """
    x_train, y_train, _ = load_feature_split(
        train_cfg.features_dir, train_cfg.train_datasets, "train"
    )
    x_val, y_val, _ = load_feature_split(train_cfg.features_dir, train_cfg.val_datasets, "val")
    pos_weight = compute_pos_weight(y_train)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = resolve_study(storage_path, study_name, resume, sampler, pruner)
    objective = build_objective(x_train, y_train, x_val, y_val, pos_weight, seed, epochs)
    return run_trials(study, objective, n_trials, timeout)


@dataclass(frozen=True)
class WinnerConfig:
    units: int
    dropout: float
    learning_rate: float
    batch_size: int
    best_epoch: int
    best_val_auc: float
    trial_number: int


def best_winner_config(study: optuna.Study) -> WinnerConfig:
    best = study.best_trial
    return WinnerConfig(
        units=best.params["lstm_units"],
        dropout=best.params["dropout"],
        learning_rate=best.params["learning_rate"],
        batch_size=best.params["batch_size"],
        best_epoch=best.user_attrs.get("best_epoch", TUNE_EPOCHS),
        best_val_auc=best.value,
        trial_number=best.number,
    )


def write_best_model_yaml(
    winner: WinnerConfig, base_model_cfg_path: Path | str, out_path: Path | str
) -> dict:
    """Writes `configs/model.best.yaml`: same schema as `configs/model.yaml`
    with `lstm_head` replaced by the winning trial's architecture. Loading it
    through `safestreets.models.factory.build_model` reproduces that
    architecture exactly.
    """
    with open(base_model_cfg_path) as f:
        base = yaml.safe_load(f)

    best_cfg = {
        "arch": "lstm_head",
        "lstm_head": {
            "input_dim": base["lstm_head"]["input_dim"],
            "n_frames": base["lstm_head"]["n_frames"],
            "units": winner.units,
            "dropout": winner.dropout,
            "learning_rate": winner.learning_rate,
        },
        "scratch": base["scratch"],
        "tuned_batch_size": winner.batch_size,
        "tuned_best_epoch": winner.best_epoch,
        "tuned_best_val_auc": winner.best_val_auc,
        "tuned_trial_number": winner.trial_number,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    return best_cfg


def retrain_winner_on_train_plus_val(
    winner: WinnerConfig,
    train_cfg: TrainConfig,
    checkpoints_dir: Path | str,
    run_tag: str = "lstm_head_tuned",
) -> dict:
    """Final fit: the winning architecture, trained on train+val combined (no
    held-out split remains, so training runs for exactly `winner.best_epoch`
    epochs, the epoch count at which the winning trial peaked during tuning,
    rather than early-stopping on a validation signal that no longer exists).
    """
    x_train, y_train, _ = load_feature_split(
        train_cfg.features_dir, train_cfg.train_datasets, "train"
    )
    x_val, y_val, _ = load_feature_split(train_cfg.features_dir, train_cfg.val_datasets, "val")
    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    pos_weight = compute_pos_weight(y_all)
    cfg = LSTMHeadConfig(
        units=winner.units, dropout=winner.dropout, learning_rate=winner.learning_rate
    )
    model = build_lstm_head(cfg)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=weighted_bce(pos_weight),
        metrics=[
            tf_keras.metrics.BinaryAccuracy(name="accuracy"),
            tf_keras.metrics.AUC(name="auc"),
            tf_keras.metrics.Precision(name="precision"),
            tf_keras.metrics.Recall(name="recall"),
        ],
    )
    model.fit(
        x_all,
        y_all.astype(np.float32),
        batch_size=winner.batch_size,
        epochs=winner.best_epoch,
        verbose=2,
    )
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{run_tag}.keras"
    model.save(checkpoint_path)
    return {"run_tag": run_tag, "checkpoint": str(checkpoint_path), "pos_weight": pos_weight}


def render_optuna_figures(study: optuna.Study, figures_dir: Path | str) -> None:
    """`optuna_history.png` (objective per trial, pruned trials marked) and
    `optuna_importances.png` (hyperparameter importances). Both DoD items.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ax = plot_optimization_history(study)
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "optuna_history.png", dpi=120)
    plt.close(ax.figure)

    ax = plot_param_importances(study)
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "optuna_importances.png", dpi=120)
    plt.close(ax.figure)


def evaluate_tuned_winner(
    checkpoint_path: Path | str,
    train_cfg: TrainConfig,
    threshold: float,
    reports_dir: Path | str,
    tag: str = "tuned",
) -> dict:
    """Scores the tuned winner on test exactly once, through Sprint 2's
    `score_split`/`compute_metrics`/`build_matrix`, at the **baseline's**
    F1-optimal threshold (`configs/infer.yaml`) so the two rows in
    `docs/RESULTS.md` are directly comparable. The baseline's threshold is
    reused rather than re-selected because the winner's final fit trains on
    train+val combined (see `retrain_winner_on_train_plus_val`), leaving no
    held-out validation split to select a new one from without touching test.
    """
    from safestreets.evaluation.cross_dataset import TEST_DATASETS

    model = load_checkpoint(checkpoint_path)
    reports_dir = Path(reports_dir)

    per_dataset: dict[str, dict] = {}
    for dataset, in_domain in TEST_DATASETS:
        y, s, _ = score_split(model, train_cfg.features_dir, [dataset], "test")
        per_dataset[dataset] = {"y": y, "s": s, "in_domain": in_domain}

    y_test = np.concatenate([v["y"] for v in per_dataset.values()])
    s_test = np.concatenate([v["s"] for v in per_dataset.values()])
    eval_test = compute_metrics(y_test, s_test, threshold)
    eval_test["datasets"] = [d for d, _ in TEST_DATASETS]
    eval_test["split"] = "test"

    cross_dataset = build_matrix(
        model, train_cfg.features_dir, lambda y, s: compute_metrics(y, s, threshold)
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"eval_test_{tag}.json").write_text(
        json.dumps(eval_test, indent=2, sort_keys=True) + "\n"
    )
    (reports_dir / f"cross_dataset_{tag}.json").write_text(
        json.dumps(cross_dataset, indent=2, sort_keys=True) + "\n"
    )
    return {"eval_test": eval_test, "cross_dataset": cross_dataset}
