"""Shared helpers: interval coverage, history features, evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fixed a priori. They are never tuned on evaluation folds, so there is no
# hyperparameter-tuning leakage to correct for.
RF_PARAMS = dict(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=1)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, random_state=42,
)


def overlap_minutes(starts: np.ndarray, ends: np.ndarray, lo: float, hi: float) -> float:
    """Minutes of [starts, ends) intervals (seconds) inside [lo, hi)."""
    if len(starts) == 0 or hi <= lo:
        return 0.0
    s = np.clip(starts, lo, hi)
    e = np.clip(ends, lo, hi)
    return float(np.clip(e - s, 0, None).sum() / 60.0)


def add_history_features(frame: pd.DataFrame, group: str, order: str, target: str, prefix: str = "hist") -> pd.DataFrame:
    """Personal history of the target from strictly earlier rows of the same participant.

    Rows must already be restricted to nights whose target was known before the
    current night's cutoff (true for any earlier night). ``shift(1)`` excludes
    the current night, so the current target can never enter its own features.
    """
    frame = frame.sort_values([group, order]).copy()
    shifted = frame.groupby(group)[target].shift(1)
    grouped = shifted.groupby(frame[group])
    frame[f"{prefix}_mean"] = grouped.transform(lambda s: s.expanding().mean())
    frame[f"{prefix}_mean_last7"] = grouped.transform(lambda s: s.rolling(7, min_periods=1).mean())
    frame[f"{prefix}_count"] = grouped.transform(lambda s: s.expanding().count())
    return frame


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def make_models() -> dict:
    import xgboost as xgb

    return {
        "random_forest": lambda: RandomForestRegressor(**RF_PARAMS),
        "xgboost": lambda: xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", **XGB_PARAMS),
    }


@dataclass
class TaskResult:
    task: str
    dataset: str
    target: str
    validation: str
    n_participants: int
    n_nights: int
    metrics: dict = field(default_factory=dict)  # name -> metrics dict
    paired_vs_best_baseline: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str, carry_col: str | None, hist_col: str | None) -> dict:
    """Simple forecasts available at the cutoff, fitted on training rows only."""
    mean = float(train[target].mean())
    preds = {"baseline_training_mean": np.full(len(test), mean)}
    if hist_col and hist_col in test:
        preds["baseline_personal_history_mean"] = test[hist_col].fillna(mean).to_numpy()
    if carry_col and carry_col in test:
        preds["baseline_previous_night"] = test[carry_col].fillna(mean).to_numpy()
    return preds


def participant_bootstrap_mae_difference(
    frame: pd.DataFrame, group: str, target: str, model_col: str, baseline_col: str, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Paired bootstrap over participants of MAE(model) - MAE(baseline).

    Negative means the model has lower error. Resampling participants, not
    nights, respects the dependence between nights of the same person.
    """
    rng = np.random.default_rng(seed)
    err = frame.assign(
        m=(frame[model_col] - frame[target]).abs(),
        b=(frame[baseline_col] - frame[target]).abs(),
    ).groupby(group)[["m", "b"]].agg(["sum", "count"])
    m_sum, b_sum, cnt = err[("m", "sum")].to_numpy(), err[("b", "sum")].to_numpy(), err[("m", "count")].to_numpy()
    ids = np.arange(len(cnt))
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        n = cnt[pick].sum()
        diffs.append((m_sum[pick].sum() - b_sum[pick].sum()) / n)
    diffs = np.asarray(diffs)
    observed = (m_sum.sum() - b_sum.sum()) / cnt.sum()
    return {
        "model": model_col,
        "baseline": baseline_col,
        "mae_difference": float(observed),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "share_bootstrap_model_better": float((diffs < 0).mean()),
    }


def evaluate_grouped(
    frame: pd.DataFrame, features: list[str], target: str, group: str, splits, carry_col: str | None, hist_col: str | None,
) -> pd.DataFrame:
    """Out-of-fold predictions for every model and baseline.

    ``splits`` yields (train_index, test_index) positional arrays. Models are
    fitted on training rows only; nothing about the test rows (including their
    feature distributions) is used before prediction.
    """
    frame = frame.reset_index(drop=True)
    out = frame[[group, target]].copy()
    out["fold"] = -1
    models = make_models()
    for fold, (tr, te) in enumerate(splits):
        train, test = frame.iloc[tr], frame.iloc[te]
        assert not set(train[group]) & set(test[group]) or fold_is_chronological(train, test), "participant leakage"
        out.loc[te, "fold"] = fold
        for name, pred in baseline_predictions(train, test, target, carry_col, hist_col).items():
            out.loc[te, name] = pred
        for name, factory in models.items():
            model = factory()
            model.fit(train[features].to_numpy(dtype=float), train[target].to_numpy(dtype=float))
            out.loc[te, name] = model.predict(test[features].to_numpy(dtype=float))
        if hist_col:
            # Residual variant: forecast the deviation from the person's own
            # history (training-mean fallback), so participants whose usual
            # level lies outside the training range are not pulled to it.
            mean = float(train[target].mean())
            anchor_tr = train[hist_col].fillna(mean).to_numpy(dtype=float)
            anchor_te = test[hist_col].fillna(mean).to_numpy(dtype=float)
            for name, factory in models.items():
                model = factory()
                model.fit(train[features].to_numpy(dtype=float), train[target].to_numpy(dtype=float) - anchor_tr)
                out.loc[te, f"{name}_residual"] = anchor_te + model.predict(test[features].to_numpy(dtype=float))
    return out[out["fold"] >= 0]


def fold_is_chronological(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """Chronological splits may share participants; they must not share time."""
    if "night_order" not in train or "night_order" not in test:
        return False
    return train["night_order"].max() < test["night_order"].min()


MODEL_COLUMNS = ("random_forest", "xgboost", "random_forest_residual", "xgboost_residual")


def summarise(oof: pd.DataFrame, target: str, group: str, task: str, dataset: str, validation: str, notes=None) -> TaskResult:
    pred_cols = [c for c in oof.columns if c.startswith("baseline_") or c in MODEL_COLUMNS]
    result = TaskResult(
        task=task, dataset=dataset, target=target, validation=validation,
        n_participants=int(oof[group].nunique()), n_nights=int(len(oof)), notes=list(notes or []),
    )
    for col in pred_cols:
        result.metrics[col] = regression_metrics(oof[target], oof[col])
    baselines = [c for c in pred_cols if c.startswith("baseline_")]
    best = min(baselines, key=lambda c: result.metrics[c]["mae"])
    for model in MODEL_COLUMNS:
        if model in oof:
            result.paired_vs_best_baseline[model] = participant_bootstrap_mae_difference(oof, group, target, model, best)
    return result
