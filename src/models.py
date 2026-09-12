"""
Model factory. Defines every candidate model as an sklearn-compatible
Pipeline(preprocessor -> classifier) so they can all be trained, evaluated,
and served through one uniform interface.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.feature_engineering import build_preprocessor

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_model_registry_specs(scale_pos_weight: float = 1.0) -> dict:
    """
    Returns {model_name: sklearn Pipeline} for every model we want to train
    and compare. scale_pos_weight corrects for class imbalance (defaults are
    ~17% positive class) for the boosting models; LogReg/RF use
    class_weight="balanced" instead.
    """
    specs = {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    if HAS_XGB:
        specs["xgboost"] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    if HAS_LGBM:
        specs["lightgbm"] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LGBMClassifier(
                n_estimators=400,
                max_depth=-1,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )),
        ])

    if HAS_CATBOOST:
        specs["catboost"] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                verbose=False,
            )),
        ])

    return specs
