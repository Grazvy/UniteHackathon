import numpy as np
import lightgbm as lgb
from v3.config import LGBM_PARAMS_CLF, LGBM_PARAMS_REG, SAVINGS_RATE, TOTAL_FEE


def train_two_models(train_feats, feature_cols):
    """Train classifier (buy/no-buy) and regressor (spend if buy)."""
    X = train_feats[feature_cols].fillna(0)

    clf = lgb.LGBMClassifier(**LGBM_PARAMS_CLF)
    clf.fit(X, train_feats["buy_label"])

    pos_mask = train_feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**LGBM_PARAMS_REG)
    reg.fit(X.loc[pos_mask], train_feats.loc[pos_mask, "log_spend_label"])

    return clf, reg


def infer_expected_spend(feats, clf, reg, feature_cols, expected_spend_scale):
    """Two-model inference with economic decision fields."""
    X = feats[feature_cols].fillna(0)
    p_buy = clf.predict_proba(X)[:, 1]
    spend_if_buy = np.expm1(reg.predict(X)).clip(min=0)

    out = feats.copy()
    out["p_buy"] = p_buy
    out["spend_if_buy"] = spend_if_buy
    out["expected_spend"] = p_buy * spend_if_buy * expected_spend_scale
    out["expected_savings"] = out["expected_spend"] * SAVINGS_RATE
    out["is_core"] = (out["expected_savings"] > TOTAL_FEE).astype(int)
    return out
