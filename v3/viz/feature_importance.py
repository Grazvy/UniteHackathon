from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def save_feature_importance_plots(clf, reg, feature_cols, out_dir: Path):
    """Save classifier/regressor/combined importance plots."""
    clf_imp = pd.Series(clf.feature_importances_, index=feature_cols, name="clf_importance")
    reg_imp = pd.Series(reg.feature_importances_, index=feature_cols, name="reg_importance")

    imp_df = pd.concat([clf_imp, reg_imp], axis=1).fillna(0)
    imp_df["clf_norm"] = imp_df["clf_importance"] / max(imp_df["clf_importance"].sum(), 1)
    imp_df["reg_norm"] = imp_df["reg_importance"] / max(imp_df["reg_importance"].sum(), 1)
    imp_df["combined_importance"] = 0.5 * imp_df["clf_norm"] + 0.5 * imp_df["reg_norm"]

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    imp_df.sort_values("clf_importance", ascending=True)["clf_importance"].plot.barh(
        ax=axes[0], color="#3b82f6", edgecolor="white"
    )
    axes[0].set_title("Classifier Importance\\nP(purchase)")
    axes[0].set_xlabel("Importance (split count)")

    imp_df.sort_values("reg_importance", ascending=True)["reg_importance"].plot.barh(
        ax=axes[1], color="#10b981", edgecolor="white"
    )
    axes[1].set_title("Regressor Importance\\nSpend if purchased")
    axes[1].set_xlabel("Importance (split count)")

    imp_df.sort_values("combined_importance", ascending=True)["combined_importance"].plot.barh(
        ax=axes[2], color="#f59e0b", edgecolor="white"
    )
    axes[2].set_title("Combined Predictive Importance")
    axes[2].set_xlabel("Normalized importance")

    plt.suptitle("Main Predictor Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_file = out_dir / "main_predictor_features.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    return out_file, imp_df
