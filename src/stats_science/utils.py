"""Utilities functions."""

import os.path
import re
import shutil
from pathlib import Path

import pandas as pd
import rpy2
from rpy2 import robjects
from rpy2.robjects import RS4, pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import shapiro

from stats_science.constants import chars, r_models_types
from stats_science.r_modules import base, stats, utils


def df2rdf(df: pd.DataFrame) -> robjects.vectors.DataFrame:
    """Convert pandas dataframe to R dataframe.

    Export csv via pandas and import csv to R daframe via utils.

    Args:
        df: Pandas Dataframe

    Returns:
        R dataframe

    """
    assert isinstance(df, pd.DataFrame), "must be a pandas dataframe."
    dir_name = Path(os.path.realpath(Path.cwd())).parent
    save_path = Path(dir_name, "trash/df.csv")
    make_dir_from_fname_if_needed(save_path)
    df.to_csv(save_path, index=False)
    rdf = utils.read_csv(save_path.as_posix())
    shutil.rmtree(save_path.parent)
    return rdf


def rdf2df(rdf: robjects.vectors.DataFrame) -> pd.DataFrame:
    """R dataframe to Pandas datraframe.

    Args:
        rdf: R dataframe

    Returns:
        pandas dataframe

    """
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(rdf)
    return df


def _check_lm_args(
    formula: str,
    df: pd.DataFrame,
    factors: list[str],
) -> None:
    """Check lm arguments."""
    assert formula.count("~") == 1, "One tilde must be in the formula"
    response_str = formula.split("~")[0].strip()
    assert pd.notna(df[response_str]).to_numpy().any(), "NA values in response"
    assert df[response_str].std() > 0, f"response {response_str} must have variability"
    assert df.shape[0] > 0
    for factor in factors:
        assert isinstance(factor, str)
        assert factor in df.columns
        levels = df[factor].unique().tolist()
        min_levels = 2
        assert len(levels) >= min_levels, f"factor {factor} must have at least 2 levels"
        for level in levels:
            assert isinstance(level, str)
            assert any(
                char in level for char in chars
            ), f"level {level} in factor {factor} must have a character"
    assert df.shape[0] > len(factors), (
        f"The number of observations {df.shape[0]} must be greater than"
        f" number of factors: {len(factors)}"
    )


def _get_r_model_class(mod: RS4) -> str:
    """Return R model type."""
    return mod.rclass[0]


def add_residuals(
    df: pd.DataFrame,
    mod: RS4,
    *,
    residuals_type: str = "pearson",
    scale: bool = True,
) -> pd.DataFrame:
    """Add model residuals to dataframe."""
    assert residuals_type in {"deviance", "pearson", "working", "response", "partial"}
    assert df.shape[0] > 0
    r_mod_type = _get_r_model_class(mod)
    assert r_mod_type in r_models_types, "provide R model"
    if r_mod_type == "lm":
        scaled_residuals = rdf2df(stats.rstandard(model=mod))
    else:
        scaled_residuals = rdf2df(
            stats.residuals(mod, type=residuals_type, scale=scale),
        )
    residuals = rdf2df(stats.residuals(mod))
    assert len(scaled_residuals) == df.shape[0], "issue with scaled residue generation"
    df = df.assign(scaled_residuals=scaled_residuals)
    df = df.assign(residuals=residuals)
    return df


def resid_normality_checker(
    scaled_residuals: pd.Series,
    population_thresh: int = 40,
    alpha: float = 0.5,
) -> None:
    """Check if scaled residuals are normal, as qualifier to outlier removal."""
    if scaled_residuals.shape[0] <= population_thresh:
        print(f"Small smaple size ({population_thresh}), normality being checked.")
        _, p = shapiro(scaled_residuals)
        assert p <= alpha, "Not a normal distribution (shapiro's test)"


def has_outliers_checker(
    df: pd.DataFrame,
    mod: RS4,
    *,
    std_resid_outlier_thrsh: float,
    check_resid_normality: bool = True,
) -> tuple[bool, pd.DataFrame]:
    """Checks if dataset has outliers based on model residuals."""
    df = add_residuals(df, mod)
    if check_resid_normality:
        resid_normality_checker(scaled_residuals=df["scaled_residuals"])
    has_outliers = (
        df["scaled_residuals"].min() < -std_resid_outlier_thrsh
        or df["scaled_residuals"].max() > std_resid_outlier_thrsh
    )
    return has_outliers, df


def remove_outliers_rows(
    df: pd.DataFrame,
    std_resid_outlier_thrsh: float,
    outlier_path: str,
) -> tuple[pd.DataFrame, rpy2.robjects.vectors.DataFrame]:
    """Remove outliers from dataframe based on std threshold."""
    assert "scaled_residuals" in df.columns
    make_dir_from_fname_if_needed(fpath=outlier_path)
    outliers_df = df.loc[
        (df["scaled_residuals"] < -std_resid_outlier_thrsh)
        | (df["scaled_residuals"] > std_resid_outlier_thrsh)
    ].copy()
    bef = df.shape[0]
    df = df.loc[
        (df["scaled_residuals"] >= -std_resid_outlier_thrsh)
        & (df["scaled_residuals"] <= std_resid_outlier_thrsh)
    ].copy()
    aft = df.shape[0]
    print(f"{bef - aft} outliers removed!")
    outliers_df.to_csv(outlier_path, index=False)
    rdf = df2rdf(df=df)
    return df, rdf


def _get_factors_from_formula(formula: str):
    formula_right_side = "".join(formula.split("~")[1].split())
    re_expr = r"; |\:|\*|\n|\+|\)|\(|1|0|/|\|"
    factors = re.split(re_expr, formula_right_side)

    # Remove emply list members
    factors = [y for y in factors if y]

    # Remove repetitive factors
    factors = list(set(factors))

    for factor in factors:
        assert isinstance(factor, str)

    return factors


def _check_lmer_args(
    formula: str,
    df: pd.DataFrame,
) -> None:
    assert formula.count("~") == 1, "one tilde must be provided in formula."
    assert "|" in formula, "pipe | must be in formula."
    response_str = formula.split("~")[0].strip()
    assert not pd.isna(df[response_str]).to_numpy().any(), "NA values in response."
    assert df[response_str].std() > 0, f"response {response_str} must have variability."
    digits = re.findall(r"\d+", formula)
    assert len(digits) > 0
    for digit in digits:
        assert digit in {"0", "1"}, "Only 1 or 0 allowed in formula."
    factors = _get_factors_from_formula(formula=formula)
    for factor in factors:
        assert factor in df.columns, f"factor {factor} not in dataset."
        levels = df[factor].unique().tolist()
        min_levels = 2
        assert len(levels) >= min_levels
        for level in levels:
            assert any(
                char in level for char in chars
            ), f"level {level} in factor {factor} must have a character."
    assert df.shape[0] > len(factors)
    formula_components = re.findall(r"[\w]+", formula)
    while "1" in formula_components:
        formula_components.remove("1")
    for formula_component in formula_components:
        assert (
            formula_component in df.columns.tolist()
        ), f"formula component {formula_component} not in dataset."


def add_fitted(df: pd.DataFrame, mod: RS4) -> pd.DataFrame:
    """Extract model fitted values."""
    fitted = rdf2df(stats.fitted(mod))
    assert len(fitted) == df.shape[0], "Model and dataframe shape do not match."
    df = df.assign(fitted=fitted)
    return df


def get_lmer_anova_table(
    mod: RS4,
    anova_type: str = "III",
    anova_ddf: str = "Satterthwaite",
) -> pd.DataFrame:
    """Get ANOVA table."""
    assert anova_ddf in {
        "Kenward-Roger",
        "Satterthwaite",
        "lme4",
    }, f"anova type: {anova_ddf} not supported."
    anova_table = base.as_data_frame(
        stats.anova(object=mod, type=anova_type, ddf=anova_ddf),
    )
    row_names = list(rdf2df(base.row_names(anova_table)))
    anova_table = rdf2df(anova_table)
    anova_table.loc[:, "factors"] = row_names
    anova_table = anova_table.reindex(
        [
            "factors",
            "Sum Sq",
            "Mean Sq",
            "NumDF",
            "DenDF",
            "F value",
            "Pr(>F)",
        ],
        axis=1,
    )
    anova_table = anova_table.rename(columns={"Pr(>F)": "p_value"})
    return anova_table


def _convert_lsmeans_at_dict2str(lsmeans_at_dict: dict) -> str:
    keys = lsmeans_at_dict.keys()
    assert len(keys) > 0
    lsmeans_at_str = ""
    for i, key in enumerate(keys):
        vals = lsmeans_at_dict[key]
        assert len(vals) > 0, f"No values in dictionary for key {key}"
        if i == 0:
            lsmeans_at_str = lsmeans_at_str.join((key, "=c", str(tuple(vals))))
            if len(vals) == 1:
                lsmeans_at_str = lsmeans_at_str.replace(",", "")
        else:
            str_to_append = key + "=c" + str(tuple(vals))
            if len(vals) == 1:
                str_to_append = str_to_append.replace(",", "")
            lsmeans_at_str = lsmeans_at_str + "," + str_to_append
    lsmeans_at_str = f"list({lsmeans_at_str})"
    return lsmeans_at_str


def add_underscore_to_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add underscore to series in dataframe."""
    df[col] = df[col].astype(str) + "_"
    return df


def make_dir_from_fname_if_needed(fpath: str | Path) -> None:
    """Create directory if needed when saving files."""
    if isinstance(fpath, Path):
        fpath = str(fpath)
    dir_path = get_dir_path_from_fpath(fpath=fpath)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def get_dir_path_from_fpath(fpath: str) -> str:
    """Get the directory path from a given file path.

    Args:
        fpath (str): The path of the file.

    Returns:
        str: The directory path of the file.

    Raises:
        ValueError: If the provided file path is invalid or empty.
    """
    if not fpath:
        raise ValueError("Invalid or empty file path.")

    # Use os.path.dirname to get the directory path from the file path.
    directory_path = os.path.dirname(fpath)
    return directory_path
