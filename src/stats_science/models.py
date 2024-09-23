"""Main module with statistical models."""

import re

import pandas as pd
from pandas import DataFrame
from rpy2 import robjects
from rpy2.robjects import RS4, Formula, ListVector, globalenv

from stats_science.constants import chars
from stats_science.r_modules import (
    base,
    emmeans,
    graphics,
    lme4,
    lmertest,
    multcomp,
    nlme,
    stats,
)
from stats_science.utils import (
    _check_lm_args,
    _check_lmer_args,
    _convert_lsmeans_at_dict2str,
    _get_r_model_class,
    add_fitted,
    add_residuals,
    add_underscore_to_df,
    df2rdf,
    get_lmer_anova_table,
    has_outliers_checker,
    rdf2df,
    remove_outliers_rows,
    resid_normality_checker,
)


def anova_fixed_effects_model(
    *,
    formula: str,
    df: DataFrame,
    factors: list[str],
    remove_outliers: bool = False,
    outlier_path: str = "",
    std_resid_outlier_thrsh: float = 3,
    check_resid_normality: bool = True,
) -> tuple[ListVector, DataFrame, DataFrame]:
    """ANOVA fixed effects model.

    Args:
        formula: e.g., y ~ factor1 * factor2
        df: dataframe
        factors: factors in the model e.g., [factor1 * factor2]
        remove_outliers: model based outlier removal
        outlier_path: save path for outlier removal
        std_resid_outlier_thrsh: standard residual outlier removal threshold
        check_resid_normality: flag to check residuals normality

    Returns:
        R model
        ANOVA table
        model dataframe

    References:
        https://online.stat.psu.edu/stat502_fa21/lesson/4/4.5
    """
    for factor in factors:
        df = add_underscore_to_df(df=df, col=factor)
    _check_lm_args(formula=formula, df=df, factors=factors)
    if remove_outliers:
        assert outlier_path, "Please provide valid outlier path."
    rdf = df2rdf(df=df)
    formula = Formula(formula=formula)
    mod = stats.lm(formula, data=rdf)
    has_outliers, df = has_outliers_checker(
        df=df,
        mod=mod,
        std_resid_outlier_thrsh=std_resid_outlier_thrsh,
        check_resid_normality=check_resid_normality,
    )
    if has_outliers:
        df, rdf = remove_outliers_rows(
            df=df,
            std_resid_outlier_thrsh=std_resid_outlier_thrsh,
            outlier_path=outlier_path,
        )
        mod = stats.lm(formula=formula, data=rdf)
    anova_table = base.as_data_frame(stats.anova(mod))
    row_names = list(rdf2df(base.row_names(anova_table)))
    anova_table = rdf2df(anova_table)
    anova_table.loc[:, "factors"] = row_names
    anova_table = anova_table.reindex(
        ["factors", "Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)"],
        axis=1,
    )
    return mod, anova_table, df


def lmer(
    *,
    formula: str,
    df: pd.DataFrame,
    remove_outliers: bool = False,
    outlier_path: str = "",
    std_resid_outlier_thrsh: float = 3,
    check_resid_normality: bool = True,
) -> tuple[ListVector, DataFrame, DataFrame]:
    """Fit a Mixed effect ANOVA model using the lmer function.

    Args:
        formula: model formula
        df: dataframe
        remove_outliers: flag to remove outliers after fitting
        outlier_path: fpath to save outliers csv file.
        std_resid_outlier_thrsh: standard resdiual outlier threshold.
        check_resid_normality: bool to check residuals normality.

    Returns:
        R model
        anova table
        model dataframe
    """
    # Check inputs
    _check_lmer_args(
        formula=formula,
        df=df,
    )

    # Setup R variables
    rdf = df2rdf(df=df)
    formula = Formula(formula)

    # Run model
    mod = lmertest.lmer(formula=formula, data=rdf)

    # Outliers check and screening
    has_outliers, df = has_outliers_checker(
        df=df,
        mod=mod,
        std_resid_outlier_thrsh=std_resid_outlier_thrsh,
        check_resid_normality=check_resid_normality,
    )
    if remove_outliers and has_outliers:
        df, rdf = remove_outliers_rows(
            df=df,
            std_resid_outlier_thrsh=std_resid_outlier_thrsh,
            outlier_path=outlier_path,
        )
        mod = lmertest.lmer(formula=formula, data=rdf)
        df = add_residuals(df=df, mod=mod)
        df = add_fitted(df=df, mod=mod)
    anova_table = get_lmer_anova_table(mod=mod)
    return mod, anova_table, df


def repeated_measures(
    *,
    formula_fix: str,
    formula_rand: str,
    df: pd.DataFrame,
    remove_outliers: bool = False,
    outlier_path: str = "",
    repeated_measure: str = "datc",
    std_resid_outlier_thrsh: float = 3,
    check_resid_normality: bool = True,
) -> tuple[RS4, pd.DataFrame, pd.DataFrame]:
    """Repeated measures anova model."""
    assert formula_fix.count("~") == 1
    assert formula_rand.count("~") == 1
    assert "|" in formula_rand
    response_str = formula_fix.split("~")[0].strip()
    assert response_str in df.columns
    assert df[response_str].isna().sum() == 0
    fix_rand_form = formula_fix + formula_rand
    digits = re.findall(r"\d+", fix_rand_form)
    for digit in digits:
        assert digit in {"0", "1"}
    fixed_factors = re.split("[A-Za-z]+", formula_fix.split("~")[1].strip(" "))
    random_factors = list(set(re.split("[^A-Za-z]+", formula_rand)))
    random_factors = [y for y in random_factors if y]
    factors = list(set(fixed_factors + random_factors))
    if repeated_measure in factors:
        factors.remove(repeated_measure)
    for factor in factors:
        if not factor:
            continue
        assert factor in df.columns
        levels = df[factor].unique().tolist()
        min_levels = 2
        assert len(levels) >= min_levels
        for level in levels:
            assert any(char in level for char in chars)
    assert df.shape[0] > len(factors)
    for rep_measure in df[repeated_measure].unique():
        std = df[df[repeated_measure] == rep_measure][response_str].std()
        mean = df[df[repeated_measure] == rep_measure][response_str].mean()
        cv = std / mean if mean != 0 else 0
        cv_min = 0.03
        if cv < cv_min:
            print(
                f"repeated mearue cv close to 0 ({cv}), consider removing"
                f" {rep_measure} from model",
            )
    rdf = df2rdf(df=df)
    fixed_formula = Formula(formula_fix)
    rand_formula = Formula(formula_rand)
    mod = nlme.lme(
        fixed=fixed_formula,
        random=rand_formula,
        method="REML",
        data=rdf,
        control=nlme.lmeControl(opt="optim"),
    )

    if remove_outliers:
        df = add_residuals(df=df, mod=mod)
        df = add_fitted(df=df, mod=mod)
        resid_normality_checker(scaled_residuals=df["scaled_residuals"])
        has_outliers, df = has_outliers_checker(
            df=df,
            mod=mod,
            std_resid_outlier_thrsh=std_resid_outlier_thrsh,
            check_resid_normality=check_resid_normality,
        )
        if has_outliers:
            df, rdf = remove_outliers_rows(
                df=df,
                std_resid_outlier_thrsh=std_resid_outlier_thrsh,
                outlier_path=outlier_path,
            )
            mod = nlme.lme(
                fixed=fixed_formula,
                random=rand_formula,
                method="REML",
                data=rdf,
                control=nlme.lmeControl(opt="optim"),
            )

    anova_table = base.as_data_frame(stats.anova(mod))
    row_names = list(rdf2df(base.row_names(anova_table)))
    anova_table = rdf2df(anova_table)
    anova_table.loc[:, "factors"] = row_names
    pseudo_rep_ratio_threshold = 0.95
    assert (
        anova_table["denDF"].max() / df.shape[0] < pseudo_rep_ratio_threshold
    ), "potential pseudo-replication (check ranndom effects)"
    anova_table = anova_table.reindex(
        ["factors", "numDF", "denDF", "F-value", "p-value"],
        axis=1,
    )
    return mod, anova_table, df


def mean_separation(
    *,
    mod: RS4,
    lsmeans_specs: list[str],
    lsmeans_by: bool | list[str] = False,
    lsmeans_at_dict: bool | dict = False,
    alpha: float = 0.05,
    p_adjust: str = "Tukey",
) -> pd.DataFrame:
    """Least Squares Means separation with compact letters displays (CLD).

    Args:
        mod: R model.
        lsmeans_specs: lsmeans specification
        lsmeans_by: lsmeans specification
        lsmeans_at_dict: lsmeans specification
        alpha: significance level.
        p_adjust: p value adjustment.

    References:
        https://schmidtpaul.github.io/dsfair_quarto/ch/summaryarticles/compactletterdisplay.html

    """
    globalenv["mod"] = mod
    model_class = _get_r_model_class(mod=mod)
    if model_class == "lme":
        model_elements = rdf2df(
            base.attributes(mod.rx2("terms")).rx2("term.labels"),
        ).tolist()
    elif model_class == "lm":
        model_elements = False
    else:
        model_elements = rdf2df(base.names(base.attributes(mod).rx2("frame")))
    if model_elements:
        for spec in lsmeans_specs:
            assert (
                spec in model_elements
            ), f"lsmeans_specs term {spec} not in fixed formula."
            if lsmeans_by:
                for lsmeans_by_i in lsmeans_by:
                    assert (
                        lsmeans_by_i in model_elements
                    ), f"lsmeans_by term {lsmeans_by_i} not in fixed formula."
    if lsmeans_at_dict:
        keys = lsmeans_at_dict.keys()
        for key in keys:
            assert key in lsmeans_specs
            vals = lsmeans_at_dict[key]
            if model_class == "lm":
                factor_levels = rdf2df(mod.rx2("model"))[key].unique().tolist()
            else:
                factor_levels = (
                    rdf2df(robjects.r("""mod@frame"""))[key].unique().tolist()
                )
            for val in vals:
                assert (
                    val in factor_levels
                ), f"value {val} in lsmean_at_dict not in model."
    lsmeans_specs_str = "".join(("'", "','".join(lsmeans_specs), "'"))
    lsmeans_specs = robjects.sequence_to_vector(lsmeans_specs)
    if lsmeans_by:
        lsmeans_by_str = "".join(("'", "','".join(lsmeans_by), "'"))
        lsmeans_by = robjects.sequence_to_vector(lsmeans_by)
    else:
        lsmeans_by_str = "NULL"
    if lsmeans_at_dict:
        lsmeans_at_str = _convert_lsmeans_at_dict2str(lsmeans_at_dict)
        lsmean = robjects.r(
            f"""emmeans::emmeans(object=mod, specs=c({lsmeans_specs_str}),
             by=c({lsmeans_by_str}), at={lsmeans_at_str}""",
        )
    elif lsmeans_by:
        lsmean = emmeans.emmeans(object=mod, specs=lsmeans_specs, by=lsmeans_by)
    else:
        lsmean = emmeans.emmeans(object=mod, specs=lsmeans_specs)
    pairwisecomp = multcomp.cld(
        lsmean,
        alpha=alpha,
        Letters="abcdefghijklmnoprstuvwxyz",
        adjust=p_adjust,
        reversed=True,
    )
    pairwisecomp = rdf2df(pairwisecomp)
    assert pairwisecomp.shape[0] > 0, "Empty pairwisecom dataframe."
    pairwisecomp = pairwisecomp.rename(columns={".group": "group"})
    pairwisecomp.loc[:, "group"] = pairwisecomp["group"].str.replace(" ", "")
    min_threshold = -1e6
    max_threshold = 1e6
    pairwisecomp.loc[
        (pairwisecomp["emmean"] > min_threshold)
        & (pairwisecomp["emmean"] < max_threshold),
        "emmean",
    ] = 0
    pairwisecomp = pairwisecomp.rename(columns={"emmean": "lsmean"})
    return pairwisecomp


def get_lsd(lsmeans_obj: emmeans.emmeans, alpha: float = 0.05) -> float:
    """Get least squared differences."""
    pairs = base.data_frame(base.summary(graphics.pairs(lsmeans_obj)))
    lsd = (
        rdf2df(stats.qt((1 - alpha / 2), pairs.rx2("df")))
        * rdf2df(pairs.rx2("SE")).mean()
    )[0]
    assert lsd >= 0
    return lsd


def variance_components_analysis(mod: RS4) -> pd.DataFrame:
    """Return variance components of a given lmer model."""
    assert _get_r_model_class(mod) == "lmerModLmerTest"
    variance_comps = rdf2df(base.data_frame(lme4.VarCorr_merMod(x=mod)))
    terms = variance_comps["grp"].unique()
    variances = []
    for term in terms:
        variance = variance_comps[variance_comps["grp"] == term].iloc[0]["vcov"]
        variances.append([term, variance])
    variances = pd.DataFrame(variances, columns=["term", "variance"])
    variances["variance_percent"] = variances["variance"] / variances["variance"].sum()
    return variances
