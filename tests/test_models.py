import pandas as pd

from stats_science.constants import stats_science_proj_path
from stats_science.models import (
    anova_fixed_effects_model,
    lmer,
    mean_separation,
    repeated_measures,
    variance_components_analysis,
)


class TestOneWayAnova:
    def test_one_way_anova(self, crop_data):
        formula = "yield ~ fertilizer"
        factors = ["fertilizer"]
        mod, anova_table, df = anova_fixed_effects_model(
            formula=formula, factors=factors, df=crop_data
        )

    def test_one_way_anova_with_outlier(self, crop_data_with_outlier):
        formula = "yield ~ fertilizer"
        factors = ["fertilizer"]
        outlier_path = stats_science_proj_path / "tests/trash/anova_outliers.csv"
        outlier_path = outlier_path.__str__()
        mod, anova_table, df = anova_fixed_effects_model(
            formula=formula,
            factors=factors,
            df=crop_data_with_outlier,
            outlier_path=outlier_path,
            remove_outliers=True,
        )


class TestLmer:
    def test_lmer(self, sleepstudy):
        formula = "Reaction ~ Days + (1 | Subject)"
        df = sleepstudy
        mod, anova_table, df = lmer(formula=formula, df=df)


def test_repeated_measures(sleepstudy):
    formula_fix = "Reaction ~ Subject"
    formula_rand = "~1|Days/Subject"
    mod, anova_table, df = repeated_measures(
        formula_fix=formula_fix,
        formula_rand=formula_rand,
        df=sleepstudy,
        repeated_measure="Days",
    )


def test_mean_separation(crop_lmer_model):
    lsmeans_specs = ["density", "fertilizer"]
    pairwisecomp = mean_separation(mod=crop_lmer_model, lsmeans_specs=lsmeans_specs)
    assert isinstance(pairwisecomp, pd.DataFrame)


# This test causes issues with pytest due to some sort of global variable.
# def test_get_lsd(crop_lmer_model):
#     lsmeans_obj = emmeans.emmeans(object=crop_lmer_model, specs=["fertilizer"])
#     lsd = get_lsd(lsmeans_obj=lsmeans_obj)
#     assert lsd > 0
def test_variance_components_analysis(crop_lmer_model_variance_components):
    variances = variance_components_analysis(mod=crop_lmer_model_variance_components)
    assert round(variances.iloc[2]["variance_percent"], 1) == 0.2
