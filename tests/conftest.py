import pandas as pd
import pytest
from rpy2.robjects.packages import data, importr

from stats_science.constants import lib_loc, stats_science_proj_path
from stats_science.models import lmer


@pytest.fixture(scope="session")
def mtcars():
    datasets = importr("datasets", lib_loc=lib_loc)
    return data(datasets).fetch("mtcars")["mtcars"]


@pytest.fixture(scope="session")
def sleepstudy():
    sleepstudy = pd.read_csv(stats_science_proj_path / "tests/data/sleepstudy.csv")
    sleepstudy.loc[:, "Days"] = sleepstudy["Days"].astype(str) + "_"
    sleepstudy.loc[:, "Subject"] = sleepstudy["Subject"].astype(str) + "_"
    return sleepstudy


@pytest.fixture(scope="session")
def crop_data():
    crop_data = pd.read_csv(stats_science_proj_path / "tests/data/crop_data.csv")
    crop_data.loc[:, "fertilizer"] = crop_data["fertilizer"].astype(str) + "_"
    crop_data.loc[:, "density"] = crop_data["density"].astype(str) + "_"
    crop_data.loc[:, "block"] = crop_data["block"].astype(str) + "_"
    return crop_data


@pytest.fixture(scope="session")
def crop_data_with_outlier():
    crop_data_with_outlier = pd.read_csv(
        stats_science_proj_path / "tests/data/crop_data_with_outlier.csv"
    )
    crop_data_with_outlier.loc[:, "fertilizer"] = (
        crop_data_with_outlier["fertilizer"].astype(str) + "_"
    )
    crop_data_with_outlier.loc[:, "density"] = (
        crop_data_with_outlier["density"].astype(str) + "_"
    )
    crop_data_with_outlier.loc[:, "block"] = (
        crop_data_with_outlier["block"].astype(str) + "_"
    )
    return crop_data_with_outlier


@pytest.fixture(scope="session")
def crop_lmer_model(crop_data):
    formula = "yield ~ fertilizer * density + (1|block)"
    crop_lmer_model, _, _ = lmer(formula=formula, df=crop_data)
    return crop_lmer_model


@pytest.fixture(scope="session")
def crop_lmer_model_variance_components(crop_data):
    formula = (
        "yield ~ (1|fertilizer) + (1|density) + (1|density:fertilizer) + (1|block)"
    )
    crop_lmer_model_variance_components, _, _ = lmer(formula=formula, df=crop_data)
    return crop_lmer_model_variance_components
