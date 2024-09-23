"""Constants."""

import pathlib

stats_science_proj_path: pathlib.Path = pathlib.Path(
    __file__
).parent.parent.parent.absolute()

lib_loc = "/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/library"

chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_|-+=.,></?' '~!@#$%^&*()"
r_models_types = ["lmerModLmerTest", "merModLmerTest", "lm", "lme", "lmer"]
