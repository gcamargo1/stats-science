"""R libraries.

Library location (lib_loc) pointed to the location that RStudio R installs libraries.
"""

from rpy2 import robjects
from rpy2.robjects.packages import importr

from stats_science.constants import lib_loc

rprint = robjects.globalenv.find("print")
utils = importr("utils", lib_loc=lib_loc)
stats = importr("stats", lib_loc=lib_loc)
base = importr("base", lib_loc=lib_loc)
datasets = importr("datasets", lib_loc=lib_loc)
lme4 = importr("lme4", lib_loc=lib_loc)
nlme = importr("nlme", lib_loc=lib_loc)
emmeans = importr("emmeans", lib_loc=lib_loc)
lmertest = importr("lmerTest", lib_loc=lib_loc)
multcomp = importr("multcomp", lib_loc=lib_loc)
graphics = importr("graphics", lib_loc=lib_loc)
