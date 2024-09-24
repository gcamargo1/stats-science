from stats_science.constants import stats_science_proj_path
from stats_science.plot import ggplot_scatter


def test_ggplot_scatter(mtcars):
    filename = str(stats_science_proj_path / "tests/trash/ggplot_plot.png")
    ggplot_scatter(rdf=mtcars, x="wt", y="mpg", col="factor(cyl)", filename=filename)
