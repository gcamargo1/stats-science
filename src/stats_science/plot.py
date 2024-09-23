"""Plot functions."""

import rpy2
from rpy2.robjects.lib import ggplot2  # rpy2 native ggplot2


def ggplot_scatter(
    rdf: rpy2.robjects.vectors.DataFrame,
    x: str,
    y: str,
    col: str,
    filename: str,
) -> None:
    """Generate a basic ggplot scatterplot chart with x, y, and color attributes.

    Args:
        rdf: R dataframe
        x: x variable
        y: y variable
        col: color variable
        filename: saving variable

    References:
        https://rpy2.github.io/doc/latest/html/introduction.html#examples

    """
    chart = (
        ggplot2.ggplot(data=rdf) + ggplot2.aes(x=x, y=y, col=col) + ggplot2.geom_point()
    )
    chart.save(filename=filename)
