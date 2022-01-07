import os
from typing import Union

from matplotlib import figure, axes, pyplot as plt, rcParams, cycler, legend


def set_plot_params() -> None:
    """
        Sets the default plotting params for matplotlib
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 22
    rcParams['axes.linewidth'] = 1.1
    rcParams['axes.labelpad'] = 4.0
    plot_color_cycle = cycler(
            'color',
            ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b',
             'e377c2', '7f7f7f', 'bcbd22', '17becf', ]
    )
    rcParams['axes.prop_cycle'] = plot_color_cycle
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0
    rcParams.update(
            {
                "figure.subplot.hspace": 0,
                "figure.subplot.left": 0.11,
                "figure.subplot.right": 0.946,
                "figure.subplot.bottom": 0.156,
                "figure.subplot.top": 0.965,
                "xtick.major.size": 4,
                "xtick.minor.size": 2.5,
                "xtick.major.width": 1.1,
                "xtick.minor.width": 1.1,
                "xtick.major.pad": 5,
                "xtick.minor.visible": True,
                "xtick.direction": 'in',
                "xtick.top": True,
                "ytick.major.size": 4,
                "ytick.minor.size": 2.5,
                "ytick.major.width": 1.1,
                "ytick.minor.width": 1.1,
                "ytick.major.pad": 5,
                "ytick.minor.visible": True,
                "ytick.direction": 'in',
                "ytick.right": True,
                "lines.markersize": 10,
                "lines.markeredgewidth": 0.8,
            }
    )


def create_fig(
        nrows: int = 1,
        ncols: int = 1,
        sharex='all',
        sharey='all',
        figsize=None,
        subplot_kw=None,
        gridspec_kw=None,
        top_ticks=False,
        **fig_kw
) -> tuple[figure.Figure, Union[axes.Axes, list[axes.Axes]]]:
    """ Creates the figure, axes for the plots and set the style of the plot

    Parameters
    ----------
    nrows : int
        number of rows
    ncols :
        number of columns
    sharex
    sharey
    figsize
    subplot_kw
    gridspec_kw
    top_ticks
    fig_kw

    Returns
    -------
    the figure and list of created axes
    """
    set_plot_params()

    if figsize is None:
        figsize = (6.4 * ncols, 4.8 * nrows)
    if gridspec_kw is None:
        gridspec_kw = {"hspace": 0}
    elif gridspec_kw.get("hspace") is None:
        gridspec_kw["hspace"] = 0

    fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=gridspec_kw,
            subplot_kw=subplot_kw,
            **fig_kw
    )

    if top_ticks:
        axs[0].xaxis.set_tick_params(which="both", labeltop=True)

    return fig, axs


def save_fig(
        path: str = '', fig: figure.Figure = None, show: bool = False, **kwargs
) -> None:
    """ Saves the current figure at path

    Parameters
    ----------
    path : str
        path to save the figure
    fig : matplotlib.figure.Figure
        the figure to save
    show : bool
        show figure, no saving, False: save and show figure
    **kwargs
        any Keywords for Figure.savefig
    """
    if fig is None:
        fig = plt.gcf()
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path, bbox_inches='tight', **kwargs)
    fig.canvas.draw_idle()
    if show:
        plt.show()
    plt.close(fig)


def plot_legend(
        ax: axes.Axes = None,
        loc='upper left',
        fontsize='xx-small',
        frameon=False,
        markerscale=2,
        handletextpad=0.1,
        mode='expand',
        **kwargs
) -> legend.Legend:
    """ Adds legend to an axes

    Parameters
    ----------
    ax
    loc
    fontsize
    frameon
    markerscale
    handletextpad
    mode
    kwargs

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()

    leg = ax.legend(
            loc=loc,
            fontsize=fontsize,
            frameon=True,
            framealpha=1,
            edgecolor='white',
            markerscale=markerscale,
            handletextpad=handletextpad,
            mode=None,
            borderpad=0.0,
            **kwargs
    )
    return leg
