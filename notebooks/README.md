## Figure notebooks

This directory contains the notebooks used to create various figures. _Note_: If you want to ensure your figure aesthetics are the same, either manually configure the `rc` variable in the first cell or place these lines in your `~/.config/matplotlib/matplotlibrc` file:

    lines.linewidth: 2
    lines.markeredgecolor: k
    lines.markeredgewidth: 1.3
    boxplot.whiskers: 1.5
    boxplot.patchartist: True
    boxplot.showmeans: True
    boxplot.meanline: False
    boxplot.boxprops.linewidth: 1.5
    boxplot.whiskerprops.linewidth: 1.5
    boxplot.medianprops.linewidth: 1.5
    boxplot.medianprops.color: None
    boxplot.meanprops.marker: ^
    boxplot.meanprops.markerfacecolor: w
    boxplot.meanprops.markeredgecolor: k
    axes.grid: True
    axes.linewidth: 1.5
    xtick.bottom: True
    xtick.labelbottom: True
    xtick.major.width: 1.5
    xtick.minor.width: 1 
    ytick.left: True
    ytick.labelleft: True
    ytick.major.width: 1.5
    ytick.minor.width: 1 
    grid.linestyle: --
    hist.bins: auto  
    savefig.bbox: tight