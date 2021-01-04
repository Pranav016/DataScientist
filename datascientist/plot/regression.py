import seaborn
import numpy

import seaborn.Implot
import seaborn.residplot
import seaborn.regplot
import seaborn.scatterplot
import seaborn.histplot


def _implot(*, x=None, y=None, data=None, hue=None, col=None, row=None, palette=None):
    """For more info visit :
        https://seaborn.pydata.org/generated/seaborn.lmplot.html
    """

    return Implot(x=x, y=y, data=data, hue=hue, col=col, row=row, palette=palette)

def _scatterplot(*, x=None, y=None, hue=None, style=None, data=None):
    """For more info visit :
        https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    """    
        
    return scatterplot(x=x, y=y, hue=hue, style=style, data=data)
    
def _regplot(*, x=None, y=None, data=None):
    """For more info visit :
        https://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot
    """ 

    return regplot(x=x, y=y, data=data)

def _residplot(*, x=None, y=None, data=None):
    """For more info visit :
        https://seaborn.pydata.org/generated/seaborn.residplot.html
    """

    return residplot(x=x, y=y, data=data)

def _histplot(*, x=None, y=None, data=None, hue=None):
    """For more info visit :
        https://seaborn.pydata.org/generated/seaborn.histplot.html
    """
    return histplot(x=x, y=y, data=data)

def _plots(x, y, data=None):
    _implot(x=x, y=y, data=data)
    _histplot(x=x,y=y,data=data)
    _residplot(x=x, y=y, data=data)
    _regplot(x=x, y=y, data=data)
    _scatterplot(x=x, y=y, data=data)