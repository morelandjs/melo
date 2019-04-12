# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import erf, erfc, erfcinv


class normal:
    """
    Normal probability distribution function

    """
    def cdf(x, loc=0, scale=1):
        """
        Cumulative distribution function

        """
        return 0.5*(1 + erf((x - loc)/(scale*np.sqrt(2))))

    def sf(x, loc=0, scale=1):
        """
        Survival function

        """
        return 0.5*erfc((x - loc)/(scale*np.sqrt(2)))

    def isf(x, loc=0, scale=1):
        """
        Inverse survival function

        """
        return scale*np.sqrt(2)*erfcinv(2*x) + loc


class logistic:
    """
    Logistic probability distribution function

    """
    def cdf(x, loc=0, scale=1):
        """
        Cumulative distribution function

        """
        return 1 / (1 + np.exp(-(x - loc)/scale))

    def sf(x, loc=0, scale=1):
        """
        Survival function

        """
        expz = np.exp(-(x - loc)/scale)
        return expz/(1 + expz)

    def isf(x, loc=0, scale=1):
        """
        Inverse survival function

        """
        return scale*np.log((1 - x)/x) + loc
