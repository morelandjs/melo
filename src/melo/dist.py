# MELO: Margin-dependent Elo ratings and predictions
# Copyright 2019 J. Scott Moreland
# MIT License

import numpy as np
from scipy.special import erf, erfc, erfcinv, expit


class normal:
    """
    Normal probability distribution function

    """
    @staticmethod
    def cdf(x, loc=0, scale=1):
        """
        Cumulative distribution function

        """
        return 0.5*(1 + erf((x - loc)/(scale*np.sqrt(2))))

    @staticmethod
    def sf(x, loc=0, scale=1):
        """
        Survival function

        """
        return 0.5*erfc((x - loc)/(scale*np.sqrt(2)))

    @staticmethod
    def isf(x, loc=0, scale=1):
        """
        Inverse survival function

        """
        return scale*np.sqrt(2)*erfcinv(2*x) + loc


class logistic:
    """
    Logistic probability distribution function

    """
    @staticmethod
    def cdf(x, loc=0, scale=1):
        """
        Cumulative distribution function

        """
        return expit((x - loc)/scale)

    @staticmethod
    def sf(x, loc=0, scale=1):
        """
        Survival function

        """
        return 1 - expit((x - loc)/scale)

    @staticmethod
    def isf(x, loc=0, scale=1):
        """
        Inverse survival function

        """
        np.seterr(divide='ignore')
        return scale*np.log(np.divide(1 - x, x)) + loc
