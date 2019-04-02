# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import stats


class Melo:
    """
    Margin-dependent Elo (MELO)

    **Required parameters**

    These must be array_like objects of equal length.

    - *times*   -- datetime, comparison time stamp
    - *labels1* -- string, first entity's label name
    - *labels2* -- string, second entity's label name
    - *values*  -- float, comparison value

    **Optional parameters**

    - *lines*    -- array_like, comparison threshold line(s)
    - *k*        -- float, rating update factor
    - *bias*     -- float, bias (shift) ratings toward either label
    - *decay*    -- callable function of elapsed time,
                    regresses ratings toward the mean
    - *commutes* -- bool, behavior of values under label interchange

    See online documentation <placeholder> for usage details.

    """
    def __init__(self, times, labels1, labels2, values,
                 lines=0, k=0, smooth=0, dist='normal',
                 bias=None, regress=None, commutes=False):

        self.times = np.array(times, dtype='datetime64[s]', ndmin=1)
        self.labels1 = np.array(labels1, dtype='str', ndmin=1)
        self.labels2 = np.array(labels2, dtype='str', ndmin=1)
        self.values = np.array(values, dtype='float', ndmin=1)
        self.lines = np.array(lines, dtype='float', ndmin=1)

        self.k = k

        if smooth < 0:
            raise ValueError('smooth must be non-negative')

        self.smooth = max(smooth, 1e-9)

        if dist not in ['cauchy', 'logistic', 'normal']:
            raise ValueError('no such distribution')

        self.dist = {
            'cauchy': stats.cauchy,
            'logistic': stats.logistic,
            'normal': stats.norm,
        }[dist]

        if bias is None:
            self.biases = np.zeros_like(values, dtype='float')
        elif np.isscalar(bias):
            self.biases = np.full_like(values, bias, dtype='float')
        else:
            self.biases = np.array(bias, dtype='float', ndmin=1)

        if regress is None:
            self.regress = lambda t: 0
        elif callable(regress):
            self.regress = regress
        else:
            raise ValueError('regress must be a callable function.')

        if commutes is True:
            self.conjugate = lambda x: x
        else:
            if all(self.lines != -np.flip(self.lines)):
                raise ValueError(
                    "lines must be symmetric when commutes is False"
                )
            self.conjugate = lambda x: -(
                np.fliplr(x) if x.ndim > 1 else np.flip(x)
            )

        self.comparisons = np.sort(
            np.rec.fromarrays([
                self.times,
                self.labels1,
                self.labels2,
                self.values,
                self.biases,
            ], names=(
                'time',
                'label1',
                'label2',
                'value',
                'bias',
            )), order='time', axis=0)

        self.labels = np.union1d(self.labels1, self.labels2)
        self.first_update = self.comparisons.time.min()
        self.last_update = self.comparisons.time.max()
        self.prior_rating = self.infer_prior_rating(self.values, self.lines)
        self.ratings_history = self.calculate_ratings()

    def infer_prior_rating(self, values, lines):
        """
        Infer minimim bias prior rating from the training data.

        """
        prob = np.mean(values[:, np.newaxis] - lines > 0, axis=0)

        TINY = 1e-6
        prob = np.clip(prob, TINY, 1 - TINY)

        return -0.5*self.dist.isf(prob)

    def evolve(self, rating, elapsed_time):
        """
        Evolve rating to a future time and regress to the mean.

        This function applies the user defined function 'regress'
        which defaults to zero (no regression).

        """
        x = self.regress(elapsed_time)

        return (1 - x)*rating + x*self.prior_rating

    def query_rating(self, time, label):
        """
        Query label's rating at the specified 'time' accounting
        for rating regression.

        Returns prior_rating if label == average.

        """
        time = np.datetime64(time)

        if label.lower() == 'average':
            return self.prior_rating
        elif label not in self.ratings_history:
            raise ValueError("no such label in comparisons")

        label_ratings = self.ratings_history[label]
        preceding = label_ratings.time < time

        if any(preceding):
            last_update = label_ratings[preceding][-1]
            return self.evolve(last_update.rating, time - last_update.time)

        return self.prior_rating

    def calculate_ratings(self):
        """
        Calculate each label's Elo ratings at each of the specified line(s).
        This function returns a dictionary mapping each label to a time-sorted
        structured record array of (time, rating) entries.

        """
        # initialize ratings history for each label
        ratings_history = {label: [] for label in self.labels}

        # temporary variable to store each label's last rating
        prev_update = {
            label: (self.first_update, self.prior_rating)
            for label in self.labels
        }

        # loop over all binary comparisons
        for (time, label1, label2, value, bias) in self.comparisons:

            # query ratings and evolve to the current time
            rating1, rating2 = [
                self.evolve(prev_rating, time - prev_time)
                for prev_time, prev_rating in [
                        prev_update[label1],
                        prev_update[label2],
                ]
            ]

            # expected and observed comparison outcomes
            rating_diff = rating1 + self.conjugate(rating2) + bias
            expected = self.dist.cdf(rating_diff)
            observed = self.dist.sf(self.lines, loc=value, scale=self.smooth)

            # update current ratings
            rating_change = self.k * (observed - expected)
            rating1 += rating_change
            rating2 += self.conjugate(rating_change)

            # record current ratings
            for label, rating in [(label1, rating1), (label2, rating2)]:
                ratings_history[label].append((time, rating))
                prev_update[label] = (time, rating)

        # convert ratings history to a structured rec.array
        for label in ratings_history.keys():
            ratings_history[label] = np.rec.array(
                ratings_history[label], dtype=[
                    ('time', 'datetime64[s]'),
                    ('rating', 'float', self.lines.size)
                ]
            )

        return ratings_history

    def _predict(self, time, label1, label2, bias=0):
        """
        Predict the probability that a comparison between label1 and label2
        covers every possible value of the line (internal function).

        """
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        rating_diff = rating1 + self.conjugate(rating2) + bias

        return self.lines, self.dist.cdf(rating_diff)

    def probability(self, times, labels1, labels2, bias=0, lines=0):
        """
        Predict the probabilities that a comparison between label1 and label2
        covers each specified line.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)
        lines = np.array(lines, dtype='float', ndmin=1)

        if np.isscalar(bias):
            biases = np.full_like(times, bias, dtype='float')
        else:
            biases = np.array(bias, dtype='float', ndmin=1)

        probabilities = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):

            probabilities.append(
                np.interp(lines, *self._predict(time, label1, label2, bias=bias))
            )

        return np.squeeze(probabilities)

    def percentile(self, times, labels1, labels2, bias=0, p=50):
        """
        Predict the percentiles for a comparison between label1 and label2.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(bias):
            biases = np.full_like(times, bias, dtype='float')
        else:
            biases = np.array(bias, dtype='float', ndmin=1)

        p = np.true_divide(p, 100.0)

        if np.count_nonzero(p < 0.0) or np.count_nonzero(p > 1.0):
            raise ValueError("percentiles must be in the range [0, 100]")

        percentiles = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):

            x, F = self._predict(time, label1, label2, bias=bias)

            percentiles.append(np.interp(p, np.sort(1 - F), x))

        return np.squeeze(percentiles)

    def quantile(self, times, labels1, labels2, bias=0, q=.5):
        """
        Predict the quantiles for a comparison between label1 and label2.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(bias):
            biases = np.full_like(times, bias, dtype='float')
        else:
            biases = np.array(bias, dtype='float', ndmin=1)

        q = np.asarray(q)

        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            raise ValueError("quantiles must be in the range [0, 1]")

        quantiles = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):

            x, F = self._predict(time, label1, label2, bias=bias)

            quantiles.append(np.interp(q, np.sort(1 - F), x))

        return np.squeeze(quantiles)

    def mean(self, times, labels1, labels2, bias=0):
        """
        Predict the mean value for a comparison between label1 and label2.

        Calculates the mean of the cumulative distribution function F(x)
        using integration by parts:

        E(x) = \int x P(x) dx
             = x F(x) | - \int F(x) dx

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(bias):
            biases = np.full_like(times, bias, dtype='float')
        else:
            biases = np.array(bias, dtype='float', ndmin=1)

        means = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):
            x, F = self._predict(time, label1, label2, bias=bias)
            mean = np.trapz(F, x) - (x[-1]*F[-1] - x[0]*F[0])
            means.append(np.asscalar(mean))

        return np.squeeze(means)

    def median(self, times, labels1, labels2, bias=0):
        """
        Predict the median value for a comparison between label1 and label2.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(bias):
            biases = np.full_like(times, bias, dtype='float')
        else:
            biases = np.array(bias, dtype='float', ndmin=1)

        medians = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):
            median = self.quantile(time, label1, label2, q=.5, bias=bias)
            medians.append(np.asscalar(median))

        return np.squeeze(medians)

    def residuals(self, statistic='mean', standardize=False):
        """
        Returns an array of model validation residuals.

        if standardize == False:
            residual = y_pred - y_obs

        if standardize == True:
            residual = (y_pred - y_obs) / sigma_pred.

        """
        residuals = []

        for (time, label1, label2, observed, bias) in self.comparisons:

            if statistic == 'mean':
                predicted = self.mean(time, label1, label2, bias=bias)
            elif statistic == 'median':
                predicted = self.median(time, label1, label2, bias=bias)
            else:
                raise ValueError("statistic options are 'mean' and 'median'")

            residual = predicted - observed

            if standardize is True:
                qlo, qhi = self.quantile(time, label1, label2,
                                         q=[.159, .841], bias=bias)
                residual /= .5*abs(qhi - qlo)

            residuals.append(residual)

        return np.array(residuals)

    def quantiles(self):
        """
        Returns an array of model validation quantiles.

        """
        quantiles = []

        for (time, label1, label2, observed, bias) in self.comparisons:

            quantile = self.probability(time, label1, label2,
                                        lines=observed, bias=bias)

            quantiles.append(quantile)

        return np.array(quantiles)

    def entropy(self):
        """
        Returns the simulation's total cross entropy:

        S = -\Sum obs*log(pred) + (1 - obs)*log(1 - pred).

        """
        entropy = 0

        for (time, label1, label2, observed, bias) in self.comparisons:

            lines, pred = self._predict(time, label1, label2, bias=bias)
            obs = np.heaviside(observed - lines, .5)

            entropy += -np.sum(
                obs*np.log(pred) + (1 - obs)*np.log(1 - pred)
            )

        return entropy

    def rank(self, time, statistic='mean'):
        """
        Ranks labels according to the specified statistic.
        Returns a rank sorted list of (label, rank) pairs.

        """
        if statistic == 'mean':
            func = self.mean
        elif statistic == 'median':
            func = self.median
        else:
            raise ValueError('no such distribution statistic')

        ranked_list = [
            (label, func(time, label, 'average'))
            for label in self.labels
        ]

        return sorted(ranked_list, key=lambda v: v[1], reverse=True)

    def sample(self, times, labels1, labels2, bias=0, size=100):
        """
        Draw random samples from the predicted probability distribution.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(bias):
            biases = np.full_like(times, bias, dtype='float')
        else:
            biases = np.array(bias, dtype='float', ndmin=1)

        if size < 1 or not isinstance(size, int):
            raise ValueError("sample size must be a positive integer")

        samples = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):

            x, F =  self._predict(time, label1, label2, bias=bias)
            rand = np.random.rand(size)

            samples.append(np.interp(rand, np.sort(1 - F), x))

        return np.squeeze(samples)
