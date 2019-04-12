# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .dist import normal, logistic


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
    seconds = {
        'year': 3.154e7,
        'month': 2.628e6,
        'week': 604800.,
        'day': 86400.,
        'hour': 3600.,
        'minute': 60.,
        'second': 1.,
        'millisecond': 1e-3,
        'microsecond': 1e-6,
        'nanosecond': 1e-9,
        'picsecond': 1e-12,
        'femptosecond': 1e-15,
        'attosecond': 1e-18,
    }

    def __init__(self, times, labels1, labels2, values, lines=0, k=0,
                 smooth=0, regress=lambda x: 0, regress_unit='year',
                 dist='normal', bias=None, commutes=False):

        self.times = np.array(times, dtype='datetime64[s]', ndmin=1)
        self.labels1 = np.array(labels1, dtype='str', ndmin=1)
        self.labels2 = np.array(labels2, dtype='str', ndmin=1)
        self.values = np.array(values, dtype='float', ndmin=1)
        self.lines = np.array(lines, dtype='float', ndmin=1)

        self.k = k

        if smooth < 0:
            raise ValueError('smooth must be non-negative')

        self.smooth = max(smooth, 1e-12)

        if not callable(regress):
            raise ValueError('regress must be univariate scalar function')

        self.regress = regress

        if regress_unit not in self.seconds.keys():
            raise ValueError('regress_unit must be valid time unit (see docs)')

        self.seconds_per_period = self.seconds[regress_unit]

        if dist == 'normal':
            self.dist = normal
        elif dist == 'logistic':
            self.dist = logistic
        else:
            raise ValueError('no such distribution')

        if bias is None:
            self.biases = np.zeros_like(values, dtype='float')
        elif np.isscalar(bias):
            self.biases = np.full_like(values, bias, dtype='float')
        else:
            self.biases = np.array(bias, dtype='float', ndmin=1)

        if commutes is True:
            self.conjugate = lambda x: x
        else:
            if all(self.lines != -np.flip(self.lines)):
                raise ValueError(
                    "lines must be symmetric when commutes is False"
                )
            self.conjugate = lambda x: -x[::-1]

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
        self.entropy, self.ratings_history = self.calculate_ratings()

    def infer_prior_rating(self, values, lines):
        """
        Infer minimim bias prior rating from the training data.

        """
        prob = np.mean(values[:, np.newaxis] - lines > 0, axis=0)

        TINY = 1e-6
        prob = np.clip(prob, TINY, 1 - TINY)

        return -0.5*self.dist.isf(prob)

    def evolve(self, rating, time_delta):
        """
        Evolve rating to a future time and regress to the mean.

        This function applies the user defined function 'regress'
        which defaults to zero (no regression).

        """
        elapsed_seconds = time_delta / np.timedelta64(1, 's')
        elapsed_periods = elapsed_seconds / self.seconds_per_period
        regress = self.regress(elapsed_periods)

        return rating + regress*(self.prior_rating - rating)

    def query_rating(self, time, label):
        """
        Query label's rating at the specified 'time' accounting
        for rating regression.

        Returns prior_rating if label == average.

        """
        time = np.datetime64(time, 's')

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

        entropy = 0

        # prediction cross entropy
        def cross_entropy(pred, obs):
            return -(obs*np.log(pred) + (1 - obs)*np.log(1 - pred)).sum()

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
            obs = self.dist.sf(self.lines, loc=value, scale=self.smooth)
            pred = self.dist.cdf(rating_diff)

            # update cross entropy
            entropy += cross_entropy(pred, obs)

            # update current ratings
            rating_change = self.k * (obs - pred)
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

        entropy /= (self.lines.size * np.size(self.comparisons, axis=0))

        return entropy, ratings_history

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
            predict = self._predict(time, label1, label2, bias=bias)

            probabilities.append(np.interp(lines, *predict))

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

        E(x) = \\int x P(x) dx
             = x F(x) | - \\int F(x) dx

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

    def residuals(self, predict='mean', standardize=False):
        """
        Returns an array of model validation residuals.

        if standardize == False:
            residual = y_pred - y_obs

        if standardize == True:
            residual = (y_pred - y_obs) / sigma_pred.

        """
        if predict == 'mean':
            func = self.mean
        elif predict == 'median':
            func = self.median
        else:
            raise ValueError("predict options are 'mean' and 'median'")

        residuals = []

        for (time, label1, label2, obs, bias) in self.comparisons:

            pred = func(time, label1, label2, bias=bias)
            residual = pred - obs

            if standardize is True:

                qlo, qhi = self.quantile(
                    time, label1, label2, q=[.159, .841], bias=bias
                )

                residual /= .5*abs(qhi - qlo)

            residuals.append(residual)

        return np.array(residuals)

    def quantiles(self):
        """
        Returns an array of model validation quantiles.

        """
        quantiles = []

        for (time, label1, label2, obs, bias) in self.comparisons:

            quantile = self.probability(
                time, label1, label2, lines=obs, bias=bias
            )

            quantiles.append(quantile)

        return np.array(quantiles)

    def rank(self, time, order='mean'):
        """
        Ranks labels according to the specified order parameter.
        Returns a rank sorted list of (label, rank) pairs.

        """
        if order == 'mean':
            func = self.mean
        elif order == 'median':
            func = self.median
        elif order == 'win':
            func = self.probability
        else:
            raise ValueError('no such order parameter')

        ranked_list = [
            (label, np.float(func(time, label, 'average')))
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

            x, F = self._predict(time, label1, label2, bias=bias)
            rand = np.random.rand(size)

            samples.append(np.interp(rand, np.sort(1 - F), x))

        return np.squeeze(samples)
