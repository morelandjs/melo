# MELO: Margin-dependent Elo ratings and predictions
# Copyright 2019 J. Scott Moreland
# MIT License

from __future__ import division

import numpy as np

from .dist import normal, logistic


class Melo:
    r"""
    Margin-dependent Elo (MELO) class constructor

    Parameters
    ----------
    k : float
        Prefactor multiplying each rating update
        :math:`\Delta R = k\, (P_\text{obs} - P_\text{pred})`.

    lines : array_like of float, optional
        Handicap line or sequence of lines used to construct the vector of
        binary comparisons

        :math:`\mathcal{C}(\text{label1}, \text{label2}, \text{value}) \equiv
        (\text{value} > \text{lines})`.

        The default setting, lines=0, deploys the traditional
        Elo rating system.

    sigma : float, optional
        Smearing parameter which gives the observed probability,

        :math:`P_\mathrm{obs} = 1` if value > line else 0,

        a soft edge of width sigma.
        A small sigma value helps to regulate and smooth the predicted
        probability distributions.

    regress : function, optional
        Univariate scalar function regress = f(time) which describes how
        ratings should regress to the mean as a function of elapsed time.
        When this function value is zero, the rating is unaltered, and when
        it is unity, the rating is fully regressed to the mean.
        The time units are set by the parameter regress_units (see below).

    regress_unit : string, optional
        Units of elapsed time for the regress function.
        Options are: year (default), month, week, day, hour, minute, second,
        millisecond, microsecond, nanosecond, picosecond, femtosecond, and
        attosecond.

    dist : string, optional
        Probability distribution, "normal" (default) or "logistic", which
        converts rating differences into probabilities:

        :math:`P(\text{value} > \text{line}) =
        \text{dist.cdf}(\Delta R(\text{line}))`

    commutes : bool, optional
        If this is set to True, the comparison values are assumed to be
        symmetric under label interchange.
        Otherwise the values are assumed to be *anti*-symmetric under label
        interchange.

        For example, point-spreads should use commutes=False and point-totals
        commutes=True.

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
        'picosecond': 1e-12,
        'femtosecond': 1e-15,
        'attosecond': 1e-18,
    }

    def __init__(self, k, lines=0, sigma=0, regress=lambda x: 0,
                 regress_unit='year', dist='normal', commutes=False):

        if k < 0 or not np.isscalar(k):
            raise ValueError('k must be a non-negative real number')

        self.k = k

        self.lines = np.array(lines, dtype='float', ndmin=1)

        if sigma < 0 or not np.isscalar(sigma):
            raise ValueError('sigma must be a non-negative real number')

        self.sigma = np.float(max(sigma, 1e-12))

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

        if commutes is True:
            self.conjugate = lambda x: x
        else:
            if all(self.lines != -self.lines[::-1]):
                raise ValueError(
                    "lines must be symmetric when commutes is False"
                )
            self.conjugate = lambda x: -x[::-1]

        self.loss = 0
        self.first_update = None
        self.last_update = None
        self.labels = None
        self.training_data = None
        self.prior_rating = None
        self.ratings_history = None

    def _read_training_data(self, times, labels1, labels2, values, biases):
        """
        Internal function: Read training inputs and initialize class variables

        Parameters
        ----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        values : array_like of float
            List of comparison values.

        biases : array_like of float
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.


        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)
        values = np.array(values, dtype='float', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        self.first_update = times.min()
        self.last_update = times.max()
        self.labels = np.union1d(labels1, labels2)

        self.training_data = np.sort(
            np.rec.fromarrays([
                times,
                labels1,
                labels2,
                values,
                biases,
            ], names=(
                'time',
                'label1',
                'label2',
                'value',
                'bias',
            )), order='time', axis=0)

        prior_prob = np.mean(values[:, np.newaxis] - self.lines > 0, axis=0)

        TINY = 1e-6
        prob = np.clip(prior_prob, TINY, 1 - TINY)

        self.prior_rating = -0.5*self.dist.isf(prob)

    def evolve(self, rating, time_delta):
        """
        Evolve rating to a future time and regress to the mean.

        Parameters
        ----------
        rating : ndarray of float
            Array of label ratings.

        time_delta : np.timedelta64
            Elapsed time since the label's ratings were last updated.

        Returns
        -------
        rating : ndarray of float
            Label ratings after regression to the mean.

        """
        elapsed_seconds = time_delta / np.timedelta64(1, 's')
        elapsed_periods = elapsed_seconds / self.seconds_per_period
        regress = self.regress(elapsed_periods)

        return rating + regress*(self.prior_rating - rating)

    def query_rating(self, time, label):
        """
        Query label's rating at the specified 'time' accounting
        for rating regression.

        Parameters
        ----------
        time : np.datetime64
            Comparison datetime

        label : string
            Comparison entity label. If label == average, then the average
            rating of all labels is returned.

        Returns
        -------
        rating : ndarray of float
            Applies the user specified rating regression function to regress
            ratings to the mean as necessary and returns the ratings for the
            given label at the specified time.

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

    def fit(self, times, labels1, labels2, values, biases=0):
        """
        This function is used to calibrate the model on the training inputs.
        It computes and records each label's Elo ratings at the line(s) given
        in the class constructor.
        The function returns the predictions' total cross entropy loss.

        Parameters
        ----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        biases : array_like of float, optional
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.

        Returns
        -------
        loss : float
            Cross entropy loss for the model predictions.

        """
        # read training inputs and initialize class variables
        self._read_training_data(times, labels1, labels2, values, biases)

        # initialize ratings history for each label
        self.ratings_history = {label: [] for label in self.labels}

        # temporary variable to store each label's last rating
        prev_update = {
            label: (self.first_update, self.prior_rating)
            for label in self.labels
        }

        # calibration loss
        loss = 0

        # loop over all binary comparisons
        for (time, label1, label2, value, bias) in self.training_data:

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
            obs = self.dist.sf(self.lines, loc=value, scale=self.sigma)
            pred = self.dist.cdf(rating_diff)

            # update cross entropy
            loss += -(obs*np.log(pred) + (1 - obs)*np.log(1 - pred)).sum()

            # update current ratings
            rating_change = self.k * (obs - pred)
            rating1 += rating_change
            rating2 += self.conjugate(rating_change)

            # record current ratings
            for label, rating in [(label1, rating1), (label2, rating2)]:
                self.ratings_history[label].append((time, rating))
                prev_update[label] = (time, rating)

        # convert ratings history to a structured rec.array
        for label in self.ratings_history.keys():
            self.ratings_history[label] = np.rec.array(
                self.ratings_history[label], dtype=[
                    ('time', 'datetime64[s]'),
                    ('rating', 'float', self.lines.size)
                ]
            )

        self.loss = loss

        return loss

    def _predict(self, time, label1, label2, bias=0):
        """
        Internal function: Predict the survival function probability
        distribution that a comparison between label1 and label2 covers
        each value of the line.

        Parameters
        ----------
        time : np.datetime64
            Comparison datetime.

        label1 : string
            Comparison first entity label.

        label2 : string
            Comparison second entity label.

        bias : float, optional
            Bias number which adds a constant offset to the ratings.
            Positive bias factors favor label1.
            Default is 0, i.e. no bias.

        """
        rating1 = self.query_rating(time, label1)
        rating2 = self.query_rating(time, label2)

        rating_diff = rating1 + self.conjugate(rating2) + bias

        return self.lines, self.dist.cdf(rating_diff)

    def probability(self, times, labels1, labels2, biases=0, lines=0):
        r"""
        Predict the survival function probability
        .. math:: P(\text{value} > \text{lines})
        for the specified comparison(s).

        Parameters
        ----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        biases : array_like of float, optional
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.

        lines : array_like of float, optional
            Line or sequence of lines used to estimate the comparison
            distribution.
            Default is lines=0, in which case the model predicts the
            probability that value > 0.

        Returns
        -------
        probability : scalar or array_like of float
            If a single comparison is given and a single line is specified,
            this function returns a scalar.
            If multiple comparisons or multiple lines are given, this function
            returns an ndarray.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        lines = np.array(lines, dtype='float', ndmin=1)

        probabilities = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):
            predict = self._predict(time, label1, label2, bias=bias)

            probabilities.append(np.interp(lines, *predict))

        return np.squeeze(probabilities)

    def percentile(self, times, labels1, labels2, biases=0, p=50):
        """
        Predict the p-th percentile for the specified comparison(s).

        Parameters
        -----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        biases : array_like of float, optional
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.

        p : array_like of float, optional
            Percentile or sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
            Default is p=50, which computes the median.

        Returns
        -------
        percentile : scalar or ndarray of float
            If one comparison is given, and p is a single percentile,
            then the result is a scalar.
            If multiple comparisons or multiple percentiles are given, the
            result is an ndarray.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        p = np.true_divide(p, 100.0)

        if np.count_nonzero(p < 0.0) or np.count_nonzero(p > 1.0):
            raise ValueError("percentiles must be in the range [0, 100]")

        percentiles = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):

            x, F = self._predict(time, label1, label2, bias=bias)

            percentiles.append(np.interp(p, np.sort(1 - F), x))

        return np.squeeze(percentiles)

    def quantile(self, times, labels1, labels2, biases=0, q=.5):
        """
        Predict the q-th quantile for the specified comparison(s).

        Parameters
        -----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        biases : array_like of float, optional
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.

        q : array_like of float, optional
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive.
            Default is q=0.5, which computes the median.

        Returns
        -------
        quantile : scalar or ndarray of float
            If one comparison is given, and q is a single quantiles,
            then the result is a scalar.
            If multiple comparisons or multiple quantiles are given, the
            result is an ndarray.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        q = np.asarray(q)

        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            raise ValueError("quantiles must be in the range [0, 1]")

        quantiles = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):

            x, F = self._predict(time, label1, label2, bias=bias)

            quantiles.append(np.interp(q, np.sort(1 - F), x))

        return np.squeeze(quantiles)

    def mean(self, times, labels1, labels2, biases=0):
        """
        Predict the mean for the specified comparison(s).

        Parameters
        -----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        biases : array_like of float, optional
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.

        Returns
        -------
        mean : scalar or ndarray of float
            If one comparison is given, then the result is a scalar.
            If multiple comparisons are given, then the result is an ndarray.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        means = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):
            x, F = self._predict(time, label1, label2, bias=bias)
            mean = np.trapz(F, x) - (x[-1]*F[-1] - x[0]*F[0])
            means.append(np.asscalar(mean))

        return np.squeeze(means)

    def median(self, times, labels1, labels2, biases=0):
        """
        Predict the median for the specified comparison(s).

        Parameters
        -----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        biases : array_like of float, optional
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.

        Returns
        -------
        median : scalar or ndarray of float
            If one comparison is given, then the result is a scalar.
            If multiple comparisons are given, then the result is an ndarray.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        medians = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):
            median = self.quantile(time, label1, label2, q=.5, biases=bias)
            medians.append(np.asscalar(median))

        return np.squeeze(medians)

    def residuals(self, statistic='mean', standardize=False):
        """
        Prediction residuals (or Z-scores if standardize is True) for all
        comparisons in the training data.
        The Z-scores should sample a unit normal distribution.

        Parameters
        ----------
        statistic : string, optional
            Type of prediction statistic.
            Options are 'mean' (default) or 'median'.

        standardize : bool, optional
            If standardize is True, divides prediction residuals by their
            one-sigma prediction uncertainty.
            Default value is False.

        Returns
        -------
        residuals : ndarray of float
            The residuals for each comparison in the training data.
            The residuals are time ordered, and may not appear in the same
            order as originally given.

        """
        if statistic == 'mean':
            func = self.mean
        elif statistic == 'median':
            func = self.median
        else:
            raise ValueError("predict options are 'mean' and 'median'")

        residuals = []

        for (time, label1, label2, obs, bias) in self.training_data:

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
        Prediction quantiles for all comparisons in the training data.
        The quantiles should sample a uniform distribution from 0 to 1.

        Returns
        -------
        quantiles : ndarray of float
            The quantiles for each comparison in the training data.
            The quantiles are time ordered, and may not appear in the same
            order as originally given.

        """
        quantiles = []

        for (time, label1, label2, obs, bias) in self.training_data:

            quantile = self.probability(
                time, label1, label2, lines=obs, bias=bias
            )

            quantiles.append(quantile)

        return np.array(quantiles)

    def rank(self, time, statistic='mean'):
        """
        Ranks labels by comparing each label to the average label using
        the specified summary statistic.

        Parameters
        ----------
        time : np.datetime64
            The time at which the ranking should be computed.

        statistic : string, optional
            Determines the binary comparison ranking statistic.
            Options are 'mean' (default), 'median', or 'win'.

        Returns
        -------
        label rankings : list of tuples
            Returns a rank sorted list of (label, rank) pairs, where rank is
            the comparison value of the specified summary statistic.

        """
        if statistic == 'mean':
            func = self.mean
        elif statistic == 'median':
            func = self.median
        elif statistic == 'win':
            func = self.probability
        else:
            raise ValueError('no such order parameter')

        ranked_list = [
            (label, np.float(func(time, label, 'average')))
            for label in self.labels
        ]

        return sorted(ranked_list, key=lambda v: v[1], reverse=True)

    def sample(self, times, labels1, labels2, biases=0, size=100):
        """
        Draw random samples from the predicted comparison probability
        distribution.

        Parameters
        ----------
        times : array_like of np.datetime64
            List of datetimes.

        labels1 : array_like of string
            List of first entity labels.

        labels2 : array_like of string
            List of second entity labels.

        biases : array_like of float, optional
            Single bias number or list of bias numbers which match the
            comparison inputs.
            Default is 0, in which case no bias is used.

        size : int, optional
            Number of samples to be drawn.
            Default is 1, in which case a single value is returned.

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        if size < 1 or not isinstance(size, int):
            raise ValueError("sample size must be a positive integer")

        samples = []

        for time, label1, label2, bias in zip(times, labels1, labels2, biases):

            x, F = self._predict(time, label1, label2, bias=bias)
            rand = np.random.rand(size)

            samples.append(np.interp(rand, np.sort(1 - F), x))

        return np.squeeze(samples)
