Usage
=====

``melo`` is a computer model to generate rankings and predictions from paired comparison time series data.
It has obvious applications to sports, but the framework is general and can be used for numerous other purposes including consumer surveys and asset pricing.

Overview
--------

This is a brief overview of the ``melo`` Python package.
See `Theory <theory.html>`_ for an explanation of the underlying math.

1. Initialization
^^^^^^^^^^^^^^^^^

First, import the Melo class. ::

   from melo import Melo

Next, create a Melo class object and specify its constructor arguments. ::

   melo_instance = Melo(
      k, lines=lines, sigma=sigma, regress=regress,
      regress_unit=regress_unit, dist=dist, commutes=commutes
   )

Parameters
""""""""""
* **k** *(float)* -- At bare minimum, you'll need to specify the rating update factor k which is the first and only positional argument. The k factor controls the magnitude of each rating update, with larger k values making the model more responsive to each comparison outcome. Its value should be chosen by minimizing the model's predictive error.

* **lines** *(array_like of float, optional)* -- The lines array specifies the sequence of binary comparison thresholds. The default, lines=0, corresponds to a classical Elo model where the comparison outcome is True if the value is greater than 0 and False otherwise. In general, you'll want to create an array of lines spanning the range of possible outcomes (see `Example <example.html>`_).

* **sigma** *(float, optional)* -- This parameter adds some uncertainty to each observed value when training the model. When sigma=0 (default), the comparison is True if the value exceeds a given line and False otherwise. For sigma > 0, it gives the comparison operator (step function) a soft edge of width sigma. Small sigma values generally help to smooth and regulate the model predictions.

* **regress** *(function, optional)* -- This argument provides an entry point for implementing rating decay. It must be a scalar function of one input variable (elapsed time) which returns a single number (fractional regression to the mean). The default, regress = lambda time: 0, applies no regression to the mean as a function of elapsed time.

* **regress_unit** *(string, optional)* -- Sets the elapsed time units of the regress function. Options are: year (default), month, week, day, hour, minute, second, millisecond, microsecond, nanosecond, picosecond, femtosecond, and attosecond. For example, suppose regress_unit='year' and regress = lambda time: 0.2 if time > 1 else 0. This means the model will regress each rating to the mean by 20% if the elapsed time since the last update is greater than one year.

* **dist** *(string, optional)* -- Specifies the type of distribution function used to convert rating differences into probabilities. Options are normal (default) and logistic. Switching distribution types will generally require somewhat different hyperparameters.

* **commutes** *(bool, optional)* -- This parameter describes the expected behavior of the estimated values under label interchange. If commutes=False, it is assumed that the comparisons anti-commute under label interchange (default behavior), and if commutes=True, it is assumed they commute. For example, point totals require commutes=True and point spreads require commutes=False.

2. Training data
^^^^^^^^^^^^^^^^

Each ``melo`` training **input** is a tuple of the form ``(time, label1, label2)`` and each training **output** is a single number ``value``.
This training data is passed to the model as four array_like objects of equal length:

* times is an array_like object of type np.datetime64 (or compatible string). It specifies the time at which the comparison was made.
* labels1 and labels2 are array_like objects of type string. They specify the first and second label names of the entities involved in the comparison.
* values is an array_like object of type float. It specifies the numeric value of the comparison, e.g. the value of the point spread or point total.

.. warning::
   It is assumed that the elements of each array match up, i.e\. the n-th element of each array should correspond to the same comparison.
   It is not necessary that the comparisons are time ordered.

For example, the data used to train the model might look like the following: ::

   times = ['2009-09-10', '2009-09-13', '2009-09-13']
   labels1 = ['PIT', 'ATL', 'BAL']
   labels2 = ['TEN', 'MIA', 'KC']
   values = [3, 12, 14]

3. Model calibration
^^^^^^^^^^^^^^^^^^^^

The model is calibrated by calling the fit function on the training data. ::

   melo_instance.fit(times, labels1, labels2, values, biases=0)

Optionally, when training the model you can specify ``biases`` (float or array_like of floats). These are numbers which add to (or subtract from) the rating difference of each comparison, i.e.

.. math::
   \Delta R = R_\text{label1} - R_\text{label2} + \text{bias}.

These factors can be used to account for transient advantages and disadvantages such as home court advantage and temporary injuries.
Positive bias numbers increase the expected value of the comparison, and negative values decrease it.
If ``biases`` is a single number, the bias factor is assumed to be constant for all comparisons.
Otherwise, there must be a bias factor for every training input.

4. Making predictions
^^^^^^^^^^^^^^^^^^^^^

Once the model is fit to the training data, there are a number of different functions which can be called to generate predictions for new comparisons at arbitrary points in time.

At the most basic level, the model predicts the survival function probability distribution :math:`P(\text{value} > \text{line})` as a function of the line.
This distribution is generated by the function call ::

   melo_instance.probability(times, labels1, labels2, biases=biases, lines=lines)

where times, labels1, labels2, and biases are the prediction inputs, and lines is the array of lines where the probability is to be estimated.

However, this function call is just the tip of the iceberg. Given this information, the model can predict many other interesting quantities such as the mean and median comparison values ::

   melo_instance.mean(times, labels1, labels2, biases=biases)

   melo_instance.median(times, labels1, labels2, biases=biases)

...arbitrary percentiles (or quantiles) of the distribution ::

   melo_instance.percentile(times, labels1, labels2, biases=biases, p=[10, 50, 90])

and it can even draw samples from the estimated survival function probability distribution ::

   melo_instance.sample(times, labels1, labels2, biases=biases, size=100)

Perhaps one of the most useful applications of the model is using its mean and median predictions to create rankings. This is aided by the rank function ::

   melo_instance.rank(time, statistic='mean')

which ranks the labels at the specified time according to their expected performance against an average opponent, i.e. an opponent with an average rating.

Reference
---------

Main class
^^^^^^^^^^
.. autoclass:: melo.Melo

Training function
"""""""""""""""""
.. autofunction:: melo.Melo.fit

Prediction functions
""""""""""""""""""""
.. autofunction:: melo.Melo.probability

.. autofunction:: melo.Melo.percentile

.. autofunction:: melo.Melo.quantile

.. autofunction:: melo.Melo.mean

.. autofunction:: melo.Melo.median

.. autofunction:: melo.Melo.residuals

.. autofunction:: melo.Melo.quantiles

.. autofunction:: melo.Melo.rank

.. autofunction:: melo.Melo.sample
