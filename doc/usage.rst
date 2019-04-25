Usage
=====
``melo`` is a computer model to generate rankings and predictions from paired comparison time series data.
It has obvious applications to sports, but the framework is general and can be used for numerous other purposes including consumer surveys and asset pricing.

To use the model, import the ``Melo`` class ::

   from melo import Melo

The model takes four one-dimensional array-like objects of equal size, e.g\. ::

   times = ['2009-09-10', '2009-09-13', '2009-09-13']
   labels1 = ['PIT', 'ATL', 'BAL']
   labels2 = ['TEN', 'MIA', 'KC']
   values = [3, 12, 14]

Here ``times`` converts to a ``np.array`` with ``dtype='datetime64[s]'``, ``labels1`` and ``labels2`` convert to ``dtype='str'``, and ``values`` to ``dtype='float'``.
If the data type cannot be inferred, e.g\. int to float, the model will throw an error.

.. important::

   The array elements should match up, i.e\. the n-th element of each array should correspond to the same comparison. It is not necessary that the comparisons are time ordered.

Once trained, the model predicts the survival function `P(\text{value} > \ell)` for new comparisons between label1 and label2 at a given time (see below).
To estimate this distribution, the model expects a list of comparison lines `\ell`, e.g.\ ::

   lines = [-20, -10, 0, 10, 20]

to convert the distribution into a discrete (vector) representation.

.. note::

   The model's default behavior ``lines=0`` is equivalent to the traditional Elo rating system.

The model also has several hyperparameters which strongly affect its behavior:

* ``k`` – prefactor multiplying the rating update (bounty) determined by each comparison outcome

* ``bias`` – constant bias factor(s) added to each rating difference (or sum)

* ``regress`` – scalar single-variate real function; describes how the ratings should regress to the mean as a function of elapsed time

* ``regress_unit`` – time unit for regress function

* ``sigma`` – smears-out each comparison value by a fixed amount, yielding smoother probability distributions

Central to the Elo rating system is an assumed probability distribution which describes the distribution of sampled comparisons.

* ``dist`` – fixes the underlying probability distribution; options are "normal" and "logistic"

Initializing the class object also trains the model. For example, ::

   trained_model = Melo(times, labels1, labels2, values, lines=lines, k=k)

creates a calibrated ``Melo`` class instance.
Once the model is trained, predictions are made by calling various class functions, e.g\. ::

   # comparison time
   time = np.datetime64('now')

   # survival function P(value > 0)
   trained_model.probability(time, 'PIT', 'BAL', lines=0)

See :ref:`Example` section for a tutorial using NFL game data.

Reference
---------

Main class
^^^^^^^^^^
.. autoclass:: melo.Melo

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
