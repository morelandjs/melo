Tests
=====

`melo` has a standard set of unit tests, whose current CI status is:

The unit tests do not check the physical accuracy of the model which is difficult to verify automatically.
Rather, this page shows a number of visual tests which can be used to access the model manually.

Poisson toy-model
-----------------

For this purpose, I create a toy-model "sports league" of eight Poisson random variables.
The teams play against one another and randomly sample scores from their individual distributions.
The eight Poisson distributions are described by eight Poisson mean parameters,

.. math::
   \lambda_\text{team} = [12, 15, 18, 21, 22, 24, 27, 30].

Larger :math:`\lambda_\text{team}` values correspond to better teams which sample higher scores.

Each "game" between the Poisson variables :math:`\lambda_a` and :math:`\lambda_b` is therefore a tuple,

.. math::
   (\text{team } \lambda_a, \text{team } \lambda_b, \text{score } \lambda_a, \text{score } \lambda_b).

This game data can then be used to construct a list of pairwise comparisons between :math:`\lambda_a` and :math:`\lambda_b`.
For example, we could compare the point spread,

.. math::
   \text{point spread} = \text{score } \lambda_a - \text{score } \lambda_b;

or the point total,

.. math::
   \text{point total} = \text{score } \lambda_a + \text{score } \lambda_b.

I generate :math:`\mathcal{O}(10^5)` random comparisons and feed the comparisons into the model.
I then check that the model predictions agree with known analytic results for the difference and sum of two Poisson variables.

Point spread
------------

These are the model point spread predictions *before* any games are played:

.. figure:: _static/spread_prior.png
   :alt: minimum bias spread distribution

The teams initially sample points from a league-wide minimum-bias point spread distribution, i.e. they are generic and interchangeable.

The figure above shows the expected point spread distribution (colored symbols) of team :math:`\lambda_a=22` against every opponent in the league :math:`\lambda_b`, compared to the league-wide average distribution (black line).

The model updates the ratings for each Poisson team after every game.
The ratings should converge to their true values over time.

These are the model point spread predictions *after* all the games are played:

.. figure:: _static/spread_calibrated.png
   :alt: calibrated spread distribution

The updated model predictions (colored symbols) are now compared to the known `analytic result <https://en.wikipedia.org/wiki/Skellam_distribution>`_ for the difference between two Poisson random variables (black lines).
Agreement indicates that the model correctly estimated the true underlying point spread distribution.

Point total
-----------

This test is exactly the same as before, but for point total comparisons.

.. figure:: _static/total_prior.png
   :alt: minimum bias total distribution

The figure above shows the model predictions *before* the games are played, and the figure below shows the model predictions *after* the games are played.

.. figure:: _static/total_calibrated.png
   :alt: calibrated total distribution

As before, the colored symbols are the model prediction, and the black line is the exact `analytic result <https://en.wikipedia.org/wiki/Poisson_distribution>`_
, for team :math:`\lambda_a=22` playing team :math:`\lambda_b` (labeled in the legend).
Agreement indicates that the model correctly estimated the true underlying point total distribution.
