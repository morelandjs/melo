Tests
=====

`melo` has a standard set of unit tests, whose current CI status is:

The unit tests do not check the physical accuracy of the model which is difficult to verify automatically.
Rather, this page shows a number of visual tests which can be used to access the model manually.

Toy model
---------

Consider a fictitious "sports league" of eight teams. Each team samples points from a `Poisson distribution <https://en.wikipedia.org/wiki/Poisson_distribution>`_ `X_\text{team} \sim \text{Pois}(\lambda_\text{team})` where `\lambda_\text{team}` is one of eight numbers

.. math::
   \lambda_\text{team} \in \{90, 94, 98, 100, 102, 106, 110\}

specifying the team's mean expected score. I simulate a series of games between the teams by sampling pairs `(\lambda_{\text{team}~a}, \lambda_{\text{team}~b})` with replacement from the above values. Then, for each game and team, I sample a Poisson score and record the result, producing a tuple

.. math::
   (\text{time}, \text{team}~a, \text{score}~a, \text{team}~b, \text{score}~b),

where time is a np.datetime64 object recording the time of the comparison, `\text{team}~a` and `\text{team}~b` are strings labeling each team by their `\lambda` values, and `\text{score}~a` and `\text{score}~b` are random integers. This process is repeated `\mathcal{O}(10^6)` times to accumulate a large number of games.

Point spread validation
-----------------------

I then calculate the score difference or `\text{spread} \equiv \text{score}~a - \text{score}~b` for each game to form a list of comparisons `(\text{time}, \text{team}~a, \text{team}~b, \text{spread})` and use these comparisons to train the margin-dependent Elo model:

.. code-block:: python

   lines = np.arange(-49.5, 50.5)
   Melo(times, teamsa, teamsb, spreads, lines=lines, k=1e-4, commutes=False)

The figure below shows the survival function of the point spread distribution for each team matched up against the team with `\lambda_\text{team} = 100`. Colored symbols are model predictions and black lines are exact (analytic) results. The top panel validates the model's prior predictions (before the first game), and the bottom panel validates its posterior predictions (after the last game).

.. figure:: _static/validate_spreads.png
   :alt: minimum bias spread distribution

Point total validation
----------------------

This figure is the same as above but for comparisons based on each game's point `\text{total} = \text{score}~a + \text{score}~b`. In addition, when calibrating the model, I set `commutes = True` since the point total commutes under interchange of label `a` and `b`.

.. figure:: _static/validate_totals.png
   :alt: minimum bias total distribution
