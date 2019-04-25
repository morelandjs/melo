.. _example:

Example
=======

The ``melo`` package comes pre-bundled with a text file containing the final score of all regular season NFL games 2009–2018.
Let's load this dataset and use the model to predict the point spread and point total of "future" games using the historical game data.

First, let's import ``Melo`` and load the ``nfl.dat`` package data.
Let's also load numpy_ for convenience. ::

   import pkgutil
   import numpy as np

   from melo import Melo

   # the package comes pre-bundled with an example NFL dataset
   pkgdata = pkgutil.get_data('melo', 'nfl.dat').splitlines()

The ``nfl.dat`` package data looks like this:

.. code-block:: text

   # date, home, score, away, score
   2009-09-10 PIT 13 TEN 10
   2009-09-13 ATL 19 MIA  7
   2009-09-13 BAL 38 KC  24
   2009-09-13 CAR 10 PHI 38
   2009-09-13 CIN  7 DEN 12
   2009-09-13 CLE 20 MIN 34
   2009-09-13 HOU  7 NYJ 24
   2009-09-13 IND 14 JAC 12
   2009-09-13 NO  45 DET 27
   2009-09-13 TB  21 DAL 34
   2009-09-13 ARI 16 SF  20
   2009-09-13 NYG 23 WAS 17
   ...

After we've loaded the package data, we'll need to split the game data into separate columns. ::

   dates, teams_home, scores_home, teams_away, scores_away  = zip(*[l.split() for l in pkgdata[1:]])

Point spread predictions
------------------------

Let's start by analyzing the home team point spreads. ::

   spreads = [int(a) - int(b) for a, b in zip(scores_home, scores_away)]

We'll also need to specify a one-dimensional array ``lines`` which determines the paired comparisons.
The vast majority of NFL spreads fall between -50.5 and 50.5 points, so let's partition the spreads within this range.
Here I choose half point lines so there is no ambiguity as to whether a comparison falls above or below a given threshold. ::

   spread_lines = np.arange(-50.5, 51.5)

Much like traditional Elo ratings, the ``melo`` model includes a hyperparameter ``k`` which controls how fast the ratings update.
Prior experience indicates that ``k=0.245`` is a good choice for NFL games.
Generally speaking, this hyperparameter must be tuned for each use case.

The model also includes a ``bias`` parameter that can be used to implement transient effects such as home field advantage.
It can be a scalar (same for all comparisons) or a vector (specified for each comparison).
For simplicity, let's ignore the occasional NFL game on a neutral field and choose a constant home field advantage. ::

   hfa = 0.166

Additionally, we can also specify a function which describes how much the ratings should regress to the mean as a function of elapsed time between games.
Here I regress the ratings to the mean by a fixed fraction each off season. To accomplish this, I create a function ::

   def regress(dormant_months):
      """
      Regress ratings to the mean by 40% if the team
      has not played for three or more months

      """
      return .4 if dormant_months > 3 else 0

and supply this function to the ``Melo`` constructor using ``regress_unit='month'`` to set the units of the decay function.

Assembling the previous components, the model is trained as follows: ::

   nfl_spreads = Melo(
       dates, teams_home, teams_away, spreads, lines=spread_lines,
       k=.245, bias=hfa, regress=regress,
       regress_unit='month', commutes=False
   )

.. note::

   Spread comparisons are point differences which anti-commute so I must set ``commutes=False`` to ensure the appropriate behavior of the model under interchange of home and away team labels.

   Additionally, when ``commutes=False``, ``lines`` *must* be symmetric, i.e\. ``lines == lines[::-1]``.

Constructing the ``Melo`` class object also trains the model, so it will take a few seconds to load.
Once the object is created, one can easily generate predictions by calling its various instance methods: ::

   # time one day after the last model update
   time = nfl_spreads.last_update + np.timedelta64(1, 'D')

   # predict the mean outcome at 'time'
   nfl_spreads.mean(time, 'CLE', 'KC', bias=hfa)

   # predict the median outcome at 'time'
   nfl_spreads.median(time, 'CLE', 'KC', bias=hfa)

   # predict the interquartile range at 'time'
   nfl_spreads.quantile(time, 'CLE', 'KC', q=[.25, .5, .75], bias=hfa)

   # predict the win probability at 'time'
   nfl_spreads.probability(time, 'CLE', 'KC', bias=hfa)

   # generate prediction samples at 'time'
   nfl_spreads.sample(time, 'CLE', 'KC', bias=hfa, size=100)

.. note::

   Here I've used ``bias=hfa`` to apply home field advantage, but I could just as easily set ``bias=0`` to generate predictions for a neutral field.

Furthermore, the model can rank teams by their expected performance against a league average opponent on a neutral field.
Let's rank the NFL teams at the end of the 2018–2019 season according to their expected mean point spread against a league average opponent: ::

   # rank teams by expected median spread against average team
   nfl_spreads.rank(time, order='mean')

Or alternatively, we can rank teams by their expected win probability against a league average opponent: ::

   # rank teams by expected win prob against average team
   nfl_spreads.rank(time, order='win')

Point total predictions
-----------------------

Everything demonstrated so far can also be applied to point total comparisons ::

   totals = [int(a) + int(b) for a, b in zip(scores_home, scores_away)]

with a few small changes.

First, we'll need to change our ``lines`` so they cover the expected range of point total comparisons: ::

   total_lines = np.arange(-0.5, 105.5)

Next, we'll need to set ``commutes=True`` since the point total comparisons are invariant under label interchange.

Finally, we'll want to provide somewhat different inputs for the ``k``, ``bias``, and ``regress`` arguments.
Putting the pieces together: ::

   nfl_totals = Melo(
       dates, teams_home, teams_away, totals, lines=total_lines,
       k=.245, bias=0, regress=lambda months: .3 if months > 3 else 0,
       regress_unit='month', commutes=True
   )

And voila! We can easily predict the outcome of a future point total comparison: ::

   # time one day after the last model update
   time = nfl_totals.last_update + np.timedelta64(1, 'D')

   # predict the mean outcome at 'time'
   nfl_totals.mean(time, 'CLE', 'KC')


.. _numpy: http://www.numpy.org
