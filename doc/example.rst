Realistic example
=================

The ``melo`` package comes pre-bundled with a text file containing the home spread (home score minus away score) for all NFL games 2008–2018.
Let's load this dataset and see how to use the model to predict the spread of "future" games using the historical spread data.

First, let's import ``Melo`` and load the ``nfl_scores.dat`` package data.
Let's also load numpy_ for convenience.

.. code-block:: python

   import pkgutil
   import numpy as np

   from melo import Melo

   # the package comes pre-bundled with an example NFL dataset
   pkgdata = pkgutil.get_data('melo', 'nfl_scores.dat').splitlines()

The ``nfl_scores.dat`` package data looks like this:

.. code-block:: text

   2009-09-10 PIT TEN   3
   2009-09-13 ARI  SF  -4
   2009-09-13 ATL MIA  12
   2009-09-13 BAL  KC  14
   2009-09-13 CAR PHI -28
   2009-09-13 CIN DEN  -5
   2009-09-13 CLE MIN -14
   2009-09-13  GB CHI   6
   2009-09-13 HOU NYJ -17
   2009-09-13 IND JAC   2
   2009-09-13  NO DET  18
   2009-09-13 NYG WAS   6
   ...

After we've loaded the package data, we'll need to split the game data into separate columns.

.. code-block:: python

   times, teams_home, teams_away, spreads = zip(*[l.split() for l in pkgdata])

Next, we need to specify a one-dimensional array of ``lines`` which define the binary over/under outcomes. The vast majority of NFL spreads fall between -50.5 and 50.5 points, so I'll partition the spreads within this range. Here I choose half point lines so there is no ambiguity as to whether a comparison falls above or below a given threshold.

.. code-block:: python

   lines = np.arange(-50.5, 51.5)

Much like traditional Elo ratings, the ``melo`` model includes a hyperparameter ``k`` which controls how fast the ratings update. Prior experience indicates that ``k=0.245`` is a good choice for NFL games. Generally speaking, this hyperparameter must be tuned for each use case.

The model also includes a ``bias`` term which can either be a scalar (same for all comparisons) or a vector (specified for each comparison). For simplicity, let's ignore the occasional NFL game on a neutral field and choose a constant home field advantage ``bias = 0.166`` which handicaps ``teams_away``.

Optionally, we can also specify a function which describes how much the ratings should regress to the mean as a function of elapsed time between games. Here I regress the ratings to the mean by a fixed fraction each off season. To accomplish this, I create a function

.. code-block:: python

   def regress(dormant_months):
      """
      Regress ratings to the mean by 40% if the team
      has not played for three or more months

      """
      return .4 if dormant_months > 3 else 0

and supply this function to the ``Melo`` class using ``regress_period='month'`` to set the units of the decay function. The resulting constructor is thus:

.. code-block:: python

   nfl_spreads = Melo(
       times, teams_home, teams_away, spreads, lines=lines,
       k=.245, bias=.166, regress=regress, regress_period='month',
       commutes=False
   )

.. note::

   Spread comparisons are point differences which anti-commute so I must set ``commutes=False`` to ensure the appropriate behavior of the model under interchange of home and away team labels.

Constructing the ``Melo`` class object also trains the model, so it will take a few seconds to load. Once the object is created, you can easily generate predictions by supplying a new comparison. As before, I use ``bias=0.166`` to account for home field advantage.

.. code-block:: python

   # time one day after the last model update
   time = nfl_spreads.last_update + np.timedelta64(1, 'D')

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC', bias=.166)

   # predict the median outcome at that time
   median = nfl_spreads.median(time, 'CLE', 'KC', bias=.166)

   # predict the interquartile range at that time
   low, median, high = nfl_spreads.quantile(time, 'CLE', 'KC', q=[.25, .5, .75], bias=.166)

   # predict the probability that CLE wins
   win_prob =  nfl_spreads.prob(time, 'CLE', 'KC', bias=.166)

Additionally, the model can rank teams by their expected performance against a league average opponent on a neutral field. Ranking options are ``order=median``, ``mean``, and ``win``:

.. code-block:: python

   # rank teams by expected median spread against average team
   ranked_list = nfl_spreads.rank(time, order='median')

.. _numpy: http://www.numpy.org
