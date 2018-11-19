Examples
========

The ``melo`` package comes pre-bundled with a text file containing the home spread (home score minus away score) for all NFL games 2008–2018.
Let's load this dataset and see how to use the model to predict the outcome of "future" games using only the spread of the preceding games.

First, let's import ``Melo`` and load the ``nfl_scores.dat`` package data.
Let's also load numpy_ for convenience.

.. code-block:: python

   import pkgutil
   import numpy as np

   from melo import Melo

   # the package comes pre-bundled with an example NFL dataset
   pkgdata = pkgutil.get_data('melo', 'nfl_scores.dat').splitlines()

The ``nfl_scores.dat`` package data looks like this,

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

The first column is,
time, home, away, spread = zip(*[l.split() for l in pkgdata])

Here :python:`time` is a list of datetime strings, :python:`home` is a list of home team names, :python:`away` is a list of away team names, and :python:`spread` is a list of point spread comparison values.

Once the training data is loaded, we simply instantiate a Melo class object.
from datetime import timedelta

.. code-block:: python

   # specify values for the model training parameters
   nfl_spreads = Melo(
       time, home, away, spread, lines=np.arange(-50.5, 51.5),
       mode='Fermi', k=.245, bias=.166,
       decay=lambda t: 1 if t < timedelta(weeks=20) else .597
   )


.. code-block:: python

   # specify some 'current' time
   current_time = nfl_spreads.comparisons['time'][-1]

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(current_time, 'CLE', 'KC')

   # mean expected CLE vs KC point differential
   print('CLE VS KC: {}'.format(mean))
   score result

.. _numpy: http://www.numpy.org
