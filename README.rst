MELO
====

*Margin-dependent Elo ratings and predictions model*

.. image:: https://travis-ci.org/morelandjs/melo.svg?branch=master
    :target: https://travis-ci.org/morelandjs/melo

Documentation
-------------

`moreland.dev/projects/melo <https://moreland.dev/projects/melo>`_

Quick start
-----------

Requirements: Python 2.7 or 3.3+ with numpy_ and scipy_.

Install the latest release with pip_::

   pip install melo

Example usage::

   import pkgutil
   import numpy as np
   from melo import Melo

   # the package comes pre-bundled with an example dataset
   pkgdata = pkgutil.get_data('melo', 'nfl.dat').splitlines()
   dates, teams_home, scores_home, teams_away, scores_away = zip(
       *[l.split() for l in pkgdata[1:]])

   # define a binary comparison statistic
   spreads = [int(h) - int(a) for h, a
       in zip(scores_home, scores_away)]

   # hyperparameters and options
   k = 0.245
   biases = 0.166
   lines = np.arange(-50.5, 51.5)
   regress = lambda months: .413 if months > 3 else 0
   regress_unit = 'month'
   commutes = False

   # initialize the estimator
   nfl_spreads = Melo(k, lines=lines, commutes=commutes,
                      regress=regress, regress_unit=regress_unit)

   # fit the estimator to the training data
   nfl_spreads.fit(dates, teams_home, teams_away, spreads,
                   biases=biases)

   # specify a comparison time
   time = nfl_spreads.last_update

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC', biases=biases)
   print('CLE VS KC: {}'.format(mean))

   # rank nfl teams at end of 2018 regular season
   rankings = nfl_spreads.rank(time, statistic='mean')
   for team, rank in rankings:
       print('{}: {}'.format(team, rank))

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
