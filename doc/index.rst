melo
====

*Margin-dependent Elo ratings and predictions model*

* Author: J\. Scott Moreland
* Language: Python
* Source code: `github:morelandjs/melo <https://github.com/morelandjs/melo>`_

``melo`` generalizes the `Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`_ paired comparison model beyond binary outcomes to include margin-of-victory information. It does this by "redefining what it means to win" using a variable handicap to shift the threshold of paired comparison. The framework is general and has numerous applications in ranking, estimation, and time series prediction.

Quick start
-----------

Requirements: Python 2.7 or 3.3+ with numpy_ and scipy_.

Install the latest release with pip_: ::

   pip install melo

Example usage: ::

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
   bias = 0.166
   lines = np.arange(-50.5, 51.5)
   regress = lambda months: .413 if months > 3 else 0
   regress_unit = 'month'
   commutes = False

   # initialize the estimator
   nfl_spreads = Melo(k, lines=lines, commutes=commutes,
                      regress=regress, regress_unit=regress_unit)

   # fit the estimator to the training data
   nfl_spreads.fit(dates, teams_home, teams_away, spreads, bias=bias)

   # specify a comparison time
   time = nfl_spreads.last_update

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC', bias=bias)
   print('CLE VS KC: {}'.format(mean))

   # rank nfl teams at end of 2018 regular season
   rankings = nfl_spreads.rank(time, statistic='mean')
   for team, rank in rankings:
       print('{}: {}'.format(team, rank))

.. toctree::
   :caption: User guide
   :maxdepth: 2

   usage
   example

.. toctree::
   :caption: Technical info
   :maxdepth: 2

   theory
   tests

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
