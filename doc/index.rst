melo
====

*Margin-dependent Elo ratings and predictions model*

:Author: J\. Scott Moreland
:Language: Python
:Source code: `github:morelandjs/melo <https://github.com/morelandjs/melo>`_

``melo`` generalizes the `Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`_ paired comparison model beyond binary outcomes to include margin-of-victory information. It does this by "redefining what it means to win" using a variable handicap to shift the threshold of paired comparison. The framework is general and has numerous applications in ranking, estimation, and time series prediction.

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
   pkgdata = pkgutil.get_data('melo', 'nfl_scores.dat').splitlines()
   times, teams_home, teams_away, spreads = zip(*[l.split() for l in pkgdata])

   # specify values for the model training parameters
   nfl_spreads = Melo(
       times, teams_home, teams_away, spreads, commutes=False,
       lines=np.arange(-50.5, 51.5), k=.245, bias=.166,
       regress=lambda months: .413 if months > 3 else 0,
       regress_unit='month'
   )

   # specify some comparison time
   time = nfl_spreads.last_update

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC', bias=.166)

   # mean expected CLE vs KC point differential
   print('CLE VS KC: {}'.format(mean))

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
