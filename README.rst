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

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
