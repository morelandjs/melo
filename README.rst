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
Install::

   pip install melo

Basic usage:

.. code-block:: python

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
       regress=lambda months: .413*(months > 3), regress_unit='month'
   )

   # specify some comparison time
   time = nfl_spreads.last_update

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC', bias=.166)

   # mean expected CLE vs KC point differential
   print('CLE VS KC: {}'.format(mean))
