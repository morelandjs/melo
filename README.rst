MELO
====

*Margin-dependent Elo ratings and predictions model*
.. image:: https://travis-ci.org/morelandjs/melo.svg?branch=master
    :target: https://travis-ci.org/morelandjs/melo

Documentation
-------------

Work in progress...


Quick start
-----------
Install::

   pip install melo

Basic usage:

.. code-block:: python

   from datetime import timedelta
   import pkgutil

   import numpy as np

   from melo import Melo


   # the package comes pre-bundled with an example dataset
   pkgdata = pkgutil.get_data('melo', 'nfl_scores.dat').splitlines()
   time, home, away, spread = zip(*[l.split() for l in pkgdata])

   # specify values for the model training parameters
   nfl_spreads = Melo(
       time, home, away, spread, lines=np.arange(-50.5, 51.5),
       mode='Fermi', k=.245, bias=.166,
       decay=lambda t: 1 if t < timedelta(weeks=20) else .597
   )

   # specify some 'current' time
   current_time = nfl_spreads.comparisons['time'][-1]

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(current_time, 'CLE', 'KC')

   # mean expected CLE vs KC point differential
   print('CLE VS KC: {}'.format(mean))
