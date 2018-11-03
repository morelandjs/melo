MELO
====

*Margin-dependent Elo ratings and predictions model*

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
   times, home, away, spread = zip(*[l.split() for l in pkgdata])

   # specify values for the model training parameters
   melo = Melo(
       times, home, away, spread, lines=np.arange(-50.5, 51.5),
       mode='Fermi', k=.245, bias=.166,
       decay=lambda t: 1 if t < timedelta(weeks=20) else .597
   )

   # specify some 'current' time
   time = melo.comparisons['time'][-1]

   # predict the mean outcome at that time
   mean = melo.mean(time, 'CLE', 'KC')
   print('CLE VS KC: {}'.format(mean))
