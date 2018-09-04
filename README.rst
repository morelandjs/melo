MELO
====

*Margin-dependent Elo ratings and predictions model*

Documentation
-------------


Quick start
-----------
Install::

   pip install melo

Basic usage:

.. code-block:: python

   from melo import Melo

   # evaluate a list of binary comparisons
   lines = [-30.5, -29.5, ..., 29.5, 30.5]
   melo = Melo(times, labels1, labels2, values, lines=lines)

   # predict the distribution of sampled values
   lines, distribution = melo.predict(time, label1, label2)
