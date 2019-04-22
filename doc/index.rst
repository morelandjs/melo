melo
====
*Margin-dependent Elo ratings and predictions*


:Author: J\. Scott Moreland
:Language: Python
:Source code: `github:morelandjs/melo <https://github.com/morelandjs/melo>`_

``melo`` generalizes the `Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`_ paired comparison model beyond binary outcomes to include margin-of-victory information. It does this by "redefining what it means to win" using a variable handicap to shift the threshold of paired comparison. The framework is general and has numerous applications in ranking, estimation, and time series prediction.


Quick start
-----------
Requirements: Python 2.7 or 3.3+ with numpy_ and scipy_.

Install the latest release with pip_::

   pip install melo

Basic usage:

.. code-block:: python

   from melo import Melo

   # run the model on the list of paired comparisons
   melo = Melo(times, labels1, labels2, comparisons, lines=lines)

   # probability that comparison(label1, label2) > 0
   win_prob = melo.probability(time, label1, label2, lines=0)

   # probability that comparison(label1, label2) > 5
   cover_prob = melo.probability(time, label1, label2, lines=5)

   # interquartiles of the predicted comparison distribution
   interquartiles = melo.percentiles(time, label1, label2, q=[25, 50, 75])

   # ranked labels
   ranked_labels = melo.rank(time, statistic='mean')

.. toctree::
   :caption: User guide
   :maxdepth: 2

   usage
   examples

.. toctree::
   :caption: Technical info
   :maxdepth: 2

   theory
   tests

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
