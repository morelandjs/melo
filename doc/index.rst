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

Example usage:

.. code-block:: python

   from itertools import combinations
   import matplotlib.pyplot as plt
   import numpy as np
   from scipy.stats import skellam

   from melo import Melo

   # create time series of comparison data by pairing and
   # substracting 100 different Poisson distributions
   mu_values = np.random.randint(80, 110, 100)
   mu1, mu2 = map(np.array, zip(*combinations(mu_values, 2)))
   labels1, labels2 = [mu.astype(str) for mu in [mu1, mu2]]
   spreads = skellam.rvs(mu1=mu1, mu2=mu2)
   times = np.arange(spreads.size).astype('datetime64[s]')

   # MELO class arguments (explained in usage)
   lines = np.arange(-59.5, 60.5)
   k = .15

   # train the model on the list of comparisons
   melo = Melo(times, labels1, labels2, spreads, lines=lines, k=k)

   # predicted and true (analytic) comparison values
   pred_times = np.repeat(melo.last_update, times.size)
   pred = melo.mean(pred_times, labels1, labels2)
   true = skellam.mean(mu1=mu1, mu2=mu2)

   # plot predicted 'means' versus true 'means'
   plt.scatter(pred, true)
   plt.plot([-20, 20], [-20, 20], color='k')
   plt.xlabel('predicted mean')
   plt.ylabel('true mean')
   plt.show()

.. figure:: _static/quickstart_example.png
   :alt: predicted mean versus exact mean

   Model estimated mean plotted against the true mean for the expected difference of each pair of Poisson variables. Diagonal line indicates perfect agreement.

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
