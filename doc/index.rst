melo
====
*Margin-dependent Elo ratings and predictions*


:Author: Scott Moreland
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

   melo = Melo(times, labels1, labels2, values, lines=lines)

.. math::
   y = \sum\limits_{i=1}^5 x

User guide
----------

.. toctree::
   :caption: User guide
   :maxdepth: 2

   install
   usage

.. toctree::
   :caption: Technical info
   :maxdepth: 2

   theory
   tests

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
