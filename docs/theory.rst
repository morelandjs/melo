.. _theory:

Theory
======

The margin-dependent Elo model is introduced in preprint `arXiv:1802.00527 <https://arxiv.org/abs/1802.00527>`_.
The following is a short primer on the subject.

Introduction
------------

The Elo rating system is a method for predicting the outcome of paired comparisons in win/loss games such as chess.
Players (or teams) are assigned a rating :math:`R` which encodes their true skill level.
By convention, larger ratings correspond to higher skill levels and smaller ratings to lower skill levels.

Consider two players named "Lisa" and "Mark" which play some win/loss game.
Suppose Lisa has an Elo rating `R_\text{Lisa}` and Mark has a rating `R_\text{Mark}`.
The probability that Lisa beats Mark at the game is given by

.. math::

   P_\text{pred}(\text{Lisa} > \text{Mark}) = F(R_\text{Lisa} - R_\text{Mark}),

where `F` is a cumulative distribution function (CDF), typically assumed to be the logistic or normal distribution CDF with location zero and arbitrary scale parameter.
Here I'm slightly abusing the notation of the inequality operator to indicate "Lisa beats Mark".

If Lisa beats Mark, she takes some of his rating and adds it to her own (and *vice versa*).
The amount of rating that Lisa steals when she wins (or forfeits when she loses) is related to her rating difference relative to Mark,

.. math::

   \Delta R_\text{Lisa} = k~(P_\text{obs} - P_\text{pred}),

where `P_\text{obs} = 1` if Lisa wins and `P_\text{obs} = 0` if she loses, while `P_\text{obs} = 0.5` is often used if both players tie.
Here `k` is a free parameter which controls how dramatically the ratings respond to each game outcome.

Following this prescription, one can easily calculate the Elo ratings for multiple players in a league and use those ratings to rank the players and calculate their relative win probabilities.

Moving beyond win/loss outcomes
-------------------------------

The traditional Elo rating system is only defined for win/loss games where each outcome is binary (or tertiary if there are ties).
The margin-dependent Elo model extends the traditional Elo framework to account for margin-of-victory information.

The key insight is to realize that the definition of "winning" is arbitrary.
Typically we associate winning with scoring more points than the opponent but that need not be the case.
Suppose, as before, that Lisa and Mark play each other in a game, but this time the game is point-based.

Imagine an operator :math:`\mathcal{C}` which performs a comparison on an ordered tuple `(\text{Lisa}, \text{Mark})` and evaluates to True or False.
In the traditional Elo system, this operator merely determines the winner of the game

.. math::

   \mathcal{C}(\text{Lisa}, \text{Mark}) \equiv (\text{Lisa} > \text{Mark}).

Meanwhile, for a point-based game, the analogous comparison operator is

.. math::

   \mathcal{C}(\text{Lisa}, \text{Mark}) \equiv (S_\text{Lisa} > S_\text{Mark})

where `S_\text{Lisa}` and `S_\text{Mark}` are Lisa and Mark's respective scores.
This comparison operator defines "winning" as outscoring the opponent.

Alternatively, we can imagine a comparison operator :math:`\mathcal{C}_\ell` that defines winning subject to a variable handicap `\ell`:

.. math::

   \mathcal{C}_\ell(\text{Lisa}, \text{Mark}) \equiv (S_\text{Lisa} - S_\text{Mark} > \ell).

Imagine now that I insist that this is the correct criteria to determine the winner of each game (after all the rules are arbitrary), and I ask you to compute the game's Elo ratings as before subject to this criteria.

This comparison operator essentially splits one fair game into two biased games: in one version of the game Lisa is handicapped by `\ell` points and in the other she is advantaged by `\ell` points.
Consequently, we'll need two Elo ratings for every value of `\ell`: a handicapped rating `R_\text{hcap}(\ell)` and an advantaged rating `R_{adv}(\ell)`.
However, since being handicapped by `\ell` is equivalent to being advantaged by `-\ell`, we'll only need a single margin-dependent rating `R(\ell)`.

The predicted probability of the comparison `\mathcal{C}_\ell` is therefore given by

.. math::

   P_\text{pred}(\ell) = F[R_\text{Lisa}(\ell) - R_\text{Mark}(-\ell)].

Hence Lisa's rating change is given by

.. math::

   \Delta R_\text{Lisa}(\ell) = k~[P_\text{obs}(\ell) - P_\text{pred}(\ell)],

where `P_\text{obs}(\ell) = \Theta[\ell - (S_\text{Lisa} - S_\text{Mark})]` and `\Theta` is the Heaviside step function.
The points lost by the handicapped rating are absorbed by the advantaged rating so

.. math::

   \Delta R_\text{Mark}(-\ell) = -\Delta R_\text{Lisa}(\ell)

And that's it!
We've generalized the Elo model to point based games.
In practice, the ratings `R(\ell)` are discretized by calculating their values at several values of the lines

.. math::

   R(\ell) \to [R(\ell_\text{min}), R(\ell_\text{min} + \Delta \ell), \dots, R(\ell_\text{max})].

.. note::

   In the traditional Elo rating system, every rating is initialized to the same value.
   For the margin-dependent Elo model, the ratings vector is initialized so the model predicts minimum-bias point spreads for each game.

Commutative comparisons
-----------------------

The previous discussion describes the margin-dependent Elo model for point spreads (points scored minus points allowed).
This particular observable anti-commutes under label interchange, i.e\.

.. math::

   \text{Spread}(\text{Lisa}, \text{Mark}) = -\text{Spread}(\text{Mark}, \text{Lisa}).

It's also possible to estimate Elo ratings and predictions for point totals (points scored plus points allowed) which *commute* under label interchange

.. math::

   \text{Total}(\text{Lisa}, \text{Mark}) = \text{Total}(\text{Mark}, \text{Lisa}).

The model requires two small adjustments.
The predicted probability becomes the *sum* of each rating

.. math::

   P_\text{pred}(\ell) = F[R_\text{Lisa}(\ell) + R_\text{Mark}(\ell)],

and Lisa's rating change becomes equal to Mark's,

.. math::

   \Delta R_\text{Lisa}(\ell) = \Delta R_\text{Mark}(\ell).

Note, this means that the game outcomes (and ratings) are no longer zero sum.
Both competitors can cover a given point total line with higher probability.
