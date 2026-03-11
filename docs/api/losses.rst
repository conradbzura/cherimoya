Losses
======

.. module:: cherimoya.losses


_mixture_loss
-------------

.. autofunction:: _mixture_loss


Imported Losses
---------------

The following loss functions are imported from ``bpnetlite`` and used
internally:

.. function:: MNLLLoss(logps, true_counts)

   Multinomial negative log-likelihood loss. Computes the negative log
   probability of the observed counts under a multinomial distribution
   parameterized by the predicted log probabilities.

   :param logps: Predicted log probabilities, shape ``(n, length)``
   :param true_counts: Observed integer counts, shape ``(n, length)``
   :returns: Loss per example, shape ``(n,)``

.. function:: log1pMSELoss(pred_log_counts, true_counts)

   Mean squared error in log space. Computes ``MSE(pred, log(true + 1))``.

   :param pred_log_counts: Predicted log counts, shape ``(n, n_outputs)``
   :param true_counts: True counts (not in log space), shape ``(n, n_outputs)``
   :returns: Loss per example, shape ``(n,)``
