# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>


"""
This module contains the mixture loss function used for training Cherimoya
models, which is comprised of a multinomial log likelihood component and a
mean-squared error component. These losses are provided independently, so
other code can implement different ways of combining them into a single loss.
"""

import torch

from bpnetlite.losses import MNLLLoss
from bpnetlite.losses import log1pMSELoss


def _mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None):
	"""A function that takes in predictions and truth and returns the loss.
	
	This function takes in the observed integer read counts, the predicted logits,
	and the predicted logcounts, and returns the total loss. Importantly, this
	calculates a single multinomial over all strands in the tracks and a single
	count loss across all tracks.
		
	
	Parameters
	----------
	y: torch.Tensor, shape=(n, n_outputs, length)
		The observed counts for each example across each strand/output and at each
		position. This should likely be sparse integers.

	y_hat_logits: torch.Tensor, shape=(n, n_outputs, length)
		The predicted *logits* for each example across each strand/output and at
		each position. This will be normalized internally, so DO NOT run a softmax
		on your model.

	y_hat_logcounts: torch.Tensor, shape=(n, n_outputs)
		The predicted *log counts* for each example across each strand/output. The
		true log counts will be derived automatically from `y`.


	labels: torch.Tensor, shape=(n,), optional
		Whether the example is from a peak (1) or a non-peak (0). If provided, the
		profile loss will only be calculated on the peak examples. The count loss
		will always be calculated on the entire set of examples. If not provided,
		the profile loss will also be calculated on the entire set of examples.
		Default is None.
		

	Returns
	-------
	profile_loss: torch.Tensor, shape=(1,)
		The multinomial log likelihood loss averaged across examples and outputs.

	count_loss: torch.Tensor, shape=(1,)
		The mean-squared error loss, averaged across examples and outputs.
	"""
	
	y_hat_logits = y_hat_logits.reshape(y_hat_logits.shape[0], -1)
	y_hat_logits = torch.nn.functional.log_softmax(y_hat_logits, dim=-1)
	
	y = y.reshape(y.shape[0], -1)
	y_ = y.sum(dim=-1).reshape(y.shape[0], 1)

	# Calculate the profile and count losses
	if labels is not None:
		profile_loss = MNLLLoss(y_hat_logits[labels == 1], y[labels == 1]).mean()
	else:
		profile_loss = MNLLLoss(y_hat_logits, y).mean()

	count_loss = log1pMSELoss(y_hat_logcounts, y_).mean()	
	return profile_loss, count_loss
