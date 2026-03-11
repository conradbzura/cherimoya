Architecture
============

Cherimoya is a compact convolutional architecture for predicting genomic
modalities from DNA sequence. It builds on the ConvNeXt design philosophy,
adapting it to the challenges of noisy high-throughput genomics experiments.


Model Overview
--------------

.. image:: ../imgs/cheri-model.png
   :align: center
   :alt: Cherimoya model architecture

|

The model consists of three stages:

1. **Input convolution**: A 1D convolution (kernel size 19) maps the one-hot
   encoded DNA sequence (4 channels) into a higher-dimensional feature space.

2. **Cheri Blocks**: A stack of blocks with exponentially increasing dilation
   rates (1, 2, 4, 8, ...) that progressively expand the receptive field. The
   default configuration uses 9 blocks with 64 filters.

3. **Output heads**: Separate heads for profile prediction (a 1D convolution
   with kernel size 75) and count prediction (a linear layer over the
   mean-pooled features).


The Cheri Block
---------------

.. image:: ../imgs/cheri-block.png
   :align: center
   :alt: Cheri Block architecture

|

Each Cheri Block performs the following operations:

1. **Dilated depthwise convolution** — aggregates spatial information
   independently for each channel, with a kernel size of 3 and increasing
   dilation rates.

2. **Layer normalization** — stabilizes activations. The convolution and
   normalization are fused into a single custom Triton GPU kernel for
   efficiency.

3. **Expansion projection** — a linear layer projects from ``n_filters`` to
   ``2 × n_filters`` dimensions.

4. **GELU activation** — the approximate ``tanh``-based variant.

5. **Contraction projection** — projects back to ``n_filters`` dimensions.

6. **Residual connection with learned scaling** — the output is scaled by a
   learnable per-channel vector ``γ`` (initialized to a small value ``ε``) and
   added back to the input. This ensures the residual path starts near-identity
   for training stability.

In code:

.. code-block:: python

   def forward(self, X):
       X_conv = FusedDilatedConvNormFunc.apply(X, self.conv_weight, self.dilation)
       X_mlp = self.linear2(self.activation(self.linear1(X_conv)))
       return X + X_mlp * self.gamma


Custom Triton Kernel
--------------------

The dilated depthwise convolution and layer normalization are fused into a
custom Triton kernel (``FusedDilatedConvNormFunc``) with both forward and
backward passes. This fusion eliminates intermediate memory allocations and
achieves **~2–3× speedup** over the native PyTorch implementation.

The kernel is autotuned across:

- Number of warps: 4, 8, 16
- Number of pipeline stages: 2, 3, 4, 5
- Block sizes: 32, 64, 128, 256


Loss Function Design
--------------------

Cherimoya uses a two-component loss:

- **Profile loss**: Multinomial negative log-likelihood (MNLL) over the
  base-pair resolution profile predictions.
- **Count loss**: Mean squared error in log-space (``log1pMSE``) between
  predicted and observed total counts.

These are combined using **learned weighting parameters** (``lw0``, ``lw1``)
rather than fixed hyperparameters:

.. code-block:: python

   w0 = 1.0 / (2.0 * self.lw0 ** 2)
   w1 = 1.0 / (2.0 * self.lw1 ** 2)
   loss = w0 * profile_loss + w1 * count_loss

The weights are automatically frozen once their gradients become negligible,
preventing further unnecessary updates.


Training Strategy
-----------------

Cherimoya uses a **dual-optimizer** approach:

- **Muon optimizer** for 2D projection weights (the ``linear1`` and ``linear2``
  layers in each Cheri Block)
- **AdamW optimizer** for all other parameters (convolutions, biases, scaling
  vectors)

Both optimizers use a **warmup + cosine decay** learning rate schedule:

- 5 epochs of linear warmup (from 1% of the target learning rate)
- Cosine annealing to ``1e-5`` over the remaining epochs


