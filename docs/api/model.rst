Model
=====

.. module:: cherimoya.cherimoya

Cherimoya
---------

.. autoclass:: Cherimoya
   :members: forward, fit
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__


CheriBlock
----------

.. autoclass:: CheriBlock
   :members:
   :undoc-members:
   :show-inheritance:


FusedDilatedConvNormFunc
------------------------

.. autoclass:: FusedDilatedConvNormFunc
   :members: forward, backward
   :undoc-members:

   A custom ``torch.autograd.Function`` that fuses the dilated depthwise
   convolution and layer normalization into a single Triton GPU kernel.

   This is an internal implementation detail and should not need to be called
   directly. It is used inside :class:`CheriBlock`.
