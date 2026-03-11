Cherimoya
=========

.. image:: https://img.shields.io/pypi/v/cherimoya.svg
   :target: https://pypi.org/project/cherimoya/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/CUDA-required-green.svg
   :alt: CUDA required

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/jmschrei/cherimoya/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/maintenance-active-brightgreen.svg
   :alt: Maintenance



.. image:: ../imgs/cherimoya.png
   :width: 1000px
   :align: center
   :alt: Cherimoya logo

|

**A lightweight genomic sequence-to-function model.**

Cherimoya predicts genomic modalities — transcription factor binding, chromatin
accessibility, and transcription initiation — from DNA sequence alone. It builds
on concepts from BPNet and ChromBPNet while introducing architectural,
algorithmic, and systems-level improvements for better stability, efficiency,
and performance.

.. admonition:: Under Active Development

   Cherimoya is still evolving and may change in ways that are not backward
   compatible. Please note the version you are using.


Why Cherimoya?
--------------

While popular S2F models like BPNet and ChromBPNet have revolutionized our
ability to interpret regulatory sequences, they often require millions of
parameters and extensive tuning. Cherimoya provides a modern alternative:

* **Efficient Architecture**: Uses significantly fewer parameters while
  maintaining or exceeding state-of-the-art predictive performance.
* **Speed**: Runs much faster on modern GPUs (e.g., H200) thanks
  to  custom Triton kernels that fuse dilated convolutions and layer
  normalization.
* **Automated Tuning**: Replaces manual loss balancing heuristics with
  learned weighting parameters that adapt to your data's signal-to-noise
  characteristics.
* **Modern Optimization**: Leverages the Muon optimizer and dual-optimizer
  strategies to reduce training epochs and improve convergence.

---

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   architecture
   tutorials/cli_pipeline
   tutorials/python_api
   tutorials/attribution
   CHANGELOG

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/model
   api/io
   api/losses
   api/performance
