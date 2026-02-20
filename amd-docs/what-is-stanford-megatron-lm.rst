.. meta::
  :description: What is Stanford Megatron-LM?
  :keywords: Stanford Megatron-LM, documentation, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-Stanford Megatron-LM:

********************************************************************
What is Stanford Megatron-LM?
********************************************************************

Stanford Megatron-LM is a large-scale language model training framework developed 
by NVIDIA at `https://github.com/NVIDIA/Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_. 
It is designed to train massive transformer-based language models efficiently by model 
and data parallelism. 

It provides efficient tensor, pipeline, and sequence-based model parallelism for 
pre-training transformer-based language models such as GPT (Decoder Only), BERT 
(Encoder Only), and T5 (Encoder-Decoder).

Features and use cases
====================================================================

This section details models & features that are supported by Stanford Megatron-LM on ROCm.

Models:

* BERT
* GPT
* T5
* ICT

Features:

* Distributed Pre-training
* Activation Checkpointing and Recomputation
* Distributed Optimizer
* Mixture-of-Experts

Why Stanford Megatron-LM?
====================================================================

* The `Efficient MoE training on AMD ROCm: How-to use Megablocks on AMD GPUs 
  <https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html>`__ 
  blog post guides how to leverage the ROCm platform for pre-training using the 
  Megablocks framework. It introduces a streamlined approach for training Mixture-of-Experts 
  (MoE) models using the Megablocks library on AMD hardware. Focusing on GPT-2, it 
  demonstrates how block-sparse computations can enhance scalability and efficiency in MoE 
  training. The guide provides step-by-step instructions for setting up the environment, 
  including cloning the repository, building the Docker image, and running the training container. 
  Additionally, it offers insights into utilizing the ``oscar-1GB.json`` dataset for pre-training 
  language models. By leveraging Megablocks and the ROCm platform, you can optimize your MoE 
  training workflows for large-scale transformer models.

It features how to pre-process datasets and how to begin pre-training on AMD GPUs through:

* Single-GPU pre-training
* Multi-GPU pre-training