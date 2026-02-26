.. meta::
  :description: Stanford Megatron-LM documentation
  :keywords: Stanford Megatron-LM, ROCm, documentation, deep learning, framework, GPU

.. _Stanford Megatron-LM-documentation-index:

********************************************************************
Stanford Megatron-LM on ROCm documentation
********************************************************************

With Stanford Megatron-LM on ROCm, you can train massive transformer LLMs
with data, tensor, and pipeline parallelism on AMD Instinct GPUs, enabling
scale-out to hundreds of billions of parameters for enterprise multilingual
pretraining and domain adaptation.

Stanford Megatron-LM is a large-scale language model training framework developed 
by NVIDIA at `https://github.com/NVIDIA/Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_. 
It is designed to train massive transformer-based language models efficiently by model 
and data parallelism. 

Stanford Megatron-LM on ROCm supports the BERT, GPT, T5, and ICT models, providing efficient tensor,
pipeline, and sequence-based model parallelism for pre-training transformer-based
language models such as GPT (decoder-only), BERT (encoder-only), and T5 (encoder-decoder).
It also offers distributed pre-training, activation checkpointing and recomputation,
a distributed optimizer, and mixture-of-experts support.

Stanford Megatron-LM is part of the `ROCm-LLMExt toolkit
<https://rocm.docs.amd.com/projects/rocm-llm-ext/en/docs-25.08/>`__.

The Stanford Megatron-LM public repository is located at `https://github.com/ROCm/Stanford-Megatron-LM <https://github.com/ROCm/Stanford-Megatron-LM>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install Stanford Megatron-LM <install/stanford-megatron-lm-install>`

  .. grid-item-card:: Reference

      * `API reference (upstream) <https://epfllm.github.io/Megatron-LLM/#api>`__

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing <about/license>` page.
