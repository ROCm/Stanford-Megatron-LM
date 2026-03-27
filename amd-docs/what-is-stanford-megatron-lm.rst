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

Stanford Megatron-LM provides the following key features:

- **Scalable Parallelism:** Employs data, tensor, and pipeline parallelism
  to train massive transformer models efficiently across multi-GPU and
  multi-node clusters on ROCm.

- **Memory Efficiency:** Uses mixed precision (FP16/BF16), activation
  recomputation, gradient accumulation, and optimizer sharding to reduce
  memory footprint while maintaining throughput.

- **Fused Transformer Kernels:** Leverages optimized attention and MLP
  kernels to improve training speed, with support for large context lengths
  and efficient KV cache management during evaluation.

- **Modular Components:** Offers modular building blocks for GPT-style
  and encoder-decoder architectures, enabling reuse across pretraining,
  fine-tuning, and evaluation pipelines.

- **Robust I/O and Checkpointing:** Provides dataset streaming utilities,
  efficient sharded checkpointing, and resumable training for long-running
  jobs on distributed clusters.

Stanford Megatron-LM is commonly used in the following scenarios:

- **Large-Scale Pretraining:** Train dense transformer LLMs at enterprise
  scale for multilingual and domain-specific corpora.

- **Fine-Tuning and Adaptation:** Perform supervised fine-tuning and
  alignment-focused training on specialized datasets.

- **Research at Scale:** Experiment with parallelism strategies, deep
  architectures, and long-context configurations on AMD Instinct GPUs.

- **Evaluation and Benchmarking:** Run evaluation suites for throughput,
  scaling efficiency, and accuracy across different model sizes and
  hardware topologies.

Why Stanford Megatron-LM?
====================================================================

Stanford Megatron-LM is well suited for large-scale training for the following reasons:

- Its **multi-dimensional parallelism** enables efficient scaling
  from single-node to multi-rack clusters while keeping utilization high.

- **Training efficiency features** like mixed precision and activation
  recomputation reduce memory and cost without sacrificing performance.

- **Modular architecture** eases integration with existing data pipelines,
  optimizers, and evaluation workflows.

- **Production readiness** through robust checkpointing and orchestration
  utilities supports long-running jobs on ROCm-powered clusters.