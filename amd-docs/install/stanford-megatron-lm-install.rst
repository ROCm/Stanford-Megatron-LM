.. meta::
  :description: installing Stanford Megatron-LM for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, Stanford Megatron-LM

.. _stanford-megatron-lm-on-rocm-installation:

********************************************************************
Stanford Megatron-LM on ROCm installation
********************************************************************

System requirements
====================================================================

To use Stanford Megatron-LM `85f95ae <https://github.com/stanford-futuredata/Megatron-LM/commit/85f95aef3b648075fe6f291c86714fdcbd9cd1f5>`__, you need the following prerequisites:

- **ROCm version:** `6.3.0 <https://rocm.docs.amd.com/en/docs-6.3.0/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X
- **PyTorch:** `2.4.0 <https://github.com/ROCm/pytorch/tree/release/2.4>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-3129/>`__

Install Stanford Megatron-LM
================================================================================

To install Stanford Megatron-LM on ROCm, you have the following options:

- :ref:`Use the prebuilt Docker image <use-docker-with-stanford-megatron-lm-pre-installed>` **(recommended)**
- :ref:`Build your own docker image <build-your-stanford-megatron-lm-rocm-docker-image>`

.. _use-docker-with-stanford-megatron-lm-pre-installed:

Use a prebuilt Docker image with Stanford Megatron-LM pre-installed
--------------------------------------------------------------------------------

The recommended way to set up a Stanford Megatron-LM environment and avoid potential installation issues is with Docker. 
The tested, prebuilt image includes Stanford Megatron-LM, PyTorch, ROCm, and other dependencies.

Prebuilt Docker images with Stanford Megatron-LM configured for ROCm 6.3.0 are available on `Docker Hub <https://hub.docker.com/r/rocm/stanford-megatron-lm/tags>`_.

1. Pull the Docker image:

   .. code-block:: bash

      docker pull rocm/stanford-megatron-lm:stanford-megatron-lm85f95ae_rocm6.3.0_ubuntu24.04_py3.12_pytorch2.4.0

2. Launch and connect to the container:

   .. code-block:: bash

      docker run -it --rm \
      --privileged -v ./:/app \
      --network=host --device=/dev/kfd \
      --device=/dev/dri --group-add video \
      --name=my_megatron --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --ipc=host --shm-size 16G \
      rocm/stanford-megatron-lm:stanford-megatron-lm85f95ae_rocm6.3.0_ubuntu24.04_py3.12_pytorch2.4.0

.. _build-your-stanford-megatron-lm-rocm-docker-image:

Build your own Docker image
--------------------------------------------------------------------------------

1. Download a base Docker image with the correct ROCm and PyTorch version

   .. code-block:: bash

      docker pull rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0

2. Start a Docker container using the downloaded image

   .. code-block:: bash
      
      docker run -it \
      --device=/dev/kfd \
      --device=/dev/dri \
      --group-add video \
      rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0

3. Set up dependencies

   .. code-block:: bash

      pip install regex nltk pybind11

4. Git clone the `https://github.com/ROCm/Stanford-Megatron-LM <https://github.com/ROCm/Stanford-Megatron-LM>`_ repository

   .. code-block:: bash

      git clone https://github.com/ROCm/Stanford-Megatron-LM.git
      cd Stanford-Megatron-LM
      git checkout rocm_6_3_patch
      ./apply_patch

5. Install Stanford Megatron-LM

   .. code-block:: bash

      python setup.py install

Set up your datasets
======================================================================================

The following instructions explain how you use the script ``pretrain_gpt.sh`` to run the Megatron-LM training process.
If you are working with other model sizes, the process is similar, but you will need to use a different script.

1. Once inside the Stanford Megatron-LM Directory, you can download the ``BooksCorpus`` dataset or ``Oscar`` dataset 
   by utilizing the helper scripts stored in dataset directory:

   * Oscar dataset
   * gpt2-vocab.json
   * gpt2-merges.txt

   .. code-block:: bash

      cd tools
      wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
      xz -d oscar-1GB.jsonl.xz
      wget -O gpt2-vocab.json https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
      wget -O gpt2-merges.txt https://huggingface.co/mkhalifa/gpt2-biographies/resolve/main/merges.txt


2. Pre-process the datasets 

   .. code-block:: bash

      export BASE_SRC_PATH=$(dirname $(pwd))
      export BASE_DATA_PATH=${BASE_SRC_PATH}/tools
      python preprocess_data.py \
      --input ${BASE_DATA_PATH}/oscar-1GB.jsonl \
      --output-prefix ${BASE_DATA_PATH}/my-gpt2 \
      --vocab-file ${BASE_DATA_PATH}/gpt2-vocab.json \
      --dataset-impl mmap \
      --tokenizer-type GPT2BPETokenizer \
      --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
      --append-eod \
      --workers 8 \
      --chunk-size 10

3. Set up the pre-training script 

   Edit the ``pretrain_gpt.sh`` file so that it points to the correct locations.

   .. code-block:: bash

      DATA_PATH=${BASE_DATA_PATH}/my-gpt2_text_document
      CHECKPOINT_PATH=${BASE_DATA_PATH}/checkpoints
      VOCAB_FILE=${BASE_DATA_PATH}/gpt2-vocab.json
      MERGE_FILE=${BASE_DATA_PATH}/gpt2-merges.txt

4. Run the pre-training script

   From the project root directory, execute the following command:

   .. code-block:: bash

      ./examples/pretrain_gpt.sh 

Test the Stanford Megatron-LM installation
======================================================================================

Stanford Megatron-LM unit tests are optional for validating your installation if you used a
prebuilt Docker image from AMD ROCm Docker Hub.

To run unit tests manually and validate your installation fully, follow these steps:

.. note::

   Run the following from the Stanford Megatron-LM root. Running the unit tests requires 8 GPUs as they test for distributed functionality. 


.. code-block:: bash

   torchrun --nproc_per_node=8 -m pytest tests/ 2>&1 | tee "stanford_megatron_lm_tests.log"


Run a Stanford Megatron-LM example
======================================================================================

Use the example command ``pretrain_gpt.sh`` from the repository at `https://github.com/ROCm/Stanford-Megatron-LM/tree/main/examples <https://github.com/ROCm/Stanford-Megatron-LM/tree/main/examples>`__.

For detailed steps, refer to the `Efficient MoE training on AMD ROCm <https://rocm.blogs.amd.com/artificial-intelligence/megablocks/README.html>`__ blog post.
The blog mentions Megablocks, but you can run Stanford Megatron-LM with the same steps.