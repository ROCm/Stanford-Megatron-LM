FROM rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0
WORKDIR /root/

# Install the application dependencies
RUN pip install regex nltk pybind11

COPY . /root/Stanford-Megatron-LM

RUN cd /root/Stanford-Megatron-LM && \
    git checkout rocm_6_3_patch && \
    ./apply_patch.sh

