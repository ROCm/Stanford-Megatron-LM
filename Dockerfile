FROM rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0Add commentMore actions
WORKDIR /root/

# Install the application dependencies
RUN pip install regex nltk pybind11

RUN git clone https://github.com/ROCm/Stanford-Megatron-LM.git && \
    cd Stanford-Megatron-LM && \
    git checkout rocm_6_3_patch && \
    ./apply_patch.sh

CMD ["/bin/bash"]
