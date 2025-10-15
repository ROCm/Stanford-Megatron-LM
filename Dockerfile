FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1
WORKDIR /root/

# Install the application dependencies
RUN pip install regex nltk pybind11

RUN apt-get update && apt-get install -y ninja-build

RUN rm -rf Stanford-Megatron-LM && \
    git clone https://github.com/ROCm/Stanford-Megatron-LM.git && \
    cd Stanford-Megatron-LM && \
    git checkout users/peizhang56/rocm7-fix && \
    ./apply_patch.sh

CMD ["/bin/bash"]
