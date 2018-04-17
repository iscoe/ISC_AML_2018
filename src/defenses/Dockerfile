### Minorly modified keras dockerfile by Neil Fendley

#ARG cuda_version=9.0
#ARG cudnn_version=7
FROM nvidia/cuda:9.0-devel-ubuntu16.04


USER root

RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
libcudnn7=7.0.5.15-1+cuda9.0 \ 
libcudnn7-dev=7.0.5.15-1+cuda9.0 && \
rm -rf /var/lib/apt/lists/*

RUN apt-get update

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin libgl1-mesa-glx && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

USER keras

# Python
ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install tensorflow-gpu && \
    pip install redis && \
    pip install tqdm && \
    pip install Pillow && \ 
    pip install https://cntk.ai/PythonWheel/GPU/cntk-2.1-cp36-cp36m-linux_x86_64.whl && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py opencv && \
    pip install sklearn_pandas && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt


ENV PYTHONPATH='/src/:$PYTHONPATH'

# WORKDIR /src

EXPOSE 2002

# CMD jupyter notebook --port=8888 --ip=0.0.0.0
