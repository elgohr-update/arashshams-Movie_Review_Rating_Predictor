# Docker file for the Movie Review Rating Predictor
# Authors: Yuanzhe Marco Ma, Arash Shamseddini, Kaicheng Tan, Zhenrui Yu
# Nov.-Dec. 2020

FROM continuumio/miniconda3:4.9.2

# Install system dependencies. (e.g. make)
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        graphviz

# Install python dependencies via conda
RUN conda install -y -c conda-forge \
    altair=4.1.* \
    altair_saver=0.5.* \
    docopt=0.6.* \
    jupyterlab=2.2.* \
    nltk=3.4.* \
    numpy=1.19.* \
    pandas=1.1.* \
    scikit-learn=0.23.* \
    pandas-profiling=2.9.*

# Install makefile2graph via github
RUN git clone https://github.com/lindenb/makefile2graph.git && \
    make -C makefile2graph/. && \
    cp makefile2graph/makefile2graph usr/bin && \
    cp makefile2graph/make2graph usr/bin && \
    rm -rf makefile2graph

# Create and enter mount point
VOLUME /home/source
WORKDIR /home/source
