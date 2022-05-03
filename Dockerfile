FROM ubuntu:18.04

LABEL maintainer="Suxing Liu, Wes Bonelli"

COPY . /opt/smart

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3-numexpr \
    python3-distutils \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    apt-utils

RUN pip3 install --upgrade pip && \
    pip3 install -e /opt/smart

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
