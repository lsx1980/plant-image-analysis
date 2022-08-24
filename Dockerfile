#Name: Dockerfile
#Version: 1.0
#Summary: Docker recipe file for smart pipeline
#Author: suxing liu
#Author-email: suxingliu@gmail.com
#Created: 2022-10-29

#USAGE:
#docker build -t plant_test -f Dockerfile .
#docker run -v /path to test image:/images -it plant_test
#cd /opt/smart/
#python3 trait_computation_mazie_ear.py -p /images/ -ft png
#python3 trait_computation_maize_tassel.py -p /images/ -ft png


FROM ubuntu:20.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/smart


RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3 \
    python3-numexpr \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    cmake-gui \
    nano \
    libdmtx0b


RUN pip3 install --upgrade pip 
RUN pip3 install numpy \
    Pillow \
    scipy \
    scikit-image \
    scikit-learn \
    matplotlib \
    pandas \
    pytest \
    opencv-python-headless \
    openpyxl \
    imutils \
    numba \
    skan \
    tabulate \
    pylibdmtx \
    psutil \
    natsort \
    pathlib \
    pandas 





