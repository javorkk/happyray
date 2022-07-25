FROM nvidia/cuda:8.0-devel

LABEL org.label-schema.name="happyray-dev"

RUN apt-get update &&\
    apt-get install -y libpng-dev libsdl2-dev
