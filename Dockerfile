FROM ubuntu:18.04

RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
    wget ca-certificates vim

RUN useradd --create-home user
USER user
WORKDIR /home/user

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /home/user/miniconda3/bin:$PATH

COPY setup_env.sh /home/user/bootstrap/

ARG DRLND_ENV
RUN if ! [ -z DRLND_ENV ]; then bash /home/user/bootstrap/setup_env.sh drlnd_docker

ARG DISP_IP
ENV DISPLAY $DISP_IP