FROM nvidia/cuda:10.2-runtime-ubuntu18.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y \
    ffmpeg \
    python-pyglet \
    python-opengl \
    libpq-dev \
    libjpeg-dev \
    libboost-all-dev \
    libsdl2-dev \
    curl \
    cmake \
    swig \
    wget \
    unzip \
    git \
    xvfb \
    x11vnc \
    ratpoison \
    xterm \
    python-tk \
    python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

RUN apt-get update \
    && apt-get install -y \
    libffi-dev \
    libosmesa6-dev \ 
    libglfw3 \ 
    libglew2.0 \
    libgl1-mesa-glx \
    libosmesa6\
    patchelf \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

#mujoco 200
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && cp -r /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip
#Assume mjkey.txt is in current dir
#openai gym
COPY ./mjkey.txt /root/.mujoco/
#dm control suite
COPY ./mjkey.txt /root/.mujoco/mujoco200_linux/bin/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

#Assume requirement.txt is in current dir
WORKDIR /workspace   
COPY ./requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt
