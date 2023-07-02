FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# expose
EXPOSE 8080

# set working directory
WORKDIR /app

# install git, the c compiler for compiling the model, and cv2 dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git build-essential ffmpeg libsm6 libxext6

# add requirements
COPY ./requirements.txt /app/requirements.txt

# install requirements
RUN pip3 install -r requirements.txt

# add source code
COPY . /app

# run server
CMD python3 app.py --port 8080 --model_path "stabilityai/stable-diffusion-xl-base-0.9"
