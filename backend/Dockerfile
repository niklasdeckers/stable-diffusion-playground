FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# expose
EXPOSE 8080

# set working directory
WORKDIR /app

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install git
RUN apt-get install -y git

# update pip
RUN pip3 install --upgrade pip

# add requirements
COPY ./requirements.txt /app/requirements.txt

# install requirements
RUN pip3 install -r requirements.txt

# add source code
COPY . /app

# run server
CMD python3 app.py --port 8080 --model_path "./model.ckpt"
