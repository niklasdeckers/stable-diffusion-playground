FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

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
