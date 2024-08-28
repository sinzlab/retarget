FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /src