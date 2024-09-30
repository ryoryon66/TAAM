FROM ubuntu:22.04

RUN apt update
RUN apt install -y python3.10
RUN apt install -y python3-pip

COPY . ./TAAM
WORKDIR /TAAM

RUN apt-get update
RUN apt-get install graphviz graphviz-dev -y
RUN pip3 install -r requirements.txt

CMD ["pytest"]
