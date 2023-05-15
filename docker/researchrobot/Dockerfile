FROM python:3.8.12-slim-buster

VOLUME /cache
VOLUME /radius


RUN mkdir -p /opt/radius/bin
RUN mkdir -p /cache/radius
RUN mkdir /metapack

WORKDIR /opt/radius

RUN apt-get update --allow-releaseinfo-change && \
    apt-get -y install gcc git curl gpg procps libgl1-mesa-glx \
                       libglib2.0-0 libsm6 libxrender1 libxext6  \
                       libpq-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*


ENV PATH /opt/radius/bin:$PATH

ADD requirements.txt .

RUN pip install -r requirements.txt

# Setup access to private github checkout?
#RUN git config --global url."https://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:@github.com/".insteadOf #"https://github.com/"
#
#RUN git clone  https://github.com/CivicKnowledge/demosearch.git #8
#RUN cd /opt/radius/demosearch && python setup.py install

RUN touch ~/.metapack.yaml

COPY demosearch.tar.gz /opt/radius/
COPY radsrv.tar.gz /opt/radius/
ADD _ds_config.yaml /root/.config/demosearch/ds_config.yaml

RUN pip install /opt/radius/demosearch.tar.gz
RUN pip install /opt/radius/radsrv.tar.gz

EXPOSE 8888
EXPOSE 80


WORKDIR /radius

#CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token '' #--NotebookApp.password ''
