FROM python:3.10-slim-buster 

ARG S3_ACCESS_KEY
ARG S3_SECRET_KEY 

RUN apt update && \
apt install -y gcc && \
apt install -y git && \
apt-get install build-essential -y

WORKDIR /app
RUN mkdir -p /app
COPY ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r  /app/requirements.txt \
    && pip3 install torch --index-url https://download.pytorch.org/whl/cpu\
    && pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers \
    && apt-get purge -y  \
    -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/* \
        /tmp/*

COPY ./ /app
# COPY ./s3_clone.sh /app
# COPY ./s3_models.txt /app
# COPY ./get_model_from_s3.py /app
RUN ls -lah /app
RUN sh /app/s3_clone.sh
RUN cd /app/s3_wrapper && pip3 install -e ./
RUN python /app/get_model_from_s3.py
RUN  pip install --upgrade pip
RUN apt-get autoremove -y
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8989", "--backlog", "8", "--limit-concurrency", "8"] 