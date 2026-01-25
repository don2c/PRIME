FROM python:3.10-slim

WORKDIR /work
COPY . /work

RUN pip install -U pip \
    && pip install -r requirements.txt

CMD ["bash", "scripts/run_all.sh"]
