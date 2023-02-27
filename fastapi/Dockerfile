FROM python:3.9

USER root


WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    rm requirements.txt

RUN useradd -ms /bin/bash app
USER app
WORKDIR /code/app

EXPOSE 80
COPY ./ /code/app
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn_conf.py", "api.main:app"]
