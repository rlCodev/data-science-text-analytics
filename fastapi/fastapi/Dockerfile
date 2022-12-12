FROM python:3.9

USER root

# /app has already been defined and set as workdir in the base-image
ADD requirements.txt /app
RUN GIT_SSL_NO_VERIFY=1 pip3 install -U -r requirements.txt && \
    rm requirements.txt

EXPOSE 5000
USER app
ADD . /app
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn_conf.py", "api.main:app"]
