FROM python:3.10.7

EXPOSE 5000

WORKDIR /ITDtest
ADD . /ITDtest

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD [ "ITDtest/app.py" ]