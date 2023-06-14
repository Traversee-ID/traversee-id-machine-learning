FROM ubuntu:20.04

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update && \
    apt-get install -y firefox && \
    apt-get install -y firefox-geckodriver && \
    apt-get install -y python3 && \
    apt-get install -y python3-selenium && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]