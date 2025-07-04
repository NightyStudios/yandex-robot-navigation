FROM python:3.10-slim-bullseye

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
#EXPOSE 8000

CMD ["cd .."]
CMD [ "python3", "backend/run.py" ]
