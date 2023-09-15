FROM python:3.7

RUN mkdir /app

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install plotly

EXPOSE 5000

CMD ["python", "app.py"]