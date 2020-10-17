FROM python:3.8.0-slim
COPY . /work
WORKDIR /work

EXPOSE 5000
RUN pip install --no-cache-dir -r requirements.txt

CMD gunicorn --workers 2 --log-level info --timeout 90 --bind 0.0.0.0:5000 app:app
