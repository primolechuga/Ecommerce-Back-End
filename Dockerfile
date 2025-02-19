FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8080:8080


CMD [ "python", "app.main.py" ]