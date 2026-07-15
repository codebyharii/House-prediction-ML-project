# Ye file Hugging Face ko batati hai ki app kaise chalani hai

# Python ka ready-made environment le lo
FROM python:3.11-slim

# app ke liye ek folder banao
WORKDIR /app

# pehle requirements copy karke install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ab baaki saari files (app.py, index.html, model, csv) copy karo
COPY . .

# Hugging Face app ko port 7860 pe expect karta hai
EXPOSE 7860

# app chalu karo (gunicorn ek strong server hai production ke liye)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
