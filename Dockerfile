FROM python:3.12-slim

WORKDIR /app

# opencv-headless ve torch icin gerekli minimal sistem kutuphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# once bagimliliklar (katman onbellegi icin ayri)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# sonra kod + egitilmis model
COPY . .

# varsayilan: test setinde degerlendirme (tekrar uretilebilir cikti)
CMD ["python", "degerlendir.py"]