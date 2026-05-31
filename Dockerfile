FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if required for RDKit/ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
COPY requirements.txt .

# Since we prefer minimal dependencies, use pip primarily
# Note: if using conda, install miniconda here instead.
# For this dockerfile, assuming pip install.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run_ifg26_benchmark.py"]
