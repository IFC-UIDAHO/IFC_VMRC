FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5

WORKDIR /app

# System build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-dev build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Build deps for Fiona against system GDAL
RUN uv pip install --system --no-cache cython numpy setuptools wheel

COPY requirements.txt /app/requirements.txt

# Fiona reports version 0.0.0 when built from source against system GDAL.
# Install without version constraint first, then the rest of requirements.
RUN uv pip install --system --no-cache --no-build-isolation fiona && \
    uv pip install --system --no-cache -r requirements.txt

COPY backend/ /app/backend/

# Create the static directory for overlays
RUN mkdir -p /app/static/overlays

WORKDIR /app/backend

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
