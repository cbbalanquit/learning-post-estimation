FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY app.py .

# Install dependencies using uv
RUN uv pip install --system -e .

# Expose the port Streamlit runs on
EXPOSE 8501

# Create directory for model cache
RUN mkdir -p /root/.cache/torch/hub/ultralytics_yolov8_master

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]