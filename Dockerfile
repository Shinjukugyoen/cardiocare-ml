# Lightweight Python 3.10 base (matches the training environment).
FROM python:3.10-slim

# Work inside /app in the container.
WORKDIR /app

# Install ONLY the inference dependencies (keeps the image small).
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy the inference code, the trained model, and a sample input batch.
COPY src/inference.py ./src/inference.py
COPY models/ ./models/
COPY data/sample_input.csv ./data/sample_input.csv

# Run inference on the sample batch by default.
ENTRYPOINT ["python", "src/inference.py"]
CMD ["data/sample_input.csv"]