# Use Python 3.9 or 3.10 (Slim version to save space)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .

# Install dependencies
# We add --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create the directory for models if it doesn't exist
RUN mkdir -p models

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the app
# Note: Hugging Face expects the app on port 7860, not 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]