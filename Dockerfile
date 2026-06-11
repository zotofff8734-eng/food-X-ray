# Stage 1: Build with a full-featured base image to get dependencies right
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - ffmpeg is required for audio processing (Whisper, STT)
# - tesseract-ocr is the engine for pytesseract (though we mostly use EasyOCR)
# - git is needed to install whisper from github
RUN apt-get update && apt-get install -y 
    ffmpeg 
    tesseract-ocr 
    git 
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# The bot will be started by this command.
# The TELEGRAM_BOT_TOKEN must be passed as an environment variable at runtime.
# Example: docker run -e TELEGRAM_BOT_TOKEN="your_token_here" food-x-ray-bot
CMD ["python", "bot.py"]
