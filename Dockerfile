# 1. Start with a lightweight, official Linux Python environment
FROM python:3.11-slim

# 2. Create a folder inside the container called /app
WORKDIR /app

# 3. Copy only the requirements first (this makes rebuilding much faster)
COPY requirements.txt .

# 4. Install all the AI and Web Server libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files into the container
COPY . .

EXPOSE 8000

# NEW: 7. Docker Health Check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# 8. Start the FastAPI server when the container boots up
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]