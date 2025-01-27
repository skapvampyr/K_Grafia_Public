FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

COPY app/common/requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the requirements.txt file into the container
COPY app/ .

# Expose the port on which the application will run
EXPOSE 8000

# gunicorn
CMD exec uvicorn server:app --host 0.0.0.0 --port 8000
