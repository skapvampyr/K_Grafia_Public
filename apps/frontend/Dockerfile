FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY common/requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the requirements.txt file into the container

COPY common/ .
COPY pages pages
COPY Home.py .


# Expose the port on which the application will run
EXPOSE 8501

# gunicorn
CMD ["streamlit", "run", "Home.py"]