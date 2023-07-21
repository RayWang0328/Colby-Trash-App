FROM python:3.8.5

WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Installing ffmpeg libsm6 libxext6
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install GroundingDINO
WORKDIR /app/GroundingDINO
RUN pip install -e .

# Go back to app directory
WORKDIR /app

# Create and enter the weights folder
WORKDIR /app/weights
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Go back to app directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Run the command when the container launches
CMD ["python", "app.py"]
