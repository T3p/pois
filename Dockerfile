# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:latest-py3

# Set the working directory to /app
WORKDIR /baselines

# Copy the current directory contents into the container at /app
COPY ./baselines /baselines
