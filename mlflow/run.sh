#!/bin/sh

docker run -p 5000:5000 -d --name mlflow ghcr.io/mlflow/mlflow:latest mlflow ui -h 0.0.0.0