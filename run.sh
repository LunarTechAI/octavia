#!/bin/bash
set -e

echo "Checking for Docker..."
if ! command -v docker &> /dev/null; then
  echo "Docker not found. Please install Docker."
  exit 1
fi

echo "Building Docker image..."
docker-compose build

echo "Running container..."
docker-compose up
