#!/bin/bash

echo "---------- DOCKER AI Pipeline ----------"

if [[ "$1" == "--build" ]]; then
    echo "Deployment with Building Image Steps"
    docker build -f Dockerfile.base -t demo_base_image .
    cd modules
    docker compose up --build                

else
    echo "Deploying without rebuilding images"
    cd modules
    docker compose up    

fi