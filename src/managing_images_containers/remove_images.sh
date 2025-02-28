#!/bin/bash

echo "Search and remove Docker images named as 'modules-module_number_service' and 'base_image'"

# Find matching images
images=$(docker images --filter "reference=modules-module_*_service" --format "{{.ID}} {{.Repository}}:{{.Tag}}")
base_image=$(docker images --filter "reference=demo_base_image" --format "{{.ID}} {{.Repository}}:{{.Tag}}")

# Combine the results of both searches
all_images=$(echo -e "$images\n$base_image" | sed '/^$/d') # Remove empty lines

if [[ -z "$all_images" ]]; then
    echo "No corresponding images found"
else
    echo "Images found:"
    echo "$all_images"
    
    echo "Removing these images..."
    while IFS= read -r line; do
        image_id=$(echo "$line" | awk '{print $1}')
        if docker rmi -f "$image_id"; then
            echo "Successfully removed image: $image_id"
        else
            echo "Failed to remove image: $image_id"
        fi
    done <<< "$all_images"

    echo "Docker images removed: Ready for a new deployment"
fi
