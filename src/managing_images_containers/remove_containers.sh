#!/bin/bash

echo "Search and remove docker container named as 'module_number_container'"

# Find associate container
containers=$(docker ps -a --filter "name=^module_[0-9]+_container$" --format "{{.ID}} {{.Names}}")

if [[ -z "$containers" ]]; then
    echo "No corresponding container found"
else
    echo "Containers found:"
    echo "$containers"
    
    echo "Remove these containers"
    while IFS= read -r line; do
        container_id=$(echo "$line" | awk '{print $1}')
        docker rm -f "$container_id"
    done <<< "$containers"

    echo "Docker containers removed : Ready for a new deployment"
fi
