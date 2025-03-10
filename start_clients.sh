#!/bin/bash

# List of client YAML files
clients=("client_1.yaml" "client_2.yaml" "client_3.yaml" "client_4.yaml" "client_5.yaml" \
         "client_6.yaml" "client_7.yaml" "client_8.yaml" "client_9.yaml" "client_10.yaml")

# Loop over each client and start in background
for client in "${clients[@]}"
do
    echo "Starting client with config: $client"
    fedn client start -in "$client" &
    sleep 2  # Short delay to avoid overloading the system
done

echo "✅ All clients started in background!"

