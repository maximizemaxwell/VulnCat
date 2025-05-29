#!/bin/bash

echo "Testing VulnCat CLI..."

# Test evaluation command with the real dataset
echo "1. Testing evaluation on dataset..."
cargo run --release -- evaluate -d dataset/diversevul_20230702.json -o evaluation_report.json

# Show help
echo -e "\n2. Showing help..."
cargo run --release -- --help

echo -e "\nDone!"