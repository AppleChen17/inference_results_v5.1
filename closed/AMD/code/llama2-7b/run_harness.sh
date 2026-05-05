#!/bin/bash

input_string=$@

config_path=$(echo "$input_string" | grep -oP '(?<=--config-path\s)[^ ]+')  
model_name=$(basename "$config_path")

config_name=$(echo "$input_string" | grep -oP '(?<=--config-name\s)[^\s]+')  
IFS='_' read -r scenario gpu <<< "$config_name"

output_log_dir=$(echo "$input_string" | grep -oP '(?<=harness_config\.output_log_dir=)[^\s]+')
output_log_dir=${output_log_dir:-"logs"}
mkdir -p $output_log_dir

source scripts/power_settings.sh $model_name $gpu $scenario
python main.py $input_string 2>&1 | tee $output_log_dir/output.log

