#!/bin/bash

CONFIG_DIR="files/config/ci/NEXUS-MCS23/PRB-16-SW"
LOG_DIR="NEXUS-DATA-MCS23/PRB-TL25-SW"
SLEEP_DURATION=10
SLEEP_DURATION_2=10

mkdir -p "$LOG_DIR"

CONFIG_FILES=($(ls "$CONFIG_DIR"/config*-TL-*-*-*Cores.json 2>/dev/null | sort -V))
NUM_CONFIGS=${#CONFIG_FILES[@]}

if [ $NUM_CONFIGS -eq 0 ]; then
    echo "Error: No config files found in $CONFIG_DIR."
    exit 1
fi

echo "Found $NUM_CONFIGS configuration files."

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    BASE_NAME=$(basename "$CONFIG_FILE" .json)
    LOG_FILE="${LOG_DIR}/${BASE_NAME}.log"

    echo "Running Agora with config: $CONFIG_FILE"
    echo "Logging to: $LOG_FILE"
    sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/agora "$CONFIG_FILE" >> "$LOG_FILE" &
    AGORA_PID=$!

    echo "Sleeping for $SLEEP_DURATION seconds..."
    sleep $SLEEP_DURATION

    echo "Running sender..."
    sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/sender --num_threads=1 --core_offset=0 --enable_slow_start=1 "$CONFIG_FILE"

    echo "Stopping Agora process..."
    sudo pkill -SIGINT -P $AGORA_PID
    sudo kill -INT $AGORA_PID 2>/dev/null
    sleep 5

    if ps -p $AGORA_PID > /dev/null; then
        echo "Force killing remaining Agora processes..."
        sudo pkill -9 -P $AGORA_PID
        sudo kill -9 $AGORA_PID
    fi

    echo "Completed: $BASE_NAME"
    echo "------------------------------------"

    sleep $SLEEP_DURATION_2
done

echo "All experiments completed."
