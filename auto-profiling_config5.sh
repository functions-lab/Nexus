#!/bin/bash

CONFIG_DIR="files/config/ci/Config5-VeryCells"
LOG_PREFIX="MCS17-Config5"
NUM_CORES=(2 3 4 5 6 7 8 10 12 14 16)  # Exact 11 configs
SLEEP_DURATION=40
SLEEP_DURATION_2=40

CONFIG_FILES=($(ls $CONFIG_DIR/*.json | sort -V))
NUM_CONFIGS=${#CONFIG_FILES[@]}

if [ $NUM_CONFIGS -ne 11 ]; then
    echo "Error: Expected 11 config files, but found $NUM_CONFIGS."
    exit 1
fi

for ((i=0; i<NUM_CONFIGS; i++)); do
    CONFIG_FILE="$(ls $CONFIG_DIR/*-${NUM_CORES[i]}Core.json 2>/dev/null)"
    if [ -z "$CONFIG_FILE" ]; then
        echo "Error: No matching config file found for ${NUM_CORES[i]} cores."
        exit 1
    fi

    X=${NUM_CORES[i]}
    LOG_FILE="$LOG_PREFIX-${X}Core-S4-2PF.log"

    echo "Running Agora with config: $CONFIG_FILE and $X core..."
    echo "Log file: $LOG_FILE"
    sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/agora $CONFIG_FILE >> $LOG_FILE &
    AGORA_PID=$!

    echo "Sleeping for $SLEEP_DURATION seconds..."
    sleep $SLEEP_DURATION

    echo "Running sender with $X cores..."
    sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/sender --num_threads=1 --core_offset=0 --enable_slow_start=1 $CONFIG_FILE

    echo "Stopping Agora process with Ctrl+C..."
    sudo pkill -SIGINT -P $AGORA_PID  # Kill all child processes
    sudo kill -INT $AGORA_PID 2>/dev/null
    sleep 5  # Allow graceful shutdown

    if ps -p $AGORA_PID > /dev/null; then
        echo "Force killing Agora and its subprocesses..."
        sudo pkill -9 -P $AGORA_PID
        sudo kill -9 $AGORA_PID
    fi

    echo "Experiment for $X cores completed."
    echo "----------------------------------"

    sleep $SLEEP_DURATION_2

done

echo "All experiments completed."
