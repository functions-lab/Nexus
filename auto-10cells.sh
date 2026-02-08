#!/bin/bash

CONFIG_PREFIX="files/config/ci/tddconfig-sim-ul-fr2-mu3-100Mhz-cell"
LOG_PREFIX="2-2-100MHz"
NUM_CELLS=10
START_CELL=10
SLEEP_DURATION=40
SLEEP_DURATION_2=40

for ((i=START_CELL; i<=NUM_CELLS; i++)); do
    CONFIG_FILES=""
    for ((j=1; j<=i; j++)); do
        CONFIG_FILES+="$CONFIG_PREFIX$j.json "
    done
    LOG_FILE="$LOG_PREFIX-${i}Cell.log"

    echo "Running Agora with $i cell(s)..."
    sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/agora $CONFIG_FILES >> $LOG_FILE &
    AGORA_PID=$!

    echo "Sleeping for $SLEEP_DURATION seconds..."
    sleep $SLEEP_DURATION

    echo "Running sender with $i cell(s)..."
    sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/sender --num_threads=1 --core_offset=10 --enable_slow_start=1 $CONFIG_FILES

    echo "Stopping Agora process with Ctrl+C..."
    sudo pkill -SIGINT -P $AGORA_PID  # Kill all child processes
    sudo kill -INT $AGORA_PID 2>/dev/null
    sleep 5  # Allow graceful shutdown

    if ps -p $AGORA_PID > /dev/null; then
        echo "Force killing Agora and its subprocesses..."
        sudo pkill -9 -P $AGORA_PID
        sudo kill -9 $AGORA_PID
    fi

    echo "Experiment for $i cell(s) completed."
    echo "----------------------------------"

    sleep $SLEEP_DURATION_2

done

echo "All experiments completed."
