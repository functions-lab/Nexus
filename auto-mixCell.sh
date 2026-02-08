#!/bin/bash

LOG_PREFIX="MixCell"
SLEEP_DURATION=40
SLEEP_DURATION_2=40
CONFIG_PATTERN="files/config/ci/config%d-tddconfig-sim-ul-fr2-mu3-100Mhz-cell%d.json"

# Generate unique configurations per cell count
for ((num_cells=2; num_cells<=6; num_cells++)); do
    declare -A seen_combinations
    
    # Generate all valid configurations
    configs=($(seq 1 3))
    
    # Generate all possible valid combinations
    function generate_combinations() {
        local n=$1
        local arr=()
        for ((i=0; i<n; i++)); do
            arr+=(1)
        done
        
        while true; do
            echo "${arr[@]}"
            local idx=$((n-1))
            while ((idx >= 0 && arr[idx] == 3)); do
                ((idx--))
            done
            if ((idx < 0)); then break; fi
            ((arr[idx]++))
            for ((j=idx+1; j<n; j++)); do
                arr[j]=1
            done
        done
    }
    
    while read -r line; do
        config_counts=($line)
        declare -A count_check
        valid=true
        for val in "${config_counts[@]}"; do
            ((count_check[$val]++))
            if [[ ${count_check[$val]} -eq $num_cells ]]; then
                valid=false
                break
            fi
        done
        
        if [[ "$valid" == "false" ]]; then continue; fi
        
        config_files=""
        for ((c=0; c<num_cells; c++)); do
            config_x=${config_counts[$c]}
            config_files+=$(printf "$CONFIG_PATTERN" $config_x $((c+1)))" "
        done
        
        # Ensure cell indices (Y values) increase linearly from 1 to num_cells
        expected_config_files=""
        for ((c=0; c<num_cells; c++)); do
            expected_config_files+=$(printf "$CONFIG_PATTERN" ${config_counts[$c]} $((c+1)))" "
        done
        
        if [[ "$config_files" != "$expected_config_files" ]]; then
            continue
        fi
        
        sorted_combo=$(echo $config_files | tr ' ' '\n' | sort | tr '\n' ' ')
        
        if [[ -z "${seen_combinations[$sorted_combo]}" ]]; then
            seen_combinations[$sorted_combo]=1
            CONFIG_FILES="$config_files"
            LOG_FILE="$LOG_PREFIX-${num_cells}Cell-$(echo $sorted_combo | sed -E 's/files\/config\/ci\/config([0-9]+)-tddconfig-sim-ul-fr2-mu3-100Mhz-cell[0-9]+\.json/config\1.json/g' | tr ' ' '_').log"
            
            echo $LOG_FILE
            echo $CONFIG_FILES
            echo "----------------------------------"
            echo "Running Agora with $num_cells cell(s)..."
            sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/agora $CONFIG_FILES >> $LOG_FILE &
            AGORA_PID=$!
            
            echo "Sleeping for $SLEEP_DURATION seconds..."
            sleep $SLEEP_DURATION
            
            echo "Running sender with $num_cells cell(s)..."
            sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ./build/sender --num_threads=1 --core_offset=10 --enable_slow_start=1 $CONFIG_FILES
            
            echo "Stopping Agora process with Ctrl+C..."
            sudo pkill -SIGINT -P $AGORA_PID  
            sudo kill -INT $AGORA_PID 2>/dev/null
            sleep 5
            
            if ps -p $AGORA_PID > /dev/null; then
                echo "Force killing Agora and its subprocesses..."
                sudo pkill -9 -P $AGORA_PID
                sudo kill -9 $AGORA_PID
            fi
            
            echo "Experiment for $num_cells cell(s) completed."
            echo "----------------------------------"
            sleep $SLEEP_DURATION_2
        fi
    done < <(generate_combinations $num_cells)
    
    unset seen_combinations
done

echo "All experiments completed."
