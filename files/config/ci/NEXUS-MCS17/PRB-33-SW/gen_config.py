import json
import os

# Mapping from TL to number of U symbols
TL_to_U = {
    25: 4,
    50: 8,
    75: 12,
    100: 16
}

base_config = {
    "bs_radio_num": 1,
    "ue_radio_num": 1,
    "bs_rru_port": 9000,
    "bs_server_port": 8000,
    "acc100_addr_1": "18:00.0",
    "bbdev_id_1": 0,
    "solution": "s1",
    "cp_size": 128,
    "fft_size": 2048,
    "ofdm_data_num": 800,
    "demul_block_size": 800,
    "frame_schedule": [""],  # will be overwritten
    "ul_mcs": {
        "mcs_index": 17
    },
    "dl_mcs": {
        "mcs_index": 17
    },
    "sample_rate": 245.76e6,
    "max_frame": 23000,
    "small_mimo_acc": True,
    "freq_orthogonal_pilot": False,
    "group_pilot_sc": False,
    "beam_block_size": 800,
    "client_ul_pilot_syms": 2,
    "core_offset": 56,
    "exclude_cores": [0],
    "worker_thread_num": 1,  # will be overwritten
    "socket_thread_num": 1,
    "dpdk_port_offset": 2
}

output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

solutions = {
    "scsw": [1],
    "sw": list(range(2, 9))
}

for tl, num_u in TL_to_U.items():
    for sol, worker_range in solutions.items():
        for workers in worker_range:
            config = base_config.copy()
            config["solution"] = sol
            config["worker_thread_num"] = workers
            num_g = 70 - 1 - num_u  # subtract 1 for P symbol
            frame_schedule = "P" + "U" * num_u + "G" * num_g
            config["frame_schedule"] = [frame_schedule]

            filename = f"config2-TL-{tl}-{sol}-{workers}Cores.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Generated {filepath}")
