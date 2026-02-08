import json
import os

base_config = {
    "bs_radio_num": 1,
    "ue_radio_num": 1,
    "bs_rru_port": 9000,
    "bs_server_port": 8000,
    "acc100_addr_1": "18:00.0",
    "bbdev_id_1": 0,
    "solution": "s1",
    "cp_size": 64,
    "fft_size": 1024,
    "ofdm_data_num": 400,
    "demul_block_size": 400,
    "frame_schedule": [
        "PUUUUUUUUUUUUUUUUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
    ],
    "ul_mcs": {
        "mcs_index": 17
    },
    "dl_mcs": {
        "mcs_index": 17
    },
    "sample_rate": 122.88e6,
    "max_frame": 23000,
    "small_mimo_acc": True,
    "freq_orthogonal_pilot": False,
    "group_pilot_sc": False,
    "beam_block_size": 400,
    "client_ul_pilot_syms": 2,
    "core_offset": 56,
    "exclude_cores": [0],
    "worker_thread_num": 7,  # will be overwritten
    "socket_thread_num": 1,
    "dpdk_port_offset": 2
}

output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

for x in range(2, 9):
    config = base_config.copy()
    config["worker_thread_num"] = x
    filename = f"config6-TL-100-s1-{x}Cores.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Generated {filepath}")
