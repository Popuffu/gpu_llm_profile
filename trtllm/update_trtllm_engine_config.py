
import json
import os
import shutil
import numpy as np

from profile_config import BACKEND, PROFILE_CFG, TRT_ENGINE_DIR

assert BACKEND == "trtllm"
trt_engine_config_dir = os.path.join(TRT_ENGINE_DIR, "config.json")
trt_engine_config_sav_dir = os.path.join(TRT_ENGINE_DIR, "config_sav.json")

if not os.path.exists(trt_engine_config_sav_dir):
    shutil.copyfile(trt_engine_config_dir, trt_engine_config_sav_dir) # save the origin config file if not exist
    print("Save the origin trt engine config file: ", trt_engine_config_sav_dir)

with open(trt_engine_config_sav_dir, "r") as f:
    data = json.load(f) # read from the origin config file

np_profile_cfg_list = np.array(PROFILE_CFG)
max_seq_len = int(max(np_profile_cfg_list[:, 1] + np_profile_cfg_list[:, 2]))
data["pretrained_config"]["max_position_embeddings"] = max_seq_len
data["build_config"]["max_seq_len"] = max_seq_len

with open(trt_engine_config_dir, "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated the max_seq_len and max_position_embeddings in {trt_engine_config_dir} to {int(max_seq_len)}")
