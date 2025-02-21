from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

# config = config.get_config("pi0_aloha")
config = config.get_config("pi0_act_rebar_infer")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
import time
for i in range(20):
    start_time = time.time()
    example = {
        # "images": {
        #     "gopro_0": np.random.uniform(-1, 1, (3, 640, 480)).astype(np.float32),
        #     "webcam_1": np.random.uniform(-1, 1, (3, 640, 480)).astype(np.float32),
        #     "webcam_2": np.random.uniform(-1, 1, (3, 640, 480)).astype(np.float32)},
        "images": {
            "gopro_0": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "webcam_1": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "webcam_2": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)},
        "state": np.random.rand(7),
        "prompt": "pick up the fork"
    }

    # from openpi.policies import aloha_policy

    # example = aloha_policy.make_aloha_example()

    action_chunk = policy.infer(example)["actions"]
    # print(action_chunk)
    end_time = time.time()
    print(end_time-start_time)