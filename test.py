from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.datasets.utils import write_json, serialize_dict
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.utils import dataset_to_policy_features
import torch
import torch_npu
from PIL import Image
import torchvision

device = "npu:0"
torch.npu.set_device(device)

# try:
#     dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./demo_data_language')
# except:
#     dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')
dataset_metadata = LeRobotDatasetMetadata("lerobot/aloha_sim_transfer_cube_human")

features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Temporal ensemble to make smoother trajectory predictions
cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size=10, n_action_steps=10)
# cfg = PI0FASTConfig(input_features=input_features, output_features=output_features, chunk_size=10, n_action_steps=10)
# cfg = PI0Config(input_features=input_features, output_features=output_features, chunk_size=10, n_action_steps=10)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

policy = SmolVLAPolicy.from_pretrained('./ckpt/smolvla_aloha/checkpoints/last/pretrained_model', config=cfg)
# policy = PI0FASTPolicy.from_pretrained('./ckpt/pi0_train/checkpoints/last/pretrained_model', config=cfg, dataset_stats=dataset_metadata.stats)
# policy = PI0Policy.from_pretrained('./ckpt/train_pi0/checkpoints/last/pretrained_model', config=cfg, dataset_stats=dataset_metadata.stats)

from mujoco_env.y_env2 import SimpleEnv2
xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

from torchvision import transforms
# Approach 1: Using torchvision.transforms
def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that:
     Converts to a FloatTensor and scales pixel values [0,255] -> [0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0–255] -> FloatTensor [0.0–1.0], shape C×H×W
    ])

step = 0
PnPEnv.reset(seed=0)
policy.reset()
policy.eval()
save_image = True
IMG_TRANSFORM = get_default_transform()
while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Check if the task is completed
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # Reset the environment and action queue
            policy.reset()
            PnPEnv.reset()
            step = 0
            save_image = False
        # Get the current state of the environment
        state = PnPEnv.get_joint_state()[:6]
        # Get the current image from the environment
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = IMG_TRANSFORM(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = IMG_TRANSFORM(wrist_image)
        data = {
            'observation.state': torch.tensor(np.array([state])).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [PnPEnv.instruction],
        }
        # Select an action
        action = policy.select_action(data)
        action = action[0,:7].cpu().detach().numpy()
        # Take a step in the environment
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1
        if step == 101:
            break
        success = PnPEnv.check_success()
        if success:
            print('Success')
            break



import torch
import torch_npu
from safetensors.torch import load_file

device = "npu:0"
torch.npu.set_device(device)

weights = load_file("./ckpt/smolvla_pusht/checkpoints/last/pretrained_model/model.safetensors")
print("Loaded successfully!")