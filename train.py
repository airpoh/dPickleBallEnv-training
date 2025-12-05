from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import time
import os

from mylib import SharedObsUnityGymWrapper, CustomCNN

# --- CONFIG ---
MODEL_DIR = "./model"  # Directory to save models
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.zip")  # Path for resuming training
TOTAL_TIMESTEPS = 1_000_000  # Total timesteps to train (or continue training)
# Note: If model already trained 1M steps, increase this to continue (e.g., 2_000_000 for 2M total)
CHECKPOINT_FREQ = 10_000  # Save checkpoint every N steps (lower = more frequent saves, safer but uses more disk space)

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Setup side channels
string_channel = StringSideChannel()
channel = CustomDataChannel()

# Game config: 213 = match point 21, random serve
# The last digit is the serve mode:
#     1 = left serve only
#     2 = right serve only
#     3 = random serve
# The remaining digits (before the last one) indicate the match point.
# Example: 213 → match point = 21, serve = 3 (random serve)
reward_cum = [0, 0]
channel.send_data(serve=213, p1=reward_cum[0], p2=reward_cum[1])

print("Creating Unity environment...")
unity_env = UnityEnvironment(
    r"C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles\Training\Windows\dp.exe",
    side_channels=[string_channel, channel]
)

print("Wrapping environment with reward shaping...")
env = SharedObsUnityGymWrapper(unity_env, frame_stack=64, img_size=(168, 84), grayscale=True)

print("Creating A2C model...")
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# --- CHECKPOINT CALLBACK (saves during training) ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=MODEL_DIR,
    name_prefix="checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
    verbose=1,
)

# --- LOAD EXISTING MODEL OR CREATE NEW ONE ---
if os.path.isfile(LAST_MODEL_PATH):
    print(f"Loading existing model from {LAST_MODEL_PATH}")
    print("Resuming training...")
    # Don't specify policy_kwargs when loading - let it use stored kwargs
    model = A2C.load(
        LAST_MODEL_PATH,
        env=env,
        device=device,
        # policy_kwargs=policy_kwargs,  # Removed - causes mismatch error
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    reset_timesteps = False  # Don't reset timestep counter when resuming
else:
    print("Creating new model...")
    model = A2C(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        tensorboard_log="./tensorboard_logs/"
    )
    reset_timesteps = True  # Reset timestep counter for new model

print("Starting training...")
print("Your agent (right side) will play against the pretrained opponent (left side)")
print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
        tb_log_name="dpickleball_run01"
    )

    # Save the final model
    model.save(LAST_MODEL_PATH)
    print(f"\nTraining complete! Final model saved as {LAST_MODEL_PATH}")
    
    # Also save with timestamp
    model_name = f"pickleball_agent_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_name)
    print(f"Also saved as {model_name}.zip")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    # Save current progress
    try:
        model.save(LAST_MODEL_PATH)
        model_name = f"pickleball_agent_interrupted_{time.strftime('%Y%m%d-%H%M%S')}"
        model.save(model_name)
        print(f"Model saved as {LAST_MODEL_PATH} and {model_name}.zip")
        print("You can resume training by running this script again!")
    except Exception as save_error:
        print(f"Warning: Could not save model: {save_error}")

except Exception as e:
    print(f"\nTraining error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Try to save progress if possible
    try:
        print("Attempting to save current progress...")
        model.save(LAST_MODEL_PATH)
        model_name = f"pickleball_agent_crashed_{time.strftime('%Y%m%d-%H%M%S')}"
        model.save(model_name)
        print(f"Progress saved to {LAST_MODEL_PATH} and {model_name}.zip")
        print("You can resume training by running this script again!")
    except Exception as save_error:
        print(f"Could not save progress: {save_error}")
        print("Check checkpoint files in ./model/ directory")
    
    # Re-raise the error so user knows what happened
    raise

finally:
    try:
        env.close()
        print("Environment closed")
    except:
        pass  # Ignore errors when closing
