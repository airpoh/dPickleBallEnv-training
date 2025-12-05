"""
Quick test script to verify training setup works before starting full training.
Tests Unity environment, model creation, and runs just 100 steps.
"""
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from stable_baselines3 import A2C
import torch
import os

from mylib import SharedObsUnityGymWrapper, CustomCNN

print("=" * 60)
print("TRAINING SETUP TEST")
print("=" * 60)
print("This will test Unity environment and run 100 training steps")
print("Press Ctrl+C at any time to stop\n")

# Setup
MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)

string_channel = StringSideChannel()
channel = CustomDataChannel()
reward_cum = [0, 0]
channel.send_data(serve=213, p1=reward_cum[0], p2=reward_cum[1])

# Test 1: Unity Environment
print("[TEST 1/4] Creating Unity environment...")
try:
    unity_env = UnityEnvironment(
        r"C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles\Training\Windows\dp.exe",
        side_channels=[string_channel, channel]
    )
    print("✓ Unity environment created successfully!")
except Exception as e:
    print(f"✗ FAILED: {e}")
    print("\nUnity environment failed. Check Unity build path.")
    exit(1)

# Test 2: Environment Wrapper
print("\n[TEST 2/4] Wrapping environment...")
try:
    env = SharedObsUnityGymWrapper(unity_env, frame_stack=64, img_size=(168, 84), grayscale=True)
    print("✓ Environment wrapped successfully!")
except Exception as e:
    print(f"✗ FAILED: {e}")
    unity_env.close()
    exit(1)

# Test 3: Model Creation/Loading
print("\n[TEST 3/4] Creating/loading model...")
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.zip")
if os.path.isfile(LAST_MODEL_PATH):
    print(f"Found existing model at {LAST_MODEL_PATH}")
    try:
        model = A2C.load(
            LAST_MODEL_PATH,
            env=env,
            device=device,
            verbose=0,
            tensorboard_log="./tensorboard_logs/"
        )
        print(f"✓ Model loaded successfully! (trained {model.num_timesteps:,} steps)")
        reset_timesteps = False
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Creating new model instead...")
        model = A2C(
            policy="CnnPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            tensorboard_log="./tensorboard_logs/"
        )
        reset_timesteps = True
else:
    print("No existing model found, creating new one...")
    model = A2C(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        tensorboard_log="./tensorboard_logs/"
    )
    reset_timesteps = True
    print("✓ New model created successfully!")

# Test 4: Run 100 training steps
print("\n[TEST 4/4] Running 100 training steps (this will take ~10-20 seconds)...")
print("NOTE: Unity window will appear - you can minimize it with Alt+Tab")
print("Press Ctrl+C to stop early if needed\n")

try:
    model.learn(
        total_timesteps=100,
        reset_num_timesteps=reset_timesteps,
        tb_log_name="test_run"
        # progress_bar=True  # Removed - requires extra dependencies
    )
    print("\n✓ Training test completed successfully!")
    print(f"Model now has {model.num_timesteps:,} total timesteps")
    
    # Save test model
    test_model_path = os.path.join(MODEL_DIR, "test_model.zip")
    model.save(test_model_path)
    print(f"✓ Test model saved to {test_model_path}")
    
except KeyboardInterrupt:
    print("\n\n⚠ Test interrupted by user (Ctrl+C)")
    print("This is OK - you can still run full training")
except Exception as e:
    print(f"\n✗ Training test FAILED: {e}")
    print(f"Error type: {type(e).__name__}")
    raise
finally:
    try:
        env.close()
        print("\n✓ Environment closed cleanly")
    except:
        pass

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)
print("\nIf all tests passed (✓), you're ready to run full training:")
print("  python train.py")
print("\nIf any tests failed (✗), fix those issues first.")
print("=" * 60)

