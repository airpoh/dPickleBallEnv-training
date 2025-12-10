# My Pickleball RL Training Project

This repository contains my reinforcement learning training setup for the dPickleBall environment.

## 🎯 Project Overview

This project trains an AI agent to play pickleball using Stable-Baselines3 (A2C algorithm) with custom reward shaping.

## 📁 Repository Contents

### Training Files
- **`train.py`** - Main training script with checkpoint support
- **`mylib.py`** - Custom environment wrapper with reward shaping
- **`test_training.py`** - Quick test script to verify setup works

### Competition/Inference Files
- **`CompetitionScripts/teamX.py`** - Agent class for competition/inference
- **`CompetitionScripts/Competition.py`** - Competition script

### Models
- **`leftmodel.zip`** - Pretrained opponent model
- **`model/last_model.zip`** - Latest training checkpoint (for resuming)

### Documentation
- **`HOW_TO_TRANSFER_TRAINING.md`** - Instructions for transferring training between computers
- **`COLLABORATIVE_TRAINING.md`** - Guide for multiple friends taking turns training
- **`SETUP_INSTRUCTIONS.md`** - Detailed setup guide for friends

### ⚠️ NOT Included
- **Unity Build (`dp.exe`)** - Must be obtained separately from the repository owner

## 🚀 Quick Start

### Prerequisites
1. Install conda environment:
   ```bash
   conda create -n dpickleball pip python=3.10.12
   conda activate dpickleball
   ```

2. **Install ML-Agents (REQUIRED - from separate repository):**
   
   ⚠️ **IMPORTANT:** You must clone the ML-Agents repository separately. These packages are NOT in this repository.
   
   ```bash
   # Clone the ML-Agents repository (do this OUTSIDE of this repository)
   git clone https://github.com/dPickleball/dpickleball-ml-agents.git
   cd dpickleball-ml-agents
   
   # Install from source (the -e flag requires the source code to be present)
   pip install -e ./ml-agents-envs
   pip install -e ./ml-agents
   
   # Go back to your training repository
   cd ..
   ```

3. Install other required packages:
   ```bash
   # Option 1: Install from requirements.txt (recommended)
   pip install -r requirements.txt
   
   # Option 2: Install individually
   pip install stable-baselines3
   pip install torch
   pip install gym
   pip install shimmy
   pip install opencv-python==4.7.0.72
   pip install numpy==1.23.5
   ```

### Training

1. **⚠️ IMPORTANT: Get Unity Build First!**
   
   **The Unity build (`dp.exe`) is NOT in this repository!** You need to get it separately.
   
   - Ask the repository owner to share the Unity build folder
   - The Unity build contains `dp.exe` and supporting files
   - Extract it to a location on your computer (e.g., `C:\dPickleball\Training\Windows\`)

2. **⚠️ IMPORTANT: Update Unity Build Path!**
   
   After getting the Unity build, you MUST update the Unity build path in both `train.py` and `test_training.py`.
   
   Look for `UNITY_BUILD_PATH` near the top of each file (around line 20) and update it:
   ```python
   UNITY_BUILD_PATH = os.getenv(
       "UNITY_BUILD_PATH",
       r"C:\Your\Path\To\Unity\Build\dp.exe"  # UPDATE THIS!
   )
   ```
   
   Replace the path with the actual location of your `dp.exe` file.
   
   Or set the `UNITY_BUILD_PATH` environment variable instead.
   
   See `SETUP_INSTRUCTIONS.md` for detailed instructions.

3. **Test your setup first:**
   ```bash
   python test_training.py
   ```

4. **Start training:**
   ```bash
   python train.py
   ```

5. **Training will:**
   - Save checkpoints every 10,000 steps in `./model/` directory
   - Save final model as `./model/last_model.zip`
   - Can be resumed by running `train.py` again

### Using Trained Model

Update `CompetitionScripts/teamX.py` line 25 with your model path:
```python
model_path = "./model/last_model.zip"
```

Then run:
```bash
python CompetitionScripts/Competition.py
```

## 📊 Training Configuration

- **Algorithm:** A2C (Advantage Actor-Critic)
- **Frame Stack:** 64 frames
- **Image Size:** 168x84 (grayscale)
- **Checkpoint Frequency:** Every 10,000 steps
- **Total Training Steps:** 1,000,000 (configurable in `train.py`)

## 🔄 Transferring Training

### Single Transfer
See `HOW_TO_TRANSFER_TRAINING.md` for instructions on:
- Uploading checkpoints to GitHub
- Resuming training on another computer
- Continuing training after reaching target steps

### Collaborative Training (Multiple Friends)
See `COLLABORATIVE_TRAINING.md` for instructions on:
- Taking turns training the model
- Sharing progress via GitHub
- Resuming from another friend's checkpoint
- Best practices for team training

## 📝 Notes

- Training uses reward shaping for better learning
- Opponent model (`leftmodel.zip`) is used during training
- Unity window will appear during training (can be minimized)
- Checkpoints allow resuming training if interrupted

## 🛠️ Troubleshooting

- **Unity crashes:** Make sure Unity build path is correct in `train.py`
- **Model loading errors:** Check that model files exist in `./model/` directory
- **Memory errors:** Opponent model may fail to load - training continues without it
