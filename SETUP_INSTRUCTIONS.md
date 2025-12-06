# Setup Instructions for Friends

## ⚠️ Common Issue: "Cannot install ml-agents-envs and ml-agents"

If you're getting errors like:
- `ERROR: File "setup.py" not found`
- `ERROR: Directory ./ml-agents-envs not found`

**This is because these packages are NOT in this repository!** You need to install them from a separate repository.

## ✅ Correct Setup Steps

### Step 1: Clone This Training Repository
```bash
git clone <your-friend's-repo-url>
cd dPickleBallEnv
```

### Step 2: Set Up Conda Environment
```bash
conda create -n dpickleball pip python=3.10.12
conda activate dpickleball
```

### Step 3: Install ML-Agents (FROM SEPARATE REPOSITORY)
```bash
# Go to a different directory (NOT inside dPickleBallEnv)
cd ..
# OR go to your home directory, desktop, etc.

# Clone the ML-Agents repository
git clone https://github.com/dPickleball/dpickleball-ml-agents.git
cd dpickleball-ml-agents

# Now install from source
pip install -e ./ml-agents-envs
pip install -e ./ml-agents

# Go back to your training repository
cd ../dPickleBallEnv
```

### Step 4: Install Other Packages
```bash
# Make sure you're in the dPickleBallEnv directory
pip install -r requirements.txt
```

### Step 5: Update Unity Path
Edit `train.py` and update the Unity build path (line 37):
```python
unity_env = UnityEnvironment(
    r"C:\Your\Path\To\Unity\Build\dp.exe",  # Update this!
    side_channels=[string_channel, channel]
)
```

### Step 6: Test Setup
```bash
python test_training.py
```

## Why This Happens

The `pip install -e ./ml-agents-envs` command uses the `-e` (editable) flag, which requires the **source code** to be present locally. The ML-Agents source code is in a separate repository (`dpickleball-ml-agents`), not in this training repository.

## Quick Reference

- **This repository** (`dPickleBallEnv`): Contains training scripts, models, and checkpoints
- **ML-Agents repository** (`dpickleball-ml-agents`): Contains the ML-Agents source code that must be installed separately

