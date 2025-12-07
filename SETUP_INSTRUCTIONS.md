# Setup Instructions for Friends

## ⚠️ Common Issues

### Issue 1: "Cannot install ml-agents-envs and ml-agents"

If you're getting errors like:
- `ERROR: File "setup.py" not found`
- `ERROR: Directory ./ml-agents-envs not found`

**This is because these packages are NOT in this repository!** You need to install them from a separate repository.

### Issue 2: "Unity build file not found" or "Couldn't launch the .exe environment"

If you see errors like:
- `❌ ERROR: Unity build file not found!`
- `Couldn't launch the C:\Users\...\dp.exe environment`

**This means you haven't updated the Unity build path!** See Step 5 below.

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

### Step 5: Update Unity Path ⚠️ REQUIRED

**You MUST update the Unity build path before running training!**

**Option 1: Edit the file directly (Recommended)**
Edit `train.py` and `test_training.py` - look for `UNITY_BUILD_PATH` near the top (around line 20):

```python
UNITY_BUILD_PATH = os.getenv(
    "UNITY_BUILD_PATH",
    r"C:\Your\Path\To\Unity\Build\dp.exe"  # UPDATE THIS PATH!
)
```

Replace the default path with your Unity build location.

**Option 2: Use environment variable**
Instead of editing files, you can set an environment variable:

**Windows PowerShell:**
```powershell
$env:UNITY_BUILD_PATH="C:\Your\Path\To\Unity\Build\dp.exe"
python train.py
```

**Windows CMD:**
```cmd
set UNITY_BUILD_PATH=C:\Your\Path\To\Unity\Build\dp.exe
python train.py
```

**How to find your Unity build:**
- Look for a file named `dp.exe` in your Unity build folder
- Usually in a folder like `Training/Windows/` or `Builds/Windows/`
- The path should end with `dp.exe`

### Step 6: Test Setup
```bash
python test_training.py
```

## Why This Happens

The `pip install -e ./ml-agents-envs` command uses the `-e` (editable) flag, which requires the **source code** to be present locally. The ML-Agents source code is in a separate repository (`dpickleball-ml-agents`), not in this training repository.

## Quick Reference

- **This repository** (`dPickleBallEnv`): Contains training scripts, models, and checkpoints
- **ML-Agents repository** (`dpickleball-ml-agents`): Contains the ML-Agents source code that must be installed separately


