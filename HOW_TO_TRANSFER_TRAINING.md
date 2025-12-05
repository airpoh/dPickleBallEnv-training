# How to Transfer Training Between Computers

## For You (Laptop - Initial Training)

### Step 1: Train until checkpoint
```powershell
conda activate dpickleball
python train.py
```
Wait until you see: `Saving checkpoint to ./model/checkpoint_10000_steps.zip`

### Step 2: Files to Upload to GitHub

**Required Files:**
- ✅ `train.py` - Training script
- ✅ `mylib.py` - Reward shaping wrapper
- ✅ `leftmodel.zip` - Opponent model
- ✅ `model/last_model.zip` - **Your checkpoint** (most important!)
- ✅ `model/checkpoint_10000_steps.zip` - Checkpoint file
- ✅ `CompetitionScripts/teamX.py` - For inference later
- ✅ `README.md` - Documentation

**Optional but helpful:**
- `tensorboard_logs/` - Training logs (can be large)

### Step 3: Upload to GitHub
```powershell
git add train.py mylib.py leftmodel.zip model/ CompetitionScripts/teamX.py
git commit -m "Training checkpoint at 10,000 steps"
git push
```

---

## For Your Friend (Desktop - Resume Training)

### Step 1: Clone/Download Repository
```powershell
git clone <your-repo-url>
cd dPickleBallEnv
```

### Step 2: Set Up Environment
```powershell
conda activate dpickleball
# Make sure all packages are installed (same as your setup)
```

### Step 3: Update Paths
Edit `train.py` line 38 - Update Unity build path:
```python
unity_env = UnityEnvironment(
    r"C:\Path\To\Your\Friend's\Unity\Build\dp.exe",  # Update this!
    side_channels=[string_channel, channel],
    no_graphics=True
)
```

### Step 4: Verify Checkpoint Exists
Check that `model/last_model.zip` exists (this is your 10,000 step checkpoint)

### Step 5: Resume Training
```powershell
python train.py
```

**Expected Output:**
```
Loading existing model from ./model/last_model.zip
Resuming training...
Starting training...
Training for 1,000,000 timesteps...
```

Training will continue from 10,000 steps → 1,000,000 steps!

---

## Important Notes

1. **Checkpoint Location**: The script looks for `./model/last_model.zip`
   - Make sure this file is uploaded to GitHub
   - Your friend needs to download it

2. **Model Directory Structure**:
   ```
   model/
   ├── last_model.zip          ← Main checkpoint (for resuming)
   └── checkpoint_10000_steps.zip  ← Backup checkpoint
   ```

3. **Training Will Resume**: 
   - Timestep counter continues from 10,000
   - All training progress is preserved
   - TensorBoard logs will continue

4. **If Checkpoint Missing**: 
   - Training will start from scratch (0 steps)
   - Make sure `model/last_model.zip` is uploaded!

---

## Quick Checklist

**Before Uploading:**
- [ ] Trained to at least 10,000 steps
- [ ] `model/last_model.zip` exists
- [ ] `model/checkpoint_10000_steps.zip` exists (backup)
- [ ] All code files are up to date

**Friend's Setup:**
- [ ] Cloned repository
- [ ] Environment activated (`dpickleball`)
- [ ] Unity build path updated in `train.py`
- [ ] `model/last_model.zip` exists in repository
- [ ] Ready to resume training!

---

## Troubleshooting

**Problem**: Training starts from 0 instead of resuming
- **Solution**: Check that `model/last_model.zip` exists and is in the correct location

**Problem**: Friend's computer can't find Unity build
- **Solution**: Update the path in `train.py` line 38 to match friend's system

**Problem**: Missing dependencies
- **Solution**: Friend needs to install all packages: `pip install stable-baselines3 torch mlagents-envs shimmy`



