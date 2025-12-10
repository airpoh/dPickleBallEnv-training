# How to Upload Unity Build to GitHub Releases

## ✅ Quick Steps

### Step 1: Create Zip File
```powershell
cd "C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles"
Compress-Archive -Path "Training" -DestinationPath "dPickleball_Unity_Build.zip"
```

This creates `dPickleball_Unity_Build.zip` (about 143MB).

### Step 2: Create GitHub Release

1. **Go to your repository:**
   - https://github.com/alvintehg/dPickleBallEnv-training

2. **Click "Releases"** (on the right sidebar)

3. **Click "Create a new release"** or "Draft a new release"

4. **Fill in the release form:**
   - **Tag version:** `v1.0.0` (or `unity-build-v1`)
   - **Release title:** `Unity Build for Training`
   - **Description:**
     ```
     Unity build required for training the pickleball agent.
     
     ## Installation
     1. Download `dPickleball_Unity_Build.zip`
     2. Extract to a location like `C:\dPickleball\Training\`
     3. Update `UNITY_BUILD_PATH` in `train.py` to point to:
        `C:\dPickleball\Training\Windows\dp.exe`
     ```

5. **Attach the zip file:**
   - Drag and drop `dPickleball_Unity_Build.zip` into the "Attach binaries" area
   - Or click "Choose your files" and select the zip

6. **Click "Publish release"**

### Step 3: Share with Your Friend

Your friend can now:
1. Go to: https://github.com/alvintehg/dPickleBallEnv-training/releases
2. Download `dPickleball_Unity_Build.zip`
3. Extract it
4. Update the path in `train.py`

## 📋 Alternative: Upload to Repository Folder (Not Recommended)

If you really want the build in the repository itself (not recommended):

1. **Update `.gitignore`** to allow Unity build:
   ```gitignore
   # Allow Unity build folder
   !UnityBuild/
   !UnityBuild/**
   ```

2. **Copy the build to repository:**
   ```powershell
   # From your repository directory
   mkdir UnityBuild
   Copy-Item -Path "C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles\Training" -Destination "UnityBuild\Training" -Recurse
   ```

3. **Commit and push:**
   ```powershell
   git add UnityBuild/
   git commit -m "Add Unity build for training"
   git push
   ```

**⚠️ Warning:** This will make your repository 143MB+ larger and slower to clone!

## ✅ Recommendation

**Use GitHub Releases (Step 1-2 above)** - It's cleaner and doesn't bloat your repository!

