# How to Share Unity Build with Friends

## 📍 Get the Unity Build

Download from: [Google Drive](https://drive.google.com/drive/folders/1lFqj6lopoIO96C_IO8yGXGzRMY1S4fzi) (zip name: `dPickleball BuildFiless.zip`)

After extraction, the build typically lives at:
```
C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles\Training\Windows\dp.exe
```

## 📦 What to Share

You need to share the **entire Unity build folder**, not just `dp.exe`. The build typically includes:
- `dp.exe` - Main executable
- `dp_Data/` - Game data folder (textures, models, etc.)
- `MonoBleedingEdge/` - Unity runtime files
- `UnityPlayer.dll` - Unity player library
- Other supporting DLLs and files

**Share the entire `Training/Windows/` folder** (or the parent folder containing it).

## 🚀 Methods to Share

### Method 1: Cloud Storage (Recommended)

**Google Drive / OneDrive / Dropbox:**

1. **Zip the Unity build folder:**
   ```powershell
   # Navigate to the parent folder
   cd "C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles"
   
   # Create a zip file (Windows 10+)
   Compress-Archive -Path "Training" -DestinationPath "dPickleball_Unity_Build.zip"
   ```

2. **Upload to cloud storage:**
   - Upload `dPickleball_Unity_Build.zip` to Google Drive/OneDrive/Dropbox
   - Share the link with your friend
   - Make sure to set permissions to "Anyone with the link can view" or share directly with your friend

3. **Your friend downloads and extracts:**
   - Download the zip file
   - Extract to a location like `C:\dPickleball\Training\`
   - Update the path in `train.py` to point to `C:\dPickleball\Training\Windows\dp.exe`

### Method 2: File Transfer Service

**WeTransfer / SendAnywhere / Firefox Send:**

1. Zip the Unity build folder (same as Method 1)
2. Go to wetransfer.com or similar service
3. Upload the zip file
4. Send the download link to your friend
5. Link expires after 7 days (WeTransfer) or as specified

### Method 3: GitHub Releases (Recommended for GitHub)

**Best option if you want to use GitHub!**

1. **Zip the Unity build folder:**
   ```powershell
   cd "C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles"
   Compress-Archive -Path "Training" -DestinationPath "dPickleball_Unity_Build.zip"
   ```

2. **Create a GitHub Release:**
   - Go to your GitHub repository: https://github.com/alvintehg/dPickleBallEnv-training
   - Click "Releases" → "Create a new release"
   - Tag: `v1.0.0` (or any version number)
   - Title: "Unity Build for Training"
   - Description: "Unity build required for training. Extract the zip and update UNITY_BUILD_PATH in train.py"
   - Drag and drop `dPickleball_Unity_Build.zip` as a release asset
   - Click "Publish release"

3. **Your friend downloads:**
   - Go to: https://github.com/alvintehg/dPickleBallEnv-training/releases
   - Download `dPickleball_Unity_Build.zip`
   - Extract and update path in `train.py`

**Note:** Your build is 143MB, which is fine for GitHub Releases (up to 2GB per file).

### Method 4: Upload to Repository (Not Recommended)

⚠️ **Not recommended** - Makes repository large and slow to clone.

If you still want to do this:
1. The build is 143MB total (under GitHub's limits)
2. No individual files exceed 100MB
3. You'll need to update `.gitignore` to allow it
4. Repository will be slower to clone

**Better to use GitHub Releases (Method 3) instead!**

### Method 4: Direct File Transfer (Same Network)

If you and your friend are on the same network:

1. **Enable file sharing on your computer**
2. **Share the folder** containing the Unity build
3. **Your friend copies it over the network**

## 📋 Quick Steps for You

1. **Navigate to the build folder:**
   ```powershell
   cd "C:\Users\User\Downloads\dPickleball BuildFiless\dPickleball BuildFiles"
   ```

2. **Create zip file:**
   ```powershell
   Compress-Archive -Path "Training" -DestinationPath "dPickleball_Unity_Build.zip"
   ```

3. **Upload to cloud storage** (Google Drive recommended)

4. **Share the link** with your friend

5. **Tell your friend:**
   - Download and extract the zip
   - Extract to a location like `C:\dPickleball\Training\`
   - Update `UNITY_BUILD_PATH` in `train.py` to: `r"C:\dPickleball\Training\Windows\dp.exe"`

## ✅ Verification

After your friend extracts the build, they should:
1. Navigate to the `Windows` folder
2. Double-click `dp.exe` to verify it runs
3. If the Unity game window opens, the build is correct!

## 📝 Notes

- **File Size:** Unity builds can be 100MB - 2GB+ depending on the game
- **Extraction Time:** Large builds may take several minutes to extract
- **Antivirus:** Some antivirus software may flag `.exe` files - your friend may need to allow it
- **Path Length:** Windows has a 260 character path limit - keep extraction paths short

## 🔗 Alternative: If You Have the Original Source

If you have access to the original Unity project or build source:
- You can rebuild it for your friend
- Or share the original download link if it's still available

