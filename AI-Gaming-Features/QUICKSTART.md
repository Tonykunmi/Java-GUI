# Quick Start Guide

Get started with AI Gaming Features in 3 easy steps!

## Step 1: Install Dependencies

```bash
cd AI-Gaming-Features
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet connection.

## Step 2: Run the Launcher

```bash
python launcher.py
```

This will show an interactive menu where you can choose which feature to run.

## Step 3: Try Each Feature!

### 🎯 No Webcam Required:
- **Option 1**: AI Decision Agent - Watch AI make strategic decisions
- **Option 5**: RL NPC Behavior - See NPCs learn through reinforcement learning

### 📷 Webcam Required:
- **Option 2**: Gesture Control - Use hand gestures to control games
- **Option 3**: Emotion Recognition - Game adapts to your emotions
- **Option 4**: AR Object Detection - Augmented reality gaming

## Alternative: Run Features Directly

You can also run any feature directly:

```bash
# AI Decision Agent
python ai_decision_agent.py

# Gesture-Controlled Game
python gesture_motion_tracking.py

# Emotion Recognition Game
python emotion_recognition_game.py

# AR 3D Object Detection
python ar_3d_object_detection.py

# RL NPC Behavior
python rl_npc_behavior.py
```

## System Requirements

### Minimum:
- Python 3.8+
- 4GB RAM
- Webcam (for camera-based features)
- 500MB disk space

### Recommended:
- Python 3.10+
- 8GB RAM
- HD Webcam
- GPU (optional, for better performance)

## Troubleshooting

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "Camera not found"
- Check if webcam is connected
- Try closing other applications using the camera
- Run: `ls /dev/video*` (Linux) to check available cameras

### "ImportError: OpenGL"
```bash
pip install PyOpenGL PyOpenGL-accelerate
```

### Performance Issues
- Close unnecessary applications
- Lower camera resolution in code
- Reduce FPS target

## What to Expect

### 1. AI Decision Agent (30 seconds to start)
- Blue circle = Player (controlled by AI)
- Red circle = Enemy
- Watch the AI make decisions: ATTACK, DEFEND, RETREAT, etc.
- Statistics shown on screen

### 2. Gesture Control (Camera window will open)
- Move your hand in front of camera
- Make different gestures:
  - FIST = Attack mode
  - OPEN_HAND = Shield mode
  - Point finger = Move mode
- Collect yellow targets!

### 3. Emotion Recognition (Camera window will open)
- Show your face to camera
- Express different emotions
- Game difficulty adapts:
  - Happy = Harder, faster
  - Sad = Easier, slower
  - Angry = Very intense
- Use arrow keys to move

### 4. AR Object Detection (Camera window will open)
- Show colored objects (red, green, blue, yellow) to camera
- 3D objects spawn in AR
- Watch them interact in 3D space

### 5. RL NPC Behavior (Instant start)
- Yellow circles = Resources
- Red squares = Enemies
- Colored circles = NPCs
- Watch NPCs learn to collect resources over time
- Press 'T' to toggle training mode

## Tips for Best Experience

1. **Lighting**: Ensure good lighting for camera features
2. **Background**: Plain background helps with gesture/face detection
3. **Camera Position**: Position camera at eye level
4. **Distance**: Sit 1-2 feet from camera
5. **Patience**: RL features take time to show learning

## Quick Demo (No Installation)

Want to see what it looks like? Check out these features first:

1. **Fastest**: RL NPC Behavior - No camera needed, instant start
2. **Most Interactive**: AI Decision Agent - Watch AI think in real-time
3. **Most Fun**: Gesture Control - If you have a webcam

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Customize features by editing the Python files
- Try combining features for your own projects!

## Support

Having issues? Check:
1. Python version: `python --version` (must be 3.8+)
2. Dependencies: All installed from requirements.txt
3. Camera: Working in other applications
4. Error messages: Often point to missing dependencies

## Feature Showcase

```
┌─────────────────────────────────────────────────────────┐
│  Feature              │ Complexity │ Webcam │ Fun Level │
├───────────────────────┼────────────┼────────┼───────────┤
│  AI Decision Agent    │    ⭐⭐     │   No   │   ⭐⭐⭐   │
│  Gesture Control      │   ⭐⭐⭐    │  Yes   │  ⭐⭐⭐⭐  │
│  Emotion Recognition  │   ⭐⭐⭐    │  Yes   │  ⭐⭐⭐⭐  │
│  AR Object Detection  │  ⭐⭐⭐⭐   │  Yes   │ ⭐⭐⭐⭐⭐ │
│  RL NPC Behavior      │  ⭐⭐⭐⭐   │   No   │  ⭐⭐⭐⭐  │
└─────────────────────────────────────────────────────────┘
```

---

**Ready to start? Run `python launcher.py` now! 🚀**
