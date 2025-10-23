# game_ai_suite

A suite of AI modules for gaming scenarios:
- Real-time decision-making agent
- CV-based motion tracking for gesture control
- Facial emotion recognition for adaptive gameplay
- 3D object detection for AR via ArUco
- Reinforcement learning-based NPC behavior

## Quickstart

Create a virtualenv and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run demos:

```bash
python -m game_ai_suite.demos.run_agent
python -m game_ai_suite.demos.run_gestures
python -m game_ai_suite.demos.run_emotion
python -m game_ai_suite.demos.run_ar
python -m game_ai_suite.demos.run_rl
```