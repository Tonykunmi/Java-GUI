"""
AI Gaming Features Launcher
Easy-to-use launcher for all gaming features.
"""

import sys
import os


def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("   🎮 AI GAMING FEATURES LAUNCHER 🎮")
    print("="*60)
    print("\nSelect a feature to run:\n")
    print("1. AI Agent for Real-Time Decision-Making")
    print("   - Watch AI make strategic decisions in real-time")
    print("   - No webcam required\n")
    
    print("2. Gesture-Controlled Gaming (Motion Tracking)")
    print("   - Control games with hand gestures")
    print("   - Requires: Webcam\n")
    
    print("3. Emotion Recognition Adaptive Game")
    print("   - Game adapts to your facial emotions")
    print("   - Requires: Webcam\n")
    
    print("4. AR 3D Object Detection")
    print("   - Augmented reality with 3D objects")
    print("   - Requires: Webcam, OpenGL\n")
    
    print("5. Reinforcement Learning NPC Behavior")
    print("   - Watch NPCs learn intelligent behavior")
    print("   - No webcam required\n")
    
    print("0. Exit")
    print("\n" + "="*60)


def run_feature(choice: str):
    """Run the selected feature"""
    features = {
        "1": ("ai_decision_agent.py", "AI Decision Agent"),
        "2": ("gesture_motion_tracking.py", "Gesture-Controlled Game"),
        "3": ("emotion_recognition_game.py", "Emotion Recognition Game"),
        "4": ("ar_3d_object_detection.py", "AR 3D Object Detection"),
        "5": ("rl_npc_behavior.py", "RL NPC Behavior Simulation")
    }
    
    if choice in features:
        filename, name = features[choice]
        print(f"\n🚀 Launching {name}...")
        print(f"{'='*60}\n")
        
        # Import and run the module
        try:
            if choice == "1":
                import ai_decision_agent
                game = ai_decision_agent.GameSimulation()
                game.run()
            elif choice == "2":
                import gesture_motion_tracking
                game = gesture_motion_tracking.GestureControlledGame()
                game.run()
            elif choice == "3":
                import emotion_recognition_game
                game = emotion_recognition_game.AdaptiveEmotionGame()
                game.run()
            elif choice == "4":
                import ar_3d_object_detection
                app = ar_3d_object_detection.ARGamingApp()
                app.run()
            elif choice == "5":
                import rl_npc_behavior
                game = rl_npc_behavior.RLSimulationGame(num_npcs=5)
                game.run()
        except Exception as e:
            print(f"\n❌ Error running {name}:")
            print(f"   {str(e)}")
            print(f"\nPlease ensure all dependencies are installed:")
            print(f"   pip install -r requirements.txt")
            print(f"\nFor camera features, ensure your webcam is connected.")
        
        print(f"\n{'='*60}")
        print(f"✅ {name} completed")
        input("\nPress Enter to return to menu...")
    elif choice == "0":
        print("\n👋 Thanks for using AI Gaming Features!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please try again.")
        input("Press Enter to continue...")


def check_dependencies():
    """Check if basic dependencies are installed"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pygame
    except ImportError:
        missing.append("pygame")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    if missing:
        print("\n⚠️  WARNING: Missing dependencies detected!")
        print("Missing packages:", ", ".join(missing))
        print("\nPlease install dependencies:")
        print("   pip install -r requirements.txt")
        input("\nPress Enter to continue anyway...")
        print()


def main():
    """Main launcher function"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("\n🎮 Welcome to AI Gaming Features! 🎮\n")
    print("Checking dependencies...")
    check_dependencies()
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_menu()
        choice = input("\nEnter your choice (0-5): ").strip()
        run_feature(choice)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
