"""
Computer Vision-based Motion Tracking for Gesture-Controlled Gaming
Uses MediaPipe for hand tracking and gesture recognition.
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
from typing import List, Tuple, Optional, Dict
from collections import deque


class HandGestureDetector:
    """Detects hand gestures using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=5)
    
    def detect_hands(self, frame: np.ndarray) -> Optional[List]:
        """Detect hands in the frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results.multi_hand_landmarks
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_finger_extended(self, landmarks, finger_tip_id: int, finger_pip_id: int) -> bool:
        """Check if a finger is extended"""
        tip = landmarks[finger_tip_id]
        pip = landmarks[finger_pip_id]
        
        # For thumb, use different logic
        if finger_tip_id == 4:
            return tip.x < landmarks[3].x if landmarks[4].x < landmarks[0].x else tip.x > landmarks[3].x
        
        # For other fingers, check if tip is above PIP joint
        return tip.y < pip.y
    
    def recognize_gesture(self, hand_landmarks) -> str:
        """Recognize hand gesture"""
        if not hand_landmarks:
            return "NONE"
        
        landmarks = hand_landmarks.landmark
        
        # Count extended fingers
        fingers_extended = [
            self.is_finger_extended(landmarks, 4, 3),   # Thumb
            self.is_finger_extended(landmarks, 8, 6),   # Index
            self.is_finger_extended(landmarks, 12, 10), # Middle
            self.is_finger_extended(landmarks, 16, 14), # Ring
            self.is_finger_extended(landmarks, 20, 18)  # Pinky
        ]
        
        extended_count = sum(fingers_extended)
        
        # Recognize specific gestures
        # Fist (all fingers closed)
        if extended_count == 0:
            return "FIST"
        
        # Open palm (all fingers extended)
        elif extended_count == 5:
            return "OPEN_HAND"
        
        # Peace/Victory sign (index and middle fingers)
        elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            return "PEACE"
        
        # Pointing (only index finger)
        elif fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            return "POINT"
        
        # Thumbs up
        elif fingers_extended[0] and not any(fingers_extended[1:]):
            return "THUMBS_UP"
        
        # Three fingers
        elif extended_count == 3 and fingers_extended[1] and fingers_extended[2] and fingers_extended[3]:
            return "THREE"
        
        # Four fingers
        elif extended_count == 4 and not fingers_extended[0]:
            return "FOUR"
        
        return "UNKNOWN"
    
    def get_hand_center(self, hand_landmarks) -> Tuple[float, float]:
        """Get the center point of the hand"""
        if not hand_landmarks:
            return (0, 0)
        
        landmarks = hand_landmarks.landmark
        center_x = sum([lm.x for lm in landmarks]) / len(landmarks)
        center_y = sum([lm.y for lm in landmarks]) / len(landmarks)
        
        return (center_x, center_y)
    
    def get_smoothed_gesture(self, current_gesture: str) -> str:
        """Get smoothed gesture using history"""
        self.gesture_history.append(current_gesture)
        
        # Return most common gesture in history
        if len(self.gesture_history) >= 3:
            gesture_counts = {}
            for gesture in self.gesture_history:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            return max(gesture_counts, key=gesture_counts.get)
        
        return current_gesture
    
    def draw_landmarks(self, frame: np.ndarray, hand_landmarks):
        """Draw hand landmarks on frame"""
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )


class GestureControlledGame:
    """Simple game controlled by hand gestures"""
    
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gesture-Controlled Game")
        self.clock = pygame.time.Clock()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize gesture detector
        self.detector = HandGestureDetector()
        
        # Game state
        self.player_pos = [self.width // 2, self.height - 100]
        self.player_size = 40
        self.player_speed = 8
        
        # Targets
        self.targets = []
        self.target_spawn_timer = 0
        self.score = 0
        
        # Game variables
        self.running = True
        self.current_gesture = "NONE"
        self.hand_position = (0.5, 0.5)  # Normalized position
        
        # Colors
        self.BG_COLOR = (20, 20, 40)
        self.PLAYER_COLOR = (0, 200, 255)
        self.TARGET_COLOR = (255, 200, 0)
        self.TEXT_COLOR = (255, 255, 255)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Gesture action mapping
        self.gesture_actions = {
            "FIST": "SHOOT",
            "OPEN_HAND": "SHIELD",
            "PEACE": "SPECIAL",
            "POINT": "MOVE",
            "THUMBS_UP": "JUMP",
            "THREE": "SPEED_BOOST",
            "FOUR": "SLOW_TIME"
        }
    
    def spawn_target(self):
        """Spawn a new target"""
        x = np.random.randint(30, self.width - 30)
        y = np.random.randint(-100, -30)
        speed = np.random.uniform(2, 5)
        self.targets.append({"pos": [x, y], "speed": speed, "size": 25})
    
    def update_player_from_gesture(self):
        """Update player position based on hand tracking"""
        # Convert normalized hand position to screen coordinates
        target_x = int(self.hand_position[0] * self.width)
        target_y = int(self.hand_position[1] * self.height)
        
        # Smooth movement towards hand position
        dx = target_x - self.player_pos[0]
        dy = target_y - self.player_pos[1]
        
        # Limit movement speed
        distance = math.sqrt(dx**2 + dy**2)
        if distance > 0:
            move_distance = min(self.player_speed, distance)
            self.player_pos[0] += int(dx / distance * move_distance)
            self.player_pos[1] += int(dy / distance * move_distance)
        
        # Keep player in bounds
        self.player_pos[0] = max(self.player_size, min(self.width - self.player_size, self.player_pos[0]))
        self.player_pos[1] = max(self.player_size, min(self.height - self.player_size, self.player_pos[1]))
    
    def check_collisions(self):
        """Check for collisions between player and targets"""
        player_rect = pygame.Rect(
            self.player_pos[0] - self.player_size // 2,
            self.player_pos[1] - self.player_size // 2,
            self.player_size,
            self.player_size
        )
        
        for target in self.targets[:]:
            target_rect = pygame.Rect(
                target["pos"][0] - target["size"] // 2,
                target["pos"][1] - target["size"] // 2,
                target["size"],
                target["size"]
            )
            
            if player_rect.colliderect(target_rect):
                self.targets.remove(target)
                self.score += 10
    
    def update(self):
        """Update game state"""
        # Process camera frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame
            hand_landmarks = self.detector.detect_hands(frame)
            
            if hand_landmarks:
                # Get first hand
                first_hand = hand_landmarks[0]
                
                # Recognize gesture
                gesture = self.detector.recognize_gesture(first_hand)
                self.current_gesture = self.detector.get_smoothed_gesture(gesture)
                
                # Get hand position
                self.hand_position = self.detector.get_hand_center(first_hand)
                
                # Draw landmarks
                self.detector.draw_landmarks(frame, first_hand)
            else:
                self.current_gesture = "NONE"
            
            # Display camera feed
            cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Tracking", frame)
        
        # Update player position based on gesture
        if self.current_gesture != "NONE":
            self.update_player_from_gesture()
        
        # Spawn targets
        self.target_spawn_timer += 1
        if self.target_spawn_timer > 60:  # Spawn every 2 seconds
            self.spawn_target()
            self.target_spawn_timer = 0
        
        # Update targets
        for target in self.targets[:]:
            target["pos"][1] += target["speed"]
            
            # Remove targets that are off screen
            if target["pos"][1] > self.height + 50:
                self.targets.remove(target)
        
        # Check collisions
        self.check_collisions()
    
    def draw(self):
        """Draw game state"""
        self.screen.fill(self.BG_COLOR)
        
        # Draw player
        pygame.draw.circle(self.screen, self.PLAYER_COLOR, self.player_pos, self.player_size // 2)
        
        # Draw player outline based on gesture
        if self.current_gesture == "FIST":
            pygame.draw.circle(self.screen, (255, 0, 0), self.player_pos, self.player_size // 2, 3)
        elif self.current_gesture == "OPEN_HAND":
            pygame.draw.circle(self.screen, (0, 255, 0), self.player_pos, self.player_size // 2 + 5, 3)
        
        # Draw targets
        for target in self.targets:
            pygame.draw.circle(self.screen, self.TARGET_COLOR, 
                             [int(target["pos"][0]), int(target["pos"][1])], 
                             target["size"] // 2)
        
        # Draw UI
        score_text = self.font.render(f"Score: {self.score}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))
        
        gesture_text = self.small_font.render(f"Gesture: {self.current_gesture}", True, self.TEXT_COLOR)
        self.screen.blit(gesture_text, (10, 50))
        
        if self.current_gesture in self.gesture_actions:
            action = self.gesture_actions[self.current_gesture]
            action_text = self.small_font.render(f"Action: {action}", True, (255, 255, 0))
            self.screen.blit(action_text, (10, 75))
        
        # Draw instructions
        instructions = [
            "Gesture Controls:",
            "FIST - Attack mode",
            "OPEN_HAND - Shield mode",
            "Move hand to control player",
            "Collect yellow targets!"
        ]
        
        y_offset = self.height - 150
        for instruction in instructions:
            inst_text = self.small_font.render(instruction, True, (150, 150, 150))
            self.screen.blit(inst_text, (10, y_offset))
            y_offset += 25
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("Gesture-Controlled Game")
        print("Use your hand to control the game!")
        print("Press ESC to quit")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Check if camera window was closed
            if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
            
            self.update()
            self.draw()
            self.clock.tick(30)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        
        print(f"\nFinal Score: {self.score}")


if __name__ == "__main__":
    game = GestureControlledGame()
    game.run()
