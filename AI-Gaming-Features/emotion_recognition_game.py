"""
Facial Emotion Recognition System for Adaptive Gameplay
Uses deep learning to detect player emotions and adapt game difficulty.
"""

import cv2
import numpy as np
import pygame
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import json


class EmotionDetector:
    """Detects facial emotions using Haar Cascades and emotion classification"""
    
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self):
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion history for smoothing
        self.emotion_history = deque(maxlen=10)
        self.emotion_confidence_history = deque(maxlen=10)
        
        # Simple emotion classifier (rule-based for demo)
        # In production, use a trained model like FER or DeepFace
        self.current_emotion = "Neutral"
        self.emotion_confidence = 0.0
        
        # Face feature analyzer
        self.previous_face_size = 0
        self.face_size_history = deque(maxlen=5)
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def analyze_facial_features(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Dict:
        """Analyze facial features for emotion estimation"""
        x, y, w, h = face_rect
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return {"emotion": "Neutral", "confidence": 0.0}
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness and contrast (indicators of emotion)
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)
        
        # Track face size changes (can indicate surprise/fear)
        face_size = w * h
        self.face_size_history.append(face_size)
        
        # Simple rule-based emotion estimation
        # This is a simplified version - use a trained CNN for production
        emotion, confidence = self._estimate_emotion_simple(brightness, contrast, face_size)
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "brightness": brightness,
            "contrast": contrast,
            "face_size": face_size
        }
    
    def _estimate_emotion_simple(self, brightness: float, contrast: float, face_size: int) -> Tuple[str, float]:
        """Simple rule-based emotion estimation"""
        
        # Calculate face size change
        face_size_change = 0
        if len(self.face_size_history) >= 2:
            face_size_change = (self.face_size_history[-1] - self.face_size_history[0]) / max(self.face_size_history[0], 1)
        
        # Emotion rules (simplified)
        if contrast > 50 and brightness < 100:
            return "Angry", 0.6
        elif face_size_change > 0.2:
            return "Surprise", 0.7
        elif brightness > 130 and contrast > 40:
            return "Happy", 0.65
        elif brightness < 90:
            return "Sad", 0.5
        elif contrast < 30:
            return "Neutral", 0.8
        else:
            # Random variation for demo purposes
            emotions_demo = ["Happy", "Neutral", "Surprise", "Happy"]
            return np.random.choice(emotions_demo), np.random.uniform(0.4, 0.8)
    
    def get_emotion(self, frame: np.ndarray) -> Dict:
        """Get current emotion from frame"""
        faces = self.detect_faces(frame)
        
        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            result = self.analyze_facial_features(frame, largest_face)
            
            # Update history
            self.emotion_history.append(result["emotion"])
            self.emotion_confidence_history.append(result["confidence"])
            
            # Get smoothed emotion
            if len(self.emotion_history) >= 3:
                emotion_counts = {}
                for emotion in self.emotion_history:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
                smoothed_confidence = np.mean(list(self.emotion_confidence_history))
            else:
                smoothed_emotion = result["emotion"]
                smoothed_confidence = result["confidence"]
            
            self.current_emotion = smoothed_emotion
            self.emotion_confidence = smoothed_confidence
            
            result["smoothed_emotion"] = smoothed_emotion
            result["smoothed_confidence"] = smoothed_confidence
            result["face"] = largest_face
            
            return result
        
        return {"emotion": "No Face", "confidence": 0.0}
    
    def draw_face_rectangle(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int], 
                           emotion: str, confidence: float):
        """Draw rectangle around face with emotion label"""
        x, y, w, h = face_rect
        
        # Color based on emotion
        color_map = {
            "Happy": (0, 255, 0),
            "Sad": (255, 0, 0),
            "Angry": (0, 0, 255),
            "Surprise": (255, 255, 0),
            "Fear": (128, 0, 128),
            "Neutral": (200, 200, 200)
        }
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw emotion label
        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)


class AdaptiveEmotionGame:
    """Game that adapts difficulty based on player emotions"""
    
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Emotion-Adaptive Game")
        self.clock = pygame.time.Clock()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize emotion detector
        self.detector = EmotionDetector()
        
        # Game state
        self.player_pos = [self.width // 2, self.height - 80]
        self.player_size = 40
        self.player_speed = 7
        
        # Obstacles
        self.obstacles = []
        self.obstacle_spawn_rate = 80  # Higher = slower
        self.obstacle_speed = 4
        self.spawn_timer = 0
        
        # Power-ups
        self.powerups = []
        self.powerup_timer = 0
        
        # Game variables
        self.score = 0
        self.lives = 3
        self.current_emotion = "Neutral"
        self.difficulty_level = "Medium"
        self.running = True
        
        # Emotion-based difficulty adaptation
        self.emotion_difficulty_map = {
            "Happy": {"speed_mult": 1.2, "spawn_rate_mult": 0.9, "color": (255, 200, 0)},
            "Sad": {"speed_mult": 0.7, "spawn_rate_mult": 1.3, "color": (100, 100, 255)},
            "Angry": {"speed_mult": 1.5, "spawn_rate_mult": 0.7, "color": (255, 50, 50)},
            "Surprise": {"speed_mult": 1.0, "spawn_rate_mult": 1.0, "color": (255, 255, 100)},
            "Fear": {"speed_mult": 0.8, "spawn_rate_mult": 1.2, "color": (150, 150, 255)},
            "Neutral": {"speed_mult": 1.0, "spawn_rate_mult": 1.0, "color": (200, 200, 200)}
        }
        
        # Statistics
        self.emotion_log = []
        self.game_start_time = time.time()
        
        # Colors
        self.BG_COLOR = (20, 20, 40)
        self.PLAYER_COLOR = (0, 255, 150)
        self.OBSTACLE_COLOR = (255, 50, 50)
        self.POWERUP_COLOR = (255, 200, 0)
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
    
    def get_difficulty_multipliers(self) -> Dict:
        """Get difficulty multipliers based on current emotion"""
        return self.emotion_difficulty_map.get(
            self.current_emotion, 
            self.emotion_difficulty_map["Neutral"]
        )
    
    def spawn_obstacle(self):
        """Spawn a new obstacle"""
        multipliers = self.get_difficulty_multipliers()
        
        x = np.random.randint(30, self.width - 30)
        y = -30
        speed = self.obstacle_speed * multipliers["speed_mult"]
        size = np.random.randint(30, 60)
        
        self.obstacles.append({
            "pos": [x, y],
            "speed": speed,
            "size": size
        })
    
    def spawn_powerup(self):
        """Spawn a power-up"""
        x = np.random.randint(30, self.width - 30)
        y = -30
        
        self.powerups.append({
            "pos": [x, y],
            "speed": 3,
            "size": 25,
            "type": np.random.choice(["life", "score", "shield"])
        })
    
    def update_player(self):
        """Update player position"""
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            self.player_pos[0] -= self.player_speed
        if keys[pygame.K_RIGHT]:
            self.player_pos[0] += self.player_speed
        if keys[pygame.K_UP]:
            self.player_pos[1] -= self.player_speed
        if keys[pygame.K_DOWN]:
            self.player_pos[1] += self.player_speed
        
        # Keep player in bounds
        self.player_pos[0] = max(self.player_size, min(self.width - self.player_size, self.player_pos[0]))
        self.player_pos[1] = max(self.player_size, min(self.height - self.player_size, self.player_pos[1]))
    
    def check_collisions(self):
        """Check collisions"""
        player_rect = pygame.Rect(
            self.player_pos[0] - self.player_size // 2,
            self.player_pos[1] - self.player_size // 2,
            self.player_size,
            self.player_size
        )
        
        # Check obstacle collisions
        for obstacle in self.obstacles[:]:
            obs_rect = pygame.Rect(
                obstacle["pos"][0] - obstacle["size"] // 2,
                obstacle["pos"][1] - obstacle["size"] // 2,
                obstacle["size"],
                obstacle["size"]
            )
            
            if player_rect.colliderect(obs_rect):
                self.obstacles.remove(obstacle)
                self.lives -= 1
                if self.lives <= 0:
                    self.running = False
        
        # Check powerup collisions
        for powerup in self.powerups[:]:
            pow_rect = pygame.Rect(
                powerup["pos"][0] - powerup["size"] // 2,
                powerup["pos"][1] - powerup["size"] // 2,
                powerup["size"],
                powerup["size"]
            )
            
            if player_rect.colliderect(pow_rect):
                self.powerups.remove(powerup)
                if powerup["type"] == "life":
                    self.lives = min(5, self.lives + 1)
                elif powerup["type"] == "score":
                    self.score += 50
                elif powerup["type"] == "shield":
                    self.score += 25
    
    def update(self):
        """Update game state"""
        # Process camera frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            emotion_data = self.detector.get_emotion(frame)
            
            if "face" in emotion_data:
                self.current_emotion = emotion_data["smoothed_emotion"]
                self.detector.draw_face_rectangle(
                    frame, 
                    emotion_data["face"],
                    self.current_emotion,
                    emotion_data["smoothed_confidence"]
                )
                
                # Log emotion
                self.emotion_log.append({
                    "time": time.time() - self.game_start_time,
                    "emotion": self.current_emotion,
                    "confidence": emotion_data["smoothed_confidence"]
                })
            
            # Display camera feed
            cv2.imshow("Emotion Recognition", frame)
        
        # Update player
        self.update_player()
        
        # Spawn obstacles
        multipliers = self.get_difficulty_multipliers()
        spawn_rate = self.obstacle_spawn_rate * multipliers["spawn_rate_mult"]
        
        self.spawn_timer += 1
        if self.spawn_timer > spawn_rate:
            self.spawn_obstacle()
            self.spawn_timer = 0
        
        # Spawn powerups occasionally
        self.powerup_timer += 1
        if self.powerup_timer > 300:
            self.spawn_powerup()
            self.powerup_timer = 0
        
        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle["pos"][1] += obstacle["speed"]
            
            if obstacle["pos"][1] > self.height + 50:
                self.obstacles.remove(obstacle)
                self.score += 5
        
        # Update powerups
        for powerup in self.powerups[:]:
            powerup["pos"][1] += powerup["speed"]
            
            if powerup["pos"][1] > self.height + 50:
                self.powerups.remove(powerup)
        
        # Check collisions
        self.check_collisions()
    
    def draw(self):
        """Draw game state"""
        self.screen.fill(self.BG_COLOR)
        
        # Draw player
        multipliers = self.get_difficulty_multipliers()
        player_color = multipliers["color"]
        pygame.draw.circle(self.screen, player_color, self.player_pos, self.player_size // 2)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(
                self.screen,
                self.OBSTACLE_COLOR,
                (obstacle["pos"][0] - obstacle["size"] // 2,
                 obstacle["pos"][1] - obstacle["size"] // 2,
                 obstacle["size"],
                 obstacle["size"])
            )
        
        # Draw powerups
        for powerup in self.powerups:
            color = (0, 255, 0) if powerup["type"] == "life" else self.POWERUP_COLOR
            pygame.draw.circle(
                self.screen,
                color,
                [int(powerup["pos"][0]), int(powerup["pos"][1])],
                powerup["size"] // 2
            )
        
        # Draw UI
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font.render(f"Lives: {self.lives}", True, (255, 100, 100))
        self.screen.blit(lives_text, (10, 50))
        
        emotion_text = self.small_font.render(f"Emotion: {self.current_emotion}", True, player_color)
        self.screen.blit(emotion_text, (10, 90))
        
        # Draw difficulty indicator
        diff_text = self.small_font.render("Difficulty adapts to your emotion!", True, (200, 200, 200))
        self.screen.blit(diff_text, (self.width - 350, 10))
        
        # Draw emotion effects
        effects_y = 40
        effects = [
            f"Happy: Faster & More Challenges",
            f"Sad: Slower & Easier",
            f"Angry: Very Fast & Intense",
        ]
        for effect in effects:
            effect_text = self.small_font.render(effect, True, (150, 150, 150))
            self.screen.blit(effect_text, (self.width - 350, effects_y))
            effects_y += 25
        
        # Draw controls
        controls = [
            "Arrow Keys: Move",
            "ESC: Quit"
        ]
        y_offset = self.height - 80
        for control in controls:
            control_text = self.small_font.render(control, True, (150, 150, 150))
            self.screen.blit(control_text, (10, y_offset))
            y_offset += 25
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("Emotion-Adaptive Game")
        print("The game adapts to your emotions!")
        print("Use arrow keys to move, avoid red obstacles")
        print("Press ESC to quit")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Check if camera window was closed
            if cv2.getWindowProperty("Emotion Recognition", cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
            
            self.update()
            self.draw()
            self.clock.tick(30)
        
        # Save emotion log
        with open("emotion_game_log.json", "w") as f:
            json.dump(self.emotion_log, f, indent=2)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        
        print(f"\nGame Over!")
        print(f"Final Score: {self.score}")
        print(f"Emotion log saved to emotion_game_log.json")


if __name__ == "__main__":
    game = AdaptiveEmotionGame()
    game.run()
