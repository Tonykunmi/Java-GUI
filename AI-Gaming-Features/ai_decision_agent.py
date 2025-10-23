"""
AI Agent for Real-Time Decision-Making in Video Games
Uses a neural network-based decision system with state evaluation.
"""

import numpy as np
import pygame
import random
from typing import List, Tuple, Dict
import json


class GameState:
    """Represents the current state of the game"""
    
    def __init__(self, player_health: int, enemy_health: int, 
                 player_position: Tuple[int, int], enemy_position: Tuple[int, int],
                 resources: int, distance: float):
        self.player_health = player_health
        self.enemy_health = enemy_health
        self.player_position = player_position
        self.enemy_position = enemy_position
        self.resources = resources
        self.distance = distance
    
    def to_vector(self) -> np.ndarray:
        """Convert game state to feature vector for neural network"""
        return np.array([
            self.player_health / 100.0,  # Normalize to 0-1
            self.enemy_health / 100.0,
            self.player_position[0] / 800.0,
            self.player_position[1] / 600.0,
            self.enemy_position[0] / 800.0,
            self.enemy_position[1] / 600.0,
            self.resources / 100.0,
            self.distance / 1000.0
        ])


class DecisionNetwork:
    """Simple neural network for decision making"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, output_size: int = 5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias2 = np.zeros((1, output_size))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward(self, state_vector: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        if len(state_vector.shape) == 1:
            state_vector = state_vector.reshape(1, -1)
        
        # Hidden layer
        hidden = self.relu(np.dot(state_vector, self.weights1) + self.bias1)
        
        # Output layer
        output = np.dot(hidden, self.weights2) + self.bias2
        
        return self.softmax(output)
    
    def predict_action(self, state_vector: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict the best action given current state"""
        probabilities = self.forward(state_vector)
        action = np.argmax(probabilities[0])
        return action, probabilities[0]


class AIDecisionAgent:
    """AI Agent that makes real-time decisions in a game environment"""
    
    ACTIONS = {
        0: "ATTACK",
        1: "DEFEND",
        2: "RETREAT",
        3: "COLLECT_RESOURCES",
        4: "ADVANCE"
    }
    
    def __init__(self):
        self.network = DecisionNetwork()
        self.decision_history = []
        self.state_history = []
    
    def evaluate_state(self, game_state: GameState) -> Dict[str, float]:
        """Evaluate the current game state"""
        state_vector = game_state.to_vector()
        
        # Calculate various metrics
        health_ratio = game_state.player_health / max(game_state.enemy_health, 1)
        threat_level = self.calculate_threat_level(game_state)
        resource_need = 1.0 - (game_state.resources / 100.0)
        
        return {
            "health_ratio": health_ratio,
            "threat_level": threat_level,
            "resource_need": resource_need,
            "distance": game_state.distance
        }
    
    def calculate_threat_level(self, game_state: GameState) -> float:
        """Calculate threat level based on enemy proximity and health"""
        distance_factor = max(0, 1.0 - game_state.distance / 500.0)
        health_factor = game_state.enemy_health / 100.0
        return distance_factor * health_factor
    
    def make_decision(self, game_state: GameState) -> Tuple[str, Dict]:
        """Make a decision based on current game state"""
        state_vector = game_state.to_vector()
        action_id, probabilities = self.network.predict_action(state_vector)
        action = self.ACTIONS[action_id]
        
        # Evaluate state
        evaluation = self.evaluate_state(game_state)
        
        # Store decision history
        decision_info = {
            "action": action,
            "action_id": action_id,
            "probabilities": {self.ACTIONS[i]: float(prob) for i, prob in enumerate(probabilities)},
            "evaluation": evaluation,
            "state": {
                "player_health": game_state.player_health,
                "enemy_health": game_state.enemy_health,
                "resources": game_state.resources,
                "distance": game_state.distance
            }
        }
        
        self.decision_history.append(decision_info)
        self.state_history.append(state_vector)
        
        return action, decision_info
    
    def update_weights(self, reward: float, learning_rate: float = 0.001):
        """Update network weights based on reward (simple reinforcement)"""
        if len(self.state_history) == 0:
            return
        
        # Simple gradient update (this is a simplified version)
        adjustment = reward * learning_rate
        self.network.weights2 += adjustment * np.random.randn(*self.network.weights2.shape) * 0.01
    
    def save_decision_history(self, filename: str = "decision_history.json"):
        """Save decision history to file"""
        with open(filename, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get statistics about decisions made"""
        if not self.decision_history:
            return {}
        
        action_counts = {}
        for decision in self.decision_history:
            action = decision["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_decisions": len(self.decision_history),
            "action_distribution": action_counts,
            "average_threat_level": np.mean([d["evaluation"]["threat_level"] for d in self.decision_history])
        }


class GameSimulation:
    """Simple game simulation to demonstrate the AI agent"""
    
    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI Decision Agent Demo")
        self.clock = pygame.time.Clock()
        
        # Game entities
        self.player_pos = [width // 4, height // 2]
        self.enemy_pos = [3 * width // 4, height // 2]
        self.player_health = 100
        self.enemy_health = 100
        self.resources = 50
        
        # AI Agent
        self.agent = AIDecisionAgent()
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.running = True
        self.current_action = "NONE"
        self.action_color = (255, 255, 255)
    
    def get_current_state(self) -> GameState:
        """Get current game state"""
        distance = np.sqrt(
            (self.player_pos[0] - self.enemy_pos[0])**2 +
            (self.player_pos[1] - self.enemy_pos[1])**2
        )
        
        return GameState(
            player_health=self.player_health,
            enemy_health=self.enemy_health,
            player_position=tuple(self.player_pos),
            enemy_position=tuple(self.enemy_pos),
            resources=self.resources,
            distance=distance
        )
    
    def update(self):
        """Update game state and AI decision"""
        state = self.get_current_state()
        action, decision_info = self.agent.make_decision(state)
        self.current_action = action
        
        # Execute action
        if action == "ATTACK":
            self.action_color = (255, 0, 0)  # Red
            if state.distance < 200:
                self.enemy_health = max(0, self.enemy_health - 2)
                self.player_health = max(0, self.player_health - 1)
        elif action == "DEFEND":
            self.action_color = (0, 0, 255)  # Blue
            self.player_health = min(100, self.player_health + 0.5)
        elif action == "RETREAT":
            self.action_color = (255, 255, 0)  # Yellow
            dx = self.player_pos[0] - self.enemy_pos[0]
            dy = self.player_pos[1] - self.enemy_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                self.player_pos[0] += int(dx / length * 3)
                self.player_pos[1] += int(dy / length * 3)
        elif action == "COLLECT_RESOURCES":
            self.action_color = (0, 255, 0)  # Green
            self.resources = min(100, self.resources + 1)
        elif action == "ADVANCE":
            self.action_color = (255, 165, 0)  # Orange
            dx = self.enemy_pos[0] - self.player_pos[0]
            dy = self.enemy_pos[1] - self.player_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                self.player_pos[0] += int(dx / length * 2)
                self.player_pos[1] += int(dy / length * 2)
        
        # Keep player in bounds
        self.player_pos[0] = max(20, min(self.width - 20, self.player_pos[0]))
        self.player_pos[1] = max(20, min(self.height - 20, self.player_pos[1]))
        
        # Random enemy movement
        self.enemy_pos[0] += random.randint(-2, 2)
        self.enemy_pos[1] += random.randint(-2, 2)
        self.enemy_pos[0] = max(20, min(self.width - 20, self.enemy_pos[0]))
        self.enemy_pos[1] = max(20, min(self.height - 20, self.enemy_pos[1]))
    
    def draw(self):
        """Draw the game state"""
        self.screen.fill((20, 20, 40))
        
        # Draw player (blue circle)
        pygame.draw.circle(self.screen, (0, 150, 255), self.player_pos, 20)
        
        # Draw enemy (red circle)
        pygame.draw.circle(self.screen, (255, 50, 50), self.enemy_pos, 20)
        
        # Draw health bars
        # Player health
        pygame.draw.rect(self.screen, (255, 0, 0), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, (0, 255, 0), (10, 10, int(200 * self.player_health / 100), 20))
        health_text = self.font.render(f"Player Health: {self.player_health:.0f}", True, (255, 255, 255))
        self.screen.blit(health_text, (10, 35))
        
        # Enemy health
        pygame.draw.rect(self.screen, (255, 0, 0), (self.width - 210, 10, 200, 20))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.width - 210, 10, int(200 * self.enemy_health / 100), 20))
        enemy_text = self.font.render(f"Enemy Health: {self.enemy_health:.0f}", True, (255, 255, 255))
        self.screen.blit(enemy_text, (self.width - 210, 35))
        
        # Draw resources
        resource_text = self.font.render(f"Resources: {self.resources:.0f}", True, (255, 255, 255))
        self.screen.blit(resource_text, (10, 65))
        
        # Draw current action
        action_text = self.font.render(f"Action: {self.current_action}", True, self.action_color)
        self.screen.blit(action_text, (self.width // 2 - 80, 10))
        
        # Draw distance
        state = self.get_current_state()
        distance_text = self.small_font.render(f"Distance: {state.distance:.0f}", True, (200, 200, 200))
        self.screen.blit(distance_text, (self.width // 2 - 60, 40))
        
        # Draw statistics
        stats = self.agent.get_statistics()
        if stats:
            y_offset = 100
            stats_title = self.small_font.render("Decision Statistics:", True, (255, 255, 255))
            self.screen.blit(stats_title, (10, y_offset))
            y_offset += 25
            
            for action, count in stats.get("action_distribution", {}).items():
                stat_text = self.small_font.render(f"  {action}: {count}", True, (200, 200, 200))
                self.screen.blit(stat_text, (10, y_offset))
                y_offset += 20
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("AI Decision Agent Demo")
        print("Watch the AI make real-time decisions!")
        print("Press ESC to quit")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            self.update()
            self.draw()
            self.clock.tick(30)  # 30 FPS
        
        # Save decision history
        self.agent.save_decision_history("ai_decision_history.json")
        print(f"\nFinal Statistics:")
        stats = self.agent.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        pygame.quit()


if __name__ == "__main__":
    game = GameSimulation()
    game.run()
