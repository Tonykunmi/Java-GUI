"""
Reinforcement Learning-based NPC Behavior Model
Uses Deep Q-Learning for intelligent NPC decision-making in a simulation game.
"""

import numpy as np
import pygame
import random
from typing import List, Tuple, Dict, Optional
from collections import deque
import json
import pickle


class NeuralNetwork:
    """Simple neural network for Q-learning"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))
        
        # For Adam optimizer
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.t = 0
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        
        return self.z3
    
    def copy_weights_from(self, other_network):
        """Copy weights from another network"""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()
        self.W3 = other_network.W3.copy()
        self.b3 = other_network.b3.copy()


class ReplayBuffer:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample a batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for NPC behavior"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update_freq = 100
        
        # Networks
        self.policy_net = NeuralNetwork(state_size, 128, action_size)
        self.target_net = NeuralNetwork(state_size, 128, action_size)
        self.target_net.copy_weights_from(self.policy_net)
        
        # Replay buffer
        self.memory = ReplayBuffer(10000)
        
        # Training stats
        self.train_step = 0
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.policy_net.forward(state)
        return np.argmax(q_values[0])
    
    def train(self):
        """Train the network on a batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Compute Q targets
        current_q = self.policy_net.forward(states)
        next_q = self.target_net.forward(next_states)
        
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Compute loss and gradients (simplified)
        loss = np.mean((current_q - target_q) ** 2)
        
        # Simple gradient descent update
        delta = (current_q - target_q) / self.batch_size
        self._backward(states, delta)
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.policy_net)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def _backward(self, states: np.ndarray, delta: np.ndarray):
        """Simplified backward pass"""
        # Gradient descent on output layer
        grad_W3 = np.dot(self.policy_net.a2.T, delta)
        grad_b3 = np.sum(delta, axis=0, keepdims=True)
        
        # Update with learning rate
        self.policy_net.W3 -= self.learning_rate * grad_W3
        self.policy_net.b3 -= self.learning_rate * grad_b3
    
    def save(self, filename: str):
        """Save the agent"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'policy_net': self.policy_net,
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards
            }, f)
    
    def load(self, filename: str):
        """Load the agent"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.policy_net = data['policy_net']
                self.target_net.copy_weights_from(self.policy_net)
                self.epsilon = data['epsilon']
                self.episode_rewards = data['episode_rewards']
            print(f"Loaded agent from {filename}")
        except FileNotFoundError:
            print(f"No saved agent found at {filename}")


class NPC:
    """Non-Player Character with RL behavior"""
    
    ACTIONS = {
        0: "IDLE",
        1: "MOVE_UP",
        2: "MOVE_DOWN",
        3: "MOVE_LEFT",
        4: "MOVE_RIGHT",
        5: "ATTACK",
        6: "DEFEND",
        7: "COLLECT"
    }
    
    def __init__(self, x: int, y: int, npc_id: int, agent: DQNAgent):
        self.pos = [x, y]
        self.id = npc_id
        self.agent = agent
        self.health = 100
        self.energy = 100
        self.resources = 0
        self.speed = 3
        
        # Visual
        self.color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
        self.size = 20
        
        # State tracking
        self.last_action = 0
        self.reward_total = 0
    
    def get_state(self, game_state: Dict) -> np.ndarray:
        """Get current state vector for RL"""
        # Normalize positions
        norm_x = self.pos[0] / game_state["width"]
        norm_y = self.pos[1] / game_state["height"]
        
        # Find nearest resource
        resources = game_state.get("resources", [])
        if resources:
            nearest_resource = min(resources, key=lambda r: 
                np.sqrt((r[0] - self.pos[0])**2 + (r[1] - self.pos[1])**2))
            resource_dx = (nearest_resource[0] - self.pos[0]) / game_state["width"]
            resource_dy = (nearest_resource[1] - self.pos[1]) / game_state["height"]
        else:
            resource_dx, resource_dy = 0, 0
        
        # Find nearest threat
        enemies = game_state.get("enemies", [])
        if enemies:
            nearest_enemy = min(enemies, key=lambda e:
                np.sqrt((e[0] - self.pos[0])**2 + (e[1] - self.pos[1])**2))
            enemy_dx = (nearest_enemy[0] - self.pos[0]) / game_state["width"]
            enemy_dy = (nearest_enemy[1] - self.pos[1]) / game_state["height"]
        else:
            enemy_dx, enemy_dy = 0, 0
        
        return np.array([
            norm_x, norm_y,
            self.health / 100.0,
            self.energy / 100.0,
            self.resources / 100.0,
            resource_dx, resource_dy,
            enemy_dx, enemy_dy
        ], dtype=np.float32)
    
    def execute_action(self, action: int, game_state: Dict) -> float:
        """Execute action and return reward"""
        reward = -0.1  # Small negative reward for each action (encourages efficiency)
        
        action_name = self.ACTIONS[action]
        
        if action_name == "MOVE_UP":
            self.pos[1] = max(0, self.pos[1] - self.speed)
            self.energy = max(0, self.energy - 0.5)
        elif action_name == "MOVE_DOWN":
            self.pos[1] = min(game_state["height"], self.pos[1] + self.speed)
            self.energy = max(0, self.energy - 0.5)
        elif action_name == "MOVE_LEFT":
            self.pos[0] = max(0, self.pos[0] - self.speed)
            self.energy = max(0, self.energy - 0.5)
        elif action_name == "MOVE_RIGHT":
            self.pos[0] = min(game_state["width"], self.pos[0] + self.speed)
            self.energy = max(0, self.energy - 0.5)
        elif action_name == "ATTACK":
            self.energy = max(0, self.energy - 5)
            reward += 0.5  # Small reward for attacking
        elif action_name == "DEFEND":
            self.energy = max(0, self.energy - 2)
            reward += 0.3
        elif action_name == "COLLECT":
            # Check if near resource
            resources = game_state.get("resources", [])
            for resource in resources:
                dist = np.sqrt((resource[0] - self.pos[0])**2 + (resource[1] - self.pos[1])**2)
                if dist < 30:
                    self.resources += 10
                    self.energy = min(100, self.energy + 5)
                    reward += 10  # Large reward for collecting
                    game_state["resources"].remove(resource)
                    break
        elif action_name == "IDLE":
            self.energy = min(100, self.energy + 1)
            reward += 0.1
        
        # Penalty for low health/energy
        if self.health < 20:
            reward -= 1
        if self.energy < 10:
            reward -= 0.5
        
        # Reward for having resources
        reward += self.resources * 0.01
        
        self.last_action = action
        self.reward_total += reward
        
        return reward
    
    def draw(self, screen: pygame.Surface):
        """Draw the NPC"""
        pygame.draw.circle(screen, self.color, [int(self.pos[0]), int(self.pos[1])], self.size)
        
        # Draw health bar
        bar_width = 40
        bar_height = 5
        health_width = int((self.health / 100.0) * bar_width)
        pygame.draw.rect(screen, (255, 0, 0), 
                        (self.pos[0] - bar_width//2, self.pos[1] - 30, bar_width, bar_height))
        pygame.draw.rect(screen, (0, 255, 0),
                        (self.pos[0] - bar_width//2, self.pos[1] - 30, health_width, bar_height))


class RLSimulationGame:
    """Simulation game with RL-controlled NPCs"""
    
    def __init__(self, num_npcs: int = 5):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL NPC Behavior Simulation")
        self.clock = pygame.time.Clock()
        
        # Create DQN agent (shared by all NPCs)
        self.agent = DQNAgent(state_size=9, action_size=8)
        
        # Try to load saved agent
        self.agent.load("npc_agent.pkl")
        
        # Create NPCs
        self.npcs = []
        for i in range(num_npcs):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            self.npcs.append(NPC(x, y, i, self.agent))
        
        # Resources
        self.resources = []
        self.spawn_resources(20)
        
        # Enemies (simple obstacles)
        self.enemies = []
        self.spawn_enemies(5)
        
        # Training
        self.training_mode = True
        self.episode = 0
        self.episode_steps = 0
        self.max_episode_steps = 1000
        
        # Stats
        self.total_reward = 0
        self.running = True
        
        # Fonts
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
    
    def spawn_resources(self, count: int):
        """Spawn resources"""
        for _ in range(count):
            x = random.randint(30, self.width - 30)
            y = random.randint(30, self.height - 30)
            self.resources.append([x, y])
    
    def spawn_enemies(self, count: int):
        """Spawn enemies"""
        for _ in range(count):
            x = random.randint(30, self.width - 30)
            y = random.randint(30, self.height - 30)
            self.enemies.append([x, y])
    
    def get_game_state(self) -> Dict:
        """Get current game state"""
        return {
            "width": self.width,
            "height": self.height,
            "resources": self.resources,
            "enemies": self.enemies
        }
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode += 1
        self.episode_steps = 0
        
        # Record episode reward
        if self.npcs:
            avg_reward = sum(npc.reward_total for npc in self.npcs) / len(self.npcs)
            self.agent.episode_rewards.append(avg_reward)
        
        # Reset NPCs
        for npc in self.npcs:
            npc.pos = [random.randint(50, self.width - 50), random.randint(50, self.height - 50)]
            npc.health = 100
            npc.energy = 100
            npc.resources = 0
            npc.reward_total = 0
        
        # Reset resources
        self.resources = []
        self.spawn_resources(20)
        
        # Save agent periodically
        if self.episode % 10 == 0:
            self.agent.save("npc_agent.pkl")
            print(f"Episode {self.episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {self.agent.epsilon:.3f}")
    
    def update(self):
        """Update game state"""
        game_state = self.get_game_state()
        
        # Update each NPC
        for npc in self.npcs:
            # Get state
            state = npc.get_state(game_state)
            
            # Select action
            action = self.agent.select_action(state, self.training_mode)
            
            # Execute action and get reward
            reward = npc.execute_action(action, game_state)
            
            # Get next state
            next_state = npc.get_state(game_state)
            
            # Store experience
            done = self.episode_steps >= self.max_episode_steps
            self.agent.memory.push(state, action, reward, next_state, done)
            
            # Train
            if self.training_mode and len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train()
        
        # Respawn resources randomly
        if len(self.resources) < 10 and random.random() < 0.05:
            self.spawn_resources(1)
        
        # Update episode
        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            self.reset_episode()
    
    def draw(self):
        """Draw game state"""
        self.screen.fill((20, 30, 40))
        
        # Draw resources (yellow circles)
        for resource in self.resources:
            pygame.draw.circle(self.screen, (255, 255, 0), resource, 8)
        
        # Draw enemies (red squares)
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, (255, 50, 50),
                           (enemy[0] - 15, enemy[1] - 15, 30, 30))
        
        # Draw NPCs
        for npc in self.npcs:
            npc.draw(self.screen)
        
        # Draw UI
        episode_text = self.font.render(f"Episode: {self.episode}", True, (255, 255, 255))
        self.screen.blit(episode_text, (10, 10))
        
        steps_text = self.font.render(f"Steps: {self.episode_steps}/{self.max_episode_steps}",
                                     True, (255, 255, 255))
        self.screen.blit(steps_text, (10, 40))
        
        epsilon_text = self.small_font.render(f"Exploration: {self.agent.epsilon:.3f}",
                                             True, (200, 200, 200))
        self.screen.blit(epsilon_text, (10, 70))
        
        mode_text = self.small_font.render(
            f"Mode: {'Training' if self.training_mode else 'Testing'}",
            True, (0, 255, 0) if self.training_mode else (255, 255, 0)
        )
        self.screen.blit(mode_text, (10, 95))
        
        # Draw NPC stats
        y_offset = 140
        for i, npc in enumerate(self.npcs):
            stats_text = self.small_font.render(
                f"NPC {i}: Res={npc.resources} HP={npc.health:.0f} E={npc.energy:.0f} Act={NPC.ACTIONS[npc.last_action]}",
                True, npc.color
            )
            self.screen.blit(stats_text, (10, y_offset))
            y_offset += 22
        
        # Draw recent rewards
        if len(self.agent.episode_rewards) > 0:
            recent_rewards = self.agent.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            reward_text = self.small_font.render(
                f"Avg Reward (last 10): {avg_reward:.2f}",
                True, (255, 200, 0)
            )
            self.screen.blit(reward_text, (10, self.height - 60))
        
        # Draw legend
        legend_y = self.height - 120
        legend_items = [
            ("Yellow circles = Resources", (255, 255, 0)),
            ("Red squares = Enemies", (255, 50, 50)),
            ("Colored circles = NPCs", (100, 200, 255))
        ]
        for text, color in legend_items:
            legend_text = self.small_font.render(text, True, color)
            self.screen.blit(legend_text, (self.width - 280, legend_y))
            legend_y += 22
        
        # Controls
        controls_text = self.small_font.render("Press T to toggle training mode | ESC to quit",
                                              True, (150, 150, 150))
        self.screen.blit(controls_text, (self.width // 2 - 200, self.height - 30))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("RL NPC Behavior Simulation")
        print("Watch NPCs learn to collect resources!")
        print("Press T to toggle training mode")
        print("Press ESC to quit")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_t:
                        self.training_mode = not self.training_mode
                        print(f"Training mode: {self.training_mode}")
            
            self.update()
            self.draw()
            self.clock.tick(30)
        
        # Save agent before exit
        self.agent.save("npc_agent.pkl")
        print(f"\nTraining completed!")
        print(f"Total episodes: {self.episode}")
        print(f"Agent saved to npc_agent.pkl")
        
        pygame.quit()


if __name__ == "__main__":
    game = RLSimulationGame(num_npcs=5)
    game.run()
