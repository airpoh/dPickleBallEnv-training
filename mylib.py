import numpy as np
import cv2
import os
from collections import deque
from gym import Env, spaces
from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import A2C
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack=64, img_size=(168, 84), grayscale=True):
        # —— Unity env setup ——
        self.env = UnityParallelEnv(unity_env)

        # left agent 0, right agent 1
        self.agent = self.env.possible_agents[1]       # agent to be controlled (right)
        self.agent_other = self.env.possible_agents[0] # opponent (left)
        self.agent_obs = self.env.possible_agents[0]   # camera index used for observations (left)

        # —— pixel-frame settings ——
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        # Observation space
        base_obs = self.env.observation_spaces[self.agent_obs][0]
        c, h, w = base_obs.shape
        self._transpose = (c == 3)

        # Final obs shape after manual preprocessing
        if grayscale:
            obs_shape = (frame_stack, img_size[1], img_size[0])  # (stack, H, W)
        else:
            obs_shape = (frame_stack * c, img_size[1], img_size[0])  # (stack*C, H, W)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = self.env.action_spaces[self.agent]

        # —— reward-shaping parameters ——
        self.prev_game_state = None
        self.steps_since_serve = 0
        self.max_fast = 100
        self.hold_streak = 0
        self.break_streak = 0
        self.violation_occurred = False
        self.phi = {0: 0.0, 1: -0.1, 2: +0.1, 3: +0.5, 4: -0.5, 5: -1.2, 6: +1.2}

        # Load pretrained left-agent (opponent) model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Try multiple possible paths for the opponent model
        possible_paths = [
            "./leftmodel.zip",  # Zipped version in root
            "./Images/leftmodel",  # Extracted version in Images folder
            "./leftmodel",  # Extracted version in root
        ]
        
        self.opponent_model = None
        for left_model_path in possible_paths:
            if os.path.exists(left_model_path):
                try:
                    print(f"Loading pretrained left-agent model from {left_model_path} on device {device}")
                    # Load with explicit device and no environment to avoid memory issues
                    self.opponent_model = A2C.load(
                        left_model_path, 
                        device=device,
                        env=None  # Don't load with environment to save memory
                    )
                    self.opponent_model.policy.to(device)
                    self.opponent_model.set_env(None)  # Set env to None for inference
                    print("Successfully loaded opponent model!")
                    break
                except MemoryError as e:
                    print(f"Memory error loading opponent model from {left_model_path}: {e}")
                    print("Skipping opponent model - will use random actions instead")
                    continue
                except Exception as e:
                    print(f"Failed to load from {left_model_path}: {e}")
                    continue
        
        if self.opponent_model is None:
            print("Warning: Opponent model not loaded. Training without opponent model.")
            print("The agent will use random actions for the opponent (this is fine for training).")

    def _mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Horizontally flip a (F,H,W) or (1,H,W) tensor.
        """
        mirrored = obs[..., ::-1]
        return mirrored

    def _preprocess(self, obs):
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        # Resize
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        # Grayscale or keep color
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        # Normalize to [0,1]
        obs = obs.astype(np.float32) / 255.0
        # Mirror observation for left-agent camera
        obs = self._mirror_obs(obs)
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()
        raw = obs_dict[self.agent_obs]['observation'][0]
        img = self._preprocess(np.asarray(raw, dtype=np.float32))

        # Initialize frame stack with mirrored first frame
        for _ in range(self.frame_stack):
            self.frames.append(img)

        # Reset reward shaping variables
        self.prev_game_state = None
        self.steps_since_serve = 0
        self.hold_streak = 0
        self.break_streak = 0
        self.violation_occurred = False

        return np.concatenate(list(self.frames), axis=0), {}  # (stack, H, W)

    def mirror_action(self, action: np.ndarray) -> np.ndarray:
        """Mirror horizontal action component: swap left↔right (1↔2)"""
        mirrored = action.copy()
        if mirrored[1] == 1:
            mirrored[1] = 2
        elif mirrored[1] == 2:
            mirrored[1] = 1
        return mirrored

    def get_opponent_action(self):
        """
        Use the pretrained A2C model (trained on right-court) to act as the left agent.
        Mirror the stacked frames back into right-court view for inference,
        then un-mirror the action output.
        """
        if self.opponent_model is None:
            # Return random action if no opponent model (for initial training)
            return np.array([0, 0, 0], dtype=np.int32)
        
        obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)
        # Flip back to right-court view for the model
        obs_for_model = self._mirror_obs(obs)
        obs_batch = np.expand_dims(obs_for_model, axis=0)  # (1, stack, H, W)
        action, _states = self.opponent_model.predict(obs_batch, deterministic=True)
        # Un-mirror the horizontal component of action
        return self.mirror_action(action[0]).astype(np.int32)

    def step(self, action):
        # Send both right-agent action and left-agent (opponent) action
        actions = {self.agent: action,
                   self.agent_other: self.get_opponent_action()}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        raw_list = obs_dict[self.agent_obs]['observation']
        raw_img = raw_list[0]
        vec_array = raw_list[1]

        # Preprocess & mirror
        img = self._preprocess(np.asarray(raw_img, dtype=np.float32))
        self.frames.append(img)
        stacked = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        # Compute shaped reward
        base_r = rewards[self.agent] - rewards[self.agent_other]
        game_state = int(np.array(vec_array, dtype=np.float32)[0])

        # Potential-based reward shaping
        gamma = 0.99
        phi_cur = self.phi.get(self.prev_game_state, 0.0)
        phi_next = self.phi[game_state]
        shaped = base_r + (gamma * phi_next - phi_cur)

        # Additional state-based bonuses (added, not replacing)
        if game_state == 0:
            shaped += 0.05
        if game_state == 1:
            shaped -= 0.02
        if game_state == 2:
            shaped += 0.05
        if game_state == 3:
            shaped += 0.8
        if game_state == 4:
            shaped -= 0.4
        if game_state == 5:
            shaped -= 1.0
        if game_state == 6:
            shaped += 2.0

        # Serve-break/lost bonuses
        if self.prev_game_state == 1 and game_state == 6:
            shaped += 0.5  # serve_break_bonus
        elif self.prev_game_state == 2 and game_state == 5:
            shaped -= 0.5  # serve_lost_penalty

        # Speed incentive
        if game_state == 2:
            self.steps_since_serve = 0
        elif self.steps_since_serve is not None:
            self.steps_since_serve += 1

        if game_state == 6 and self.prev_game_state == 2:
            shaped += 0.4 * (1 - self.steps_since_serve / self.max_fast)
        elif game_state == 5 and self.prev_game_state == 2:
            shaped -= 0.3 * (1 - self.steps_since_serve / self.max_fast)

        # Streak tracking
        if self.prev_game_state == 2 and game_state == 6:
            self.hold_streak += 1
            self.break_streak = 0
            shaped += 0.15 * min(self.hold_streak, 10)
        elif self.prev_game_state == 1 and game_state == 5:
            self.break_streak += 1
            self.hold_streak = 0
            shaped -= 0.1 * min(self.break_streak, 5)

        # Violation tracking
        if game_state in (1, 2):
            self.violation_occurred = False
        elif game_state in (3, 4):
            self.violation_occurred = True
        elif game_state in (5, 6) and not self.violation_occurred:
            shaped += 0.2

        # Small penalty for prolonged rallies
        if game_state not in (5, 6):
            shaped -= 0.005

        self.prev_game_state = game_state
        done = terminations[self.agent]
        total_r = base_r + shaped

        if (rewards[self.agent] + rewards[self.agent_other]) > 0:
            print("Rewards: ", total_r, rewards[self.agent_other])

        return (stacked, total_r, done, False, infos[self.agent])

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # typically 4 for stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the output size of CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Final linear layer to get to desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
