import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np

class WalkerEnv(gym.Env):
    def __init__(self, render=False):
        super(WalkerEnv, self).__init__()
        self.render_mode = render
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("walkers/simple_biped.urdf", [0, 0, 1.0])
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        print(f"Robot spawned at position: {base_pos}")
        return self._get_obs(), {}

    def step(self, action):
        # apply motor torques or joint control here
        # p.setJointMotorControlArray(...)
        p.stepSimulation()
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_termination()
        return obs, reward, done, False, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        # For simple_biped.urdf, only one joint: hip_joint
        # So obs = [position, velocity], shape=(2,)
        return np.array(joint_positions + joint_velocities, dtype=np.float32)

    def _compute_reward(self):
        # reward forward movement, penalize falls
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        return base_pos[0]  # reward forward x-motion

    def _check_termination(self):
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        return base_pos[2] < 0.5  # fallen

    def render(self):
        if self.render_mode:
            # Focus camera on robot's current position
            if hasattr(self, 'robot_id'):
                base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=base_pos
                )
            # Rendering handled by PyBullet GUI
        else:
            pass  # No rendering in DIRECT mode

    def close(self):
        p.disconnect()
