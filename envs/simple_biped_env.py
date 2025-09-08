import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np

class simpleBipedEnv(gym.Env):
    def __init__(self, render=False, max_steps=10000):
        super(simpleBipedEnv, self).__init__()
        self.render_mode = render
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF("walkers/simple_biped.urdf", [0, 0, 1.0])
        self.plane_id = p.loadURDF("plane.urdf")

        # Find foot link indices (adjust names to match your URDF!)
        self.left_foot = 1   # replace with actual index
        self.right_foot = 3  # replace with actual index

        num_joints = 4
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2*num_joints + 6,), dtype=np.float32)
        self.max_steps = max_steps
        self.current_step = 0
        self.last_action = None
        self.off_ground_counter = 0   # track consecutive steps with no contacts

    def _feet_in_contact(self):
        left_contacts = p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.left_foot)
        right_contacts = p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.right_foot)
        return len(left_contacts) > 0, len(right_contacts) > 0

    def _compute_reward(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        forward_reward = base_pos[0]
        alive_bonus = 1 if not self._check_termination() else -500

        # --- Penalize if both feet off ground for too long ---
        left_contact, right_contact = self._feet_in_contact()
        if not left_contact and not right_contact:
            self.off_ground_counter += 1
        else:
            self.off_ground_counter = 0

        airborne_penalty = -10 if self.off_ground_counter > 500 else 0  # tune threshold & penalty

        return forward_reward + alive_bonus + airborne_penalty



    def reset(self, seed=None, options=None):
        p.resetBasePositionAndOrientation(self.robot_id, [0,0,1], [0,0,0,1])
        num_joints = p.getNumJoints(self.robot_id)
        for j in range(num_joints):
            p.resetJointState(self.robot_id, j, targetValue=0.0, targetVelocity=0.0)
        p.setGravity(0, 0, -9.8)
        plane_id = p.loadURDF("plane.urdf")
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        num_joints = p.getNumJoints(self.robot_id)
        if np.isscalar(action) or (isinstance(action, np.ndarray) and action.shape == (1,)):
            action = np.full((num_joints,), action if np.isscalar(action) else action[0], dtype=np.float32)
        else:
            action = np.asarray(action, dtype=np.float32)
            if action.shape[0] != num_joints:
                raise ValueError(f"Action shape {action.shape} does not match number of joints {num_joints}")

        self.last_action = action

        # Get current joint positions
        joint_states = p.getJointStates(self.robot_id, range(num_joints))
        current_positions = np.array([state[0] for state in joint_states], dtype=np.float32)

        # Apply relative change to joint angles
        delta = action * 0.2  # scale step size as needed
        target_positions = current_positions + delta

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=list(range(num_joints)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[200] * num_joints  # max force
        )
        p.stepSimulation()
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_termination()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return obs, reward, done, truncated, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        return np.array(joint_positions + joint_velocities + list(base_pos) + list(base_vel), dtype=np.float32)


    def _check_termination(self):
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        return base_pos[2] < 0.2  or base_pos[2] > 1.5  # fallen

    def render(self):
        if self.render_mode:
            if hasattr(self, 'robot_id'):
                base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=base_pos
                )
        else:
            pass

    def close(self):
        p.disconnect()
