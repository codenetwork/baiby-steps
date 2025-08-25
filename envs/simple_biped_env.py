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
        self.robot_id = p.loadURDF("walkers/simple_biped.urdf", [0, 0, 1.0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        num_joints = 4  # left_hip, right_hip, left_knee, right_knee
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2*num_joints + 6,), dtype=np.float32)
        self.max_steps = max_steps
        self.current_step = 0
        self.last_action = None



    def reset(self, seed=None, options=None):
        p.resetBasePositionAndOrientation(self.robot_id, [0,0,1], [0,0,0,1])
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
        delta = action * 0.05  # scale step size as needed
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
        return base_pos[2] < 0.2 or base_pos[2] > 1  # fallen

    def _compute_reward(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, _ = p.getBaseVelocity(self.robot_id)

        forward_reward = base_vel[0]        # reward x velocity
        if not self._check_termination(): 
            alive_bonus = 1
        else: 
            alive_bonus = -500
        torque_penalty = 0.001 * np.sum(np.square(self.last_action))

        #print(forward_reward, alive_bonus, torque_penalty)

        return forward_reward + alive_bonus - torque_penalty


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
