import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        x_dist = abs(self.target_pos[0] - self.sim.pose[0]) 
        y_dist = abs(self.target_pos[1] - self.sim.pose[1]) # Distance to target in y-axis 
        z_dist = abs(self.target_pos[2] - self.sim.pose[2]) # Distance to target in z-axis
        y_velocity = self.sim.v[1] # Velocity in y 
        z_velocity = self.sim.v[2] # Velocity in z
        
        reward = 1.-.3*(z_dist).sum()
        reward = max(1, min(-1, reward))
        
#         dist = ((y_dist ** 2) + (z_dist ** 2)) ** 0.5
        
#         reward = (-10 * x_dist) + (-2 * y_dist) + (-2 * z_dist) - y_velocity - z_velocity + (100 * (1/dist))
        
#         if dist < 20:
#             reward = reward + y_velocity + z_velocity
        
#         if 5 < dist < 10:
#             reward = reward + 50
#         elif dist < 5:
#             reward = reward + 100 + (-5 * y_velocity) + (-5 * z_velocity)
        
        
        #reward = - 0.1 * dist
        #if (dist < 5) and (dist!=0):
        #    reward = (1/dist)
        
        # Penalise any significant movement in x direction
        #if x_dist > 2:
        #    reward = -100
        #if (dist > 2):
        #    reward = - 10 * (1/dist)
        #else:
        #    reward = 100
        #if ((z_velocity + y_velocity) > 5):
        #    reward = reward - 10 * (1/(z_velocity + y_velocity)
        #if (dist <= 2) and ((z_velocity + y_velocity) < 5) and ((z_velocity + y_velocity) > 2):
        #    reward = 1000
        #if (dist <= 2) and ((z_velocity + y_velocity) < 2):
        #    reward = 10000
        
        #if (y_dist > 10) or (z_dist > 10):
        #    reward = - 1
        #elif (y_dist > 10) and (z_dist < 10):
        #    reward = 1
        #elif (y_dist < 10) and (z_dist > 10):
        #    reward = 1   
        #elif (y_dist = 0) or (z_dist = 0):
        #    reward = 10
        #elif (y_dist = 0) and (z_dist = 0):   
        #    reward = 1000
        
        return reward
        
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state