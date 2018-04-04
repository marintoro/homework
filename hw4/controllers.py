import numpy as np
from cost_functions import trajectory_cost_fn
import time
import gym

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env
		discrete = isinstance(env.action_space, gym.spaces.Discrete)
		self.act_dim = env.action_space.n if discrete else env.action_space.shape[0]
		pass

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return np.random.uniform(low=-1, high=1, size=self.act_dim)
		pass


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		states_tab = []
		actions_tab = []
		next_states_tab = []
		current_states = np.array([state for i in range(self.num_simulated_paths)])
		for time_step in range(self.horizon):
			random_actions = np.random.uniform(low=-1, high=1, size=(self.num_simulated_paths, self.act_dim))
			states_tab.append(current_states)
			actions_tab.append(random_actions)
			next_states = self.dyn_model.predict(current_states, random_actions)
			next_states_tab.append(next_states)
			current_states = next_states

		actions_trajectories = np.array(actions_tab)
		states_trajectories = np.array(states_tab)
		next_states_trajectories = np.array(next_states_tab)

		best_score = -1000000000 #C est bien dégeu!!!!
		best_action = None
		for trajectory_number in range(actions_trajectories.shape[1]):
			current_score = self.cost_fn(states_trajectories[:,trajectory_number,:], actions_trajectories[:,trajectory_number,:], next_states_trajectories[:,trajectory_number+1,:])
			if current_score > best_score:
				best_score = current_score
				best_action = actions_trajectories[:,trajectory_number,:][0]

		return best_action

