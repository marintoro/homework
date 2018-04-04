import tensorflow as tf
import numpy as np
import gym

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

def normalize(data, mean, std):
    return (data - mean)/(std+ 1e-7)

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.normalization = normalization
        self.batch_size = batch_size
        self.sess = sess
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        self.obs_placeholder = tf.placeholder(shape=[None, ob_dim], name="observations", dtype=tf.float32)
        self.acs_placeholder = tf.placeholder(shape=[None, ac_dim], name="actions", dtype=tf.float32)
        input_placeholder = tf.concat([self.obs_placeholder, self.acs_placeholder], axis = 1)
        self.iterations = iterations
        self.predicted_value = build_mlp(input_placeholder, ob_dim, "dynamic_model", n_layers=n_layers, size=size,
                               activation=activation, output_activation=output_activation)

        self.target_placeholder = tf.placeholder(shape=[None, ob_dim], name="targets_delta", dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.predicted_value - self.target_placeholder))
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        normalization = self.normalization

        observations = np.concatenate([path['observations'] for path in data])
        actions = np.concatenate([path['actions'] for path in data])
        deltas = np.concatenate([path['next_observations'] - path['observations'] for path in data])

        observations_norm = normalize(observations, normalization["mean_obs"], normalization["std_obs"])
        actions_norm = normalize(actions, normalization["mean_deltas"], normalization["std_deltas"])
        deltas_norm = normalize(deltas, normalization["mean_actions"], normalization["std_actions"])
        assert observations_norm.shape[0] == actions_norm.shape[0] == deltas_norm.shape[0]
        shuffled_indices = np.arange(observations.shape[0])

        sess = self.sess

        for num_iter in range(self.iterations):
            np.random.shuffle(shuffled_indices)
            for num_batch in range(int(np.ceil(observations.shape[0]/self.batch_size))):
                current_batch_obs = observations_norm[shuffled_indices[num_batch:(num_batch+1)*self.batch_size]]
                current_batch_acs = actions_norm[shuffled_indices[num_batch:(num_batch + 1) * self.batch_size]]
                current_batch_deltas = deltas_norm[shuffled_indices[num_batch:(num_batch + 1) * self.batch_size]]


                sess.run([self.update_op], feed_dict={self.obs_placeholder : current_batch_obs,
                                                      self.acs_placeholder : current_batch_acs,
                                                      self.target_placeholder : current_batch_deltas})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        sess = self.sess
        normalization = self.normalization
        observations_norm = normalize(states, normalization["mean_obs"], normalization["std_obs"])
        actions_norm = normalize(actions, normalization["mean_actions"], normalization["std_actions"])
        predicted_delta_norm = sess.run(self.predicted_value, feed_dict={self.obs_placeholder : observations_norm,
                                                      self.acs_placeholder : actions_norm})

        return states + (predicted_delta_norm * normalization["std_deltas"]) + normalization["mean_deltas"]

