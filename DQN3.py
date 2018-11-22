import tensorflow as tf
import numpy as np
import random
import shutil
import os
import stat
import errno
import datetime
import pickle as pkl

class DQN:
    def __init__(self, obs_dim, num_actions, layers = [3,3,3], learning_rate = 0.01, batch_size = 1, e_greedy = 0.9, gamma = 0.9, memory_size = 500, board = False, logs_path = 'Logs', init_name = '' ):
        self.obs_dim= obs_dim
        self.num_actions = num_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.age = 0
        self.board = board

        self.s = tf.placeholder(tf.float32, [None, obs_dim])
        self.q_target = tf.placeholder(tf.float32, [None, num_actions])

        dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        real_logs_path = logs_path + '/' + dt
        if board:
            self.writer = tf.summary.FileWriter(real_logs_path, graph=tf.get_default_graph())

        self.memory = {'s':np.zeros([memory_size, obs_dim]),
                       'a':np.zeros([memory_size]),
                       'r':np.zeros([memory_size]),
                       's_':np.zeros([memory_size, obs_dim]),
                       'counter':0}

        hidden = self.s
        for i, num_unit in enumerate(layers):
            hidden = tf.layers.dense(hidden, num_unit, kernel_initializer=tf.random_normal_initializer(0, 0.3), bias_initializer= tf.constant_initializer(0.1), name=init_name + 'hidden' + str(i))
            hidden = self.leaky_relu(hidden, leakiness=0.1, name=init_name + str(i))

        self.output = tf.layers.dense(hidden, num_actions, kernel_initializer=tf.random_normal_initializer(0, 0.3), bias_initializer= tf.constant_initializer(0.1), name=init_name + 'output_predicted')

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.output))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def watch_and_learn(self, s, action, reward, s_):
        counter = self.memory['counter']
        self.memory['s'][counter, :] = s
        self.memory['a'][counter] = action
        self.memory['r'][counter] = reward
        self.memory['s_'][counter, :] = s_
        if counter > self.batch_size:
            self.learn_from_memory()
        if counter == self.memory_size - 1:
            counter = 0
        else:
            counter += 1

        self.memory['counter'] = counter

    def learn_from_memory(self):
        batch_indexes = np.random.choice(self.memory['counter'], self.batch_size)
        batch_s = self.memory['s'][batch_indexes,:]
        batch_a = self.memory['a'][batch_indexes]
        batch_r = self.memory['r'][batch_indexes]
        batch_s_ = self.memory['s_'][batch_indexes,:]

        q_predicted = self.sess.run(self.output, feed_dict={self.s:batch_s})
        q_predicted_next = self.sess.run(self.output, feed_dict={self.s:batch_s_})

        q_target = q_predicted.copy()
        for i in range(self.batch_size):
            q_target[i, int(batch_a[i])] = batch_r[i] + self.gamma * np.max(q_predicted_next[i])

        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.s:batch_s, self.q_target:q_target})

        weights = np.asarray(self.sess.run(tf.get_default_graph().get_tensor_by_name("hidden0/kernel:0")))
        norm0 = np.linalg.norm(weights[0,:])
        norm1 = np.linalg.norm(weights[1,:])
        norm2 = np.linalg.norm(weights[2,:])
        # norm3 = np.linalg.norm(weights[3,:])
        # norm4 = np.linalg.norm(weights[4,:])

        w20 = weights[1,0]
        if self.board:
            if self.memory['counter'] == self.memory_size - 1:
                summary = tf.Summary(value=[tf.Summary.Value(tag="norm0", simple_value=norm0),
                                            tf.Summary.Value(tag="norm1", simple_value=norm1),
                                            tf.Summary.Value(tag="norm2", simple_value=norm2),
                                            # tf.Summary.Value(tag="norm3", simple_value=norm3),
                                            # tf.Summary.Value(tag="norm4", simple_value=norm4),
                                            tf.Summary.Value(tag="w20", simple_value=w20)])
                self.writer.add_summary(summary, self.age)

        if self.memory['counter'] == self.memory_size - 1:
            self.age += 1
            print('happy new year : ' + str(self.age))

        return loss
        # print(self.sess.run(self.loss, feed_dict={self.s: batch_s, self.q_target:q_target}))
    def choose_action(self, observation):
        if random.random() > self.e_greedy:
            best_action = np.random.randint(0, self.num_actions)
        else:
            observation_reshaped = np.reshape(observation, [1, len(observation)])
            q_pred = self.sess.run(self.output, feed_dict={self.s:observation_reshaped})
            best_action = np.argmax(q_pred[0,:])

        return best_action

    def leaky_relu(self, x, leakiness=0.2, name = ''):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name = 'leaky_relu'+name)

    def save_memory(self, path='SavedMemories/memory.pkl'):
        with open(path, 'wb') as f:
            pkl.dump([self.memory], f)

    def load_memories(self, memory_list = []):
        total_size = 0
        memories = []
        for i, mem in enumerate(memory_list):
            print(mem)
            with open(mem, 'rb') as f:
                [memory] = pkl.load(f)
                memories.append(memory)
                this_size = memory['s'].shape[0]
                total_size += this_size

        total_memory = {'s':np.zeros([total_size, self.obs_dim]),
                        'a':np.zeros([total_size]),
                        'r':np.zeros([total_size]),
                        's_':np.zeros([total_size, self.obs_dim]),
                        'counter':0}
        prev_size = 0
        this_size = 0
        for memory in memories:
            this_size += memory['s'].shape[0]
            total_memory['s'][prev_size:this_size,:] = memory['s']
            total_memory['a'][prev_size:this_size] = memory['a']
            total_memory['r'][prev_size:this_size] = memory['r']
            total_memory['s_'][prev_size:this_size,:] = memory['s_']
            prev_size = this_size

        print(total_size)
        print(total_memory['s'])

        return total_memory

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        print('\nModel has been saved!')

    def load_model(self, sess, path):
        saver = tf.train.Saver()
        self.sess = sess
        saver.restore(self.sess, path)
        print('\nModel loaded successfully')