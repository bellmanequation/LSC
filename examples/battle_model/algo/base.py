import tensorflow as tf
import numpy as np
import pdb
from tensorflow.nn.rnn_cell import GRUCell

from magent.gridworld import GridWorld
import time
import math
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import sonnet as snt
import sonnet as snt
from tensorflow.contrib import rnn
import sklearn.preprocessing as preprocessing


max_len = 20


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def make_conv_model2(nums):
    return snt.Sequential([
        snt.nets.ConvNet2D(output_channels=[32, 32], kernel_shapes=[3, 3],
                           strides=[1, 1], paddings=['VALID', 'VALID'], activate_final=True),
        snt.BatchFlatten(),
        snt.nets.MLP([256, 128, 64, nums], activate_final=False),
    ])


class ValueNet:
    def __init__(self, sess, env, handle, name, len_nei, update_every=5,
                 use_mf=False, learning_rate=1e-4, tau=0.99, gamma=0.95,
                 num_bits_msg=3, msg_space=(40, 40), use_msg=False,
                 is_msg=False, use_mean_msg=False, is_hie=False, crp=False):
        self.env = env
        self.name = name
        self._saver = None
        self.sess = sess

        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        if is_msg:
            self.num_actions = num_bits_msg

        self.num_bits_msg = num_bits_msg
        self.msg_space = msg_space

        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.use_msg = use_msg
        self.use_mean_msg = use_mean_msg
        self.temperature = 0.1

        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.crp = crp
        with tf.variable_scope(name or "ValueNet"):
            self.name_scope = tf.get_variable_scope().name
            self.obs_input = tf.placeholder(
                tf.float32, (None,) + self.view_space, name="Obs-Input")
            self.feat_input = tf.placeholder(
                tf.float32, (None,) + self.feature_space, name="Feat-Input")
            self.mask = tf.placeholder(
                tf.float32, shape=(None,), name='Terminate-Mask')

            if self.use_mf:
                self.act_prob_input = tf.placeholder(
                    tf.float32, (None, self.num_actions), name="Act-Prob-Input")
            if self.use_msg:
                self.msgs_input = tf.placeholder(
                    tf.float32, (None,)+(self.num_bits_msg, self.msg_space[0], self.msg_space[1]), name="Msg-Input")

            if self.use_mean_msg:
                self.mean_msg_input = tf.placeholder(
                    tf.float32, (None, self.num_bits_msg), name="Mean-Msg-Input")

            self.act_input = tf.placeholder(tf.int32, (None,), name="Act")
            self.act_one_hot = tf.one_hot(
                self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0)
            self.batch_num = tf.placeholder(tf.float32, (None,))
            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name
                self.e_q = self._construct_net(active_func=tf.nn.relu)
                self.e_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)

            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_q = self._construct_net(active_func=tf.nn.relu)
                self.t_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            with tf.variable_scope("Update"):
                self.update_op = [tf.assign(self.t_variables[i],
                                            self.tau * self.e_variables[i] + (1. - self.tau) * self.t_variables[i])
                                  for i in range(len(self.t_variables))]

            with tf.variable_scope("Optimization"):
                self.target_q_input = tf.placeholder(
                    tf.float32, (None,), name="Q-Input")
                self.e_q_max = tf.reduce_sum(tf.multiply(
                    self.act_one_hot, self.e_q), axis=1)
                self.loss = tf.reduce_sum(
                    tf.square(self.target_q_input - self.e_q_max))/tf.reduce_sum(self.batch_num)
                self.train_op = tf.train.AdamOptimizer(
                    self.lr).minimize(self.loss)

    def _construct_net(self, active_func=None, reuse=False):

        mod = make_conv_model2(self.num_actions)
        return mod(self.obs_input)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'prob', 'dones', 'rewards'}
        """
        feed_dict = {
            self.obs_input: kwargs['obs'],
            self.feat_input: kwargs['feature']
        }

        t_q, e_q = self.sess.run([self.t_q, self.e_q], feed_dict=feed_dict)
        act_idx = np.argmax(e_q, axis=1)
        q_values = t_q[np.arange(len(t_q)), act_idx]
        q_values = q_values.reshape(-1)
        q_v = []
        i = 0
        for k in 1. - np.array(kwargs['dones']):
            if k:
                q_v.append(q_values[i])
                i += 1
            else:
                q_v.append(0)
        q_v = np.array(q_v)
        target_q_value = kwargs['rewards'] + q_v * self.gamma

        return target_q_value

    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        feed_dict = {self.obs_input: kwargs['obs']}

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['obs'])
            feed_dict[self.act_prob_input] = kwargs['prob']

        if self.use_msg:
            feed_dict[self.msgs_input] = kwargs['msgs']

        if self.use_mean_msg:
            feed_dict[self.mean_msg_input] = kwargs['mean_msg']

        q_values = self.sess.run(self.e_q, feed_dict=feed_dict)

        switch = np.random.uniform()

        if switch < kwargs['eps']:
            actions = np.random.choice(
                self.num_actions, len(kwargs['obs'])).astype(np.int32)
        else:
            actions = np.argmax(q_values, axis=1).astype(np.int32)
        return actions

    def get_q(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1]
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input] = kwargs['prob']

        if self.use_msg:
            feed_dict[self.msgs_input] = kwargs['msgs']

        if self.use_mean_msg:
            feed_dict[self.mean_msg_input] = kwargs['mean_msg']

        q_values = self.sess.run(self.e_q, feed_dict=feed_dict)
        return np.max(q_values, axis=1)

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1],
            self.target_q_input: kwargs['target_q'],
            self.mask: kwargs['masks']
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        if self.use_msg:
            feed_dict[self.msgs_input] = kwargs['msgs']

        if self.use_mean_msg:
            feed_dict[self.mean_msg_input] = kwargs['mean_msg']

        feed_dict[self.act_input] = kwargs['acts']
        feed_dict[self.batch_num] = np.ones_like(kwargs['target_q'])
        _, loss, e_q = self.sess.run(
            [self.train_op, self.loss, self.e_q_max], feed_dict=feed_dict)
        return loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}

    def get_feedback(self, **kwargs):
        kwargs['dones'] = np.array(
            [not done for done in kwargs['alives']], dtype=np.bool)
        q_values = self.get_q(**kwargs)
        tar_q = self.calc_target_q(**kwargs)
        return q_values-tar_q


def cons_datas(num_acts):
    datadicts_tmpl = [{
        "obs": [[[[0.0 for i in range(6)] for i in range(13)]for i in range(13)]for i in range(3)],
        "q": [[0.1 for i in range(num_acts)] for i in range(3)],
        "act": [0 for i in range(3)],
        "globals": [0.1, 0, 0],
        "nodes": [[0.1 for i in range(256)] for j in range(3)],
        "edges": [[101. for i in range(9)], [201. for i in range(9)]],
        "senders": [0, 2],
        "receivers": [1, 1],
        "hedges": [[0.1 for i in range(9)]],
        "hsenders": [0],
        "hreceivers": [1],
        "ledges": [[0.1 for i in range(9)]],
        "lsenders": [0],
        "lreceivers": [1],
        "lnodes": [[0.1 for i in range(32)] for i in range(3)],
        "hnodes": [[0.1 for i in range(64)] for i in range(3)]
    }]
    return datadicts_tmpl


class GCrpNet:
    def __init__(self, sess, env, handle, name, len_nei=6,
                 update_every=5, learning_rate=1e-4, tau=0.99,
                 gamma=0.95, num_bits_msg=3, isHie=False, is_comm=False):
        self.env = env
        self.name = name
        self._saver = None
        self.sess = sess

        self.isHie = isHie
        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.num_bits_msg = num_bits_msg

        self.update_every = update_every

        self.len_nei = len_nei
        self.temperature = 0.1
        self.act_input = tf.placeholder(tf.int32, (None,), name="Act")
        self.act_one_hot = tf.one_hot(
            self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0)
        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.is_comm = is_comm
        with tf.variable_scope(name or "GHCrpNet"):
            self.name_scope = tf.get_variable_scope().name
            self.input_ph = utils_tf.placeholders_from_data_dicts(
                cons_datas(self.num_actions))
            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name

                self.e_graph = self._construct_net()
                self.e_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)
            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_graph = self._construct_net()
                self.t_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            with tf.variable_scope("Update"):
                self.update_op = [tf.assign(self.t_variables[i],
                                            self.tau * self.e_variables[i] + (1. - self.tau) * self.t_variables[i])
                                  for i in range(len(self.t_variables))]

            with tf.variable_scope("Optimization"):
                self.target_q_input = tf.placeholder(
                    tf.float32, (None,), name="Q-Input")
                self.e_q_max = tf.reduce_sum(tf.multiply(
                    self.act_one_hot, self.e_graph.q), axis=1)
                self.loss = tf.reduce_sum(tf.square(self.target_q_input - self.e_q_max)) / \
                    tf.cast(tf.reduce_sum(self.e_graph.n_node), tf.float32)
                self.train_op = tf.train.AdamOptimizer(
                    self.lr).minimize(self.loss)

    def get_indexs(self, poss):
        nei_len = self.len_nei
        nei_space = nei_len**2
        poss_all = np.array([poss]*len(poss))
        poss_mine = np.array([[t]*len(poss) for t in poss])
        indexs = ((poss_all-poss_mine)**2).sum(axis=2)
        indexs = indexs-np.ones_like(indexs)*nei_space
        indexs[indexs < 0] = 0
        return indexs

    def _construct_net(self):
        if self.is_comm:
            model = models.GcommNetwork()
        else:
            if self.num_actions == 13:
                model = models.GCrpNetworkTiny()
            else:
                model = models.GCrpNetwork()
        return model(self.input_ph)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def cons_graph(self, obs, poss, chs=None, act=None):
        mini = np.zeros((52, 52))
        for i in range(len(chs)):
            mini[int(poss[chs[i]][0]) + 6, int(poss[chs[i]][1]) + 6] = 1
        tmp = []
        for i in range(len(poss)):
            tmp.append(mini[int(poss[i][0]):int(poss[i][0]) +
                            13, int(poss[i][1]):int(poss[i][1]+13)])
        tmp = np.reshape(tmp, (len(obs), 13, 13, 1))
        obs = np.concatenate((obs, tmp), axis=-1)

        datadict = {}
        datadict["obs"] = obs
        datadict["senders"] = []
        datadict["receivers"] = []
        datadict["edges"] = []
        datadict["lsenders"] = []
        datadict["lreceivers"] = []
        datadict["ledges"] = []
        datadict["hedges"] = []
        datadict["hsenders"] = []
        datadict["hreceivers"] = []

        indexs = self.get_indexs(poss)
        datadict['nodes'] = [[0 for j in range(256)]for i in range(len(obs))]
        datadict['lnodes'] = [[0 for j in range(32)]for i in range(len(obs))]
        datadict['hnodes'] = [[0 for j in range(64)]for i in range(len(obs))]
        datadict["globals"] = [0, 0, 0]
        datadict['q'] = [[0 for i in range(self.num_actions)]
                         for i in range(len(obs))]
        chs = list(set(chs))
        if chs is not None:
            for ch in chs:
                for ind in np.where(indexs[ch] == 0)[0]:
                    datadict["senders"].append(ind)
                    datadict["receivers"].append(ch)
                    datadict["lsenders"].append(ch)
                    datadict["lreceivers"].append(ind)
                    datadict["ledges"].append([0 for i in range(9)])
                    datadict["edges"].append([0 for i in range(9)])
            for ch in chs:
                for c in chs:
                    if ch != c:
                        datadict["hedges"].append([0 for i in range(9)])
                        datadict["hsenders"].append(ch)
                        datadict["hreceivers"].append(c)

        if len(datadict['edges']) == 0:
            datadict["edges"] = np.zeros(shape=(0, 9))
        if len(datadict["hedges"]) == 0:
            datadict["hedges"] = np.zeros(shape=(0, 9))
        if len(datadict["ledges"]) == 0:
            datadict["ledges"] = np.zeros(shape=(0, 9))
        if act is not None:
            datadict["act"] = act
        else:
            datadict["act"] = [0 for i in range(len(obs))]
        datadicts = [datadict]
        graphtuple = utils_np.data_dicts_to_graphs_tuple(datadicts)
        return graphtuple

    def cons_allcomm_graph(self, obs, poss, chs=None, act=None):
        datadict = {}
        datadict["obs"] = obs
        datadict["senders"] = []
        datadict["receivers"] = []
        datadict["edges"] = []
        datadict["lsenders"] = []
        datadict["lreceivers"] = []
        datadict["ledges"] = []
        datadict["hedges"] = []
        datadict["hsenders"] = []
        datadict["hreceivers"] = []

        indexs = self.get_indexs(poss)
        datadict['nodes'] = [[0 for j in range(256)]for i in range(len(obs))]
        datadict['lnodes'] = [[0 for j in range(32)]for i in range(len(obs))]
        datadict['hnodes'] = [[0 for j in range(64)]for i in range(len(obs))]
        datadict["globals"] = [0, 0, 0]
        datadict['q'] = [[0 for i in range(self.num_actions)]
                         for i in range(len(obs))]
        datadict['hreceivers'].append(0)
        datadict['hsenders'].append(0)
        datadict['hedges'].append([0, 0, 0])
        for i in range(len(obs)):
            for j in range(len(obs)):
                datadict['senders'].append(i)
                datadict['receivers'].append(j)
                datadict['lsenders'].append(j)
                datadict['lreceivers'].append(i)
                datadict["ledges"].append([0, 0, 0])
                datadict["edges"].append([0, 0, 0])
        if len(datadict['edges']) == 0:
            datadict["edges"] = np.zeros(shape=(0, 3))
        if len(datadict["hedges"]) == 0:
            datadict["hedges"] = np.zeros(shape=(0, 3))
        if len(datadict["ledges"]) == 0:
            datadict["ledges"] = np.zeros(shape=(0, 3))
        if act is not None:
            datadict["act"] = act
        else:
            datadict["act"] = [0 for i in range(len(obs))]
        datadicts = [datadict]
        graphtuple = utils_np.data_dicts_to_graphs_tuple(datadicts)
        return graphtuple

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'prob', 'dones', 'rewards'}
        """
        if self.isHie:
            feed_dict = self.cons_allcomm_graph(
                obs=kwargs['obs'], poss=kwargs['poss'], chs=kwargs['chs'])
        else:
            feed_dict = self.cons_graph(
                obs=kwargs['obs'], poss=kwargs['poss'], chs=kwargs['chs'])
        t_res = self.sess.run(self.t_graph, feed_dict={
                              self.input_ph: feed_dict})
        e_res = self.sess.run(self.e_graph, feed_dict={
                              self.input_ph: feed_dict})
        act_idx = np.argmax(e_res.q, axis=1)
        t_q = t_res.q
        q_values = t_q[np.arange(len(t_q)), act_idx]
        q_values = q_values.reshape(-1)
        q_v = []
        i = 0
        for k in 1.-kwargs['dones']:
            if k:
                q_v.append(q_values[i])
                i += 1
            else:
                q_v.append(0)
        q_v = np.array(q_v)
        target_q_value = kwargs['rewards'] + q_v * self.gamma

        return target_q_value

    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        if self.isHie:
            feed_dict = self.cons_allcomm_graph(
                obs=kwargs["obs"], poss=kwargs["poss"], chs=kwargs["chs"])
        else:
            feed_dict = self.cons_graph(
                obs=kwargs["obs"], poss=kwargs["poss"], chs=kwargs["chs"])
        e_res = self.sess.run(self.e_graph, feed_dict={
                              self.input_ph: feed_dict})

        switch = np.random.uniform()
        tmp = np.random.rand()
        if tmp < 0.005:
            print(e_res.edges[0])
            if e_res.edges.shape[0] > 2:
                print(e_res.edges[1])
        if switch < kwargs['eps']:
            actions = np.random.choice(
                self.num_actions, e_res.n_node).astype(np.int32)
        else:
            actions = np.argmax(e_res.q, axis=1).astype(np.int32)
        return actions

    def get_e_res(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        if self.isHie:
            feed_dict = self.cons_allcomm_graph(
                obs=kwargs["obs"], poss=kwargs["poss"], chs=kwargs["chs"])
        else:
            feed_dict = self.cons_graph(
                obs=kwargs["obs"], poss=kwargs["poss"], chs=kwargs["chs"])
        e_res = self.sess.run(self.e_graph, feed_dict={
                              self.input_ph: feed_dict})
        return e_res

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        if self.isHie:
            feed_dict = self.cons_allcomm_graph(
                obs=kwargs["obs"], poss=kwargs["poss"], act=kwargs["acts"], chs=kwargs["chs"])
        else:
            feed_dict = self.cons_graph(
                obs=kwargs["obs"], poss=kwargs["poss"], act=kwargs["acts"], chs=kwargs["chs"])

        _, loss, e_q = self.sess.run([self.train_op, self.loss, self.e_q_max],
                                     feed_dict={self.input_ph: feed_dict,
                                                self.target_q_input: kwargs['target_q'],
                                                self.act_input: kwargs['acts']})
        return loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}
