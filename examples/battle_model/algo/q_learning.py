import os
import tensorflow as tf
import numpy as np
import pdb

from . import base
from . import tools


class MsgDQN(base.ValueNet):
    def __init__(self, sess, name, handle, env, len_nei,
                 sub_len, msg_bits=5, memory_size=2**10,
                 batch_size=64, update_every=5):
        super().__init__(sess, env, handle, name, len_nei,
                         update_every=update_every, is_msg=True)
        self.msg_bits = msg_bits
        self.replay_buffer = tools.MemoryGroup(
            self.view_space, self.feature_space, self.msg_bits, memory_size, batch_size, sub_len)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def test_buffer(self):
        self.replay_buffer.test_buffer()

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        for i in range(batch_num):
            obs, feat, acts, obs_next, feat_next, rewards, dones, masks,  \
                act_prob, act_prob_next, msgs, msgs_next, mean_msg, mean_msg_next, \
                poss, poss_next, chs, chs_next = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards,
                                          dones=dones, prob=act_prob_next, mean_msg=mean_msg_next,
                                          msgs=msgs_next)
            loss, q = super().train(state=[obs, feat], target_q=target_q,
                                    prob=act_prob, acts=acts, masks=masks, mean_msg=mean_msg, msgs=msgs)
            if i % self.update_every == 0:
                self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "Msgdqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "Msgdqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))


class DQN(base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len,
                 len_nei, memory_size=2**10, batch_size=64, update_every=5):
        super().__init__(sess, env, handle, name, len_nei, update_every=update_every)

        self.replay_buffer = tools.MemoryGroup(
            self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()
        for i in range(batch_num):
            obs, feat, acts, obs_next, feat_next, rewards, dones, masks,  act_prob, \
                act_prob_next, msgs, msgs_next, mean_msg, mean_msg_next, \
                poss, poss_next, chs, chs_next = self.replay_buffer.sample()
            target_q = rewards.copy()

            if np.sum(dones) != len(dones):
                target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards,
                                              dones=dones, prob=act_prob_next, mean_msg=mean_msg_next,
                                              msgs=msgs_next, poss=poss_next, chs=chs_next)

            loss, q = super().train(state=[obs, feat], target_q=target_q,
                                    prob=act_prob, acts=acts, masks=masks, mean_msg=mean_msg, msgs=msgs)
            if i % self.update_every == 0:
                self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))

    def test_buffer(self):
        self.replay_buffer.test_buffer()


class GDQN(base.GCrpNet):
    def __init__(self, sess, name, handle, env, sub_len,
                 len_nei=6, memory_size=2**10, batch_size=64, update_every=5, isHie=False):

        super().__init__(sess, env, handle, name, len_nei=len_nei,
                         update_every=update_every, isHie=isHie)

        self.replay_buffer = tools.MemoryGroup(
            self.view_space, self.feature_space, self.num_actions,
            memory_size, batch_size, sub_len, needPoss=True, needChs=True)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feat, acts, obs_next, feat_next, rewards, dones, masks, act_prob, \
                act_prob_next, msgs, msgs_next, mean_msg, mean_msg_next, \
                poss, poss_next, chs, chs_next = self.replay_buffer.sample()

            fl = False
            if batch_num < 500 and i % 100 == 0:
                fl = True
            if rewards[0] > 5:
                print(rewards)
            target_q = rewards.copy()
            if np.sum(dones) != len(dones):
                target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards,
                                              dones=dones, prob=act_prob_next, mean_msg=mean_msg_next,
                                              msgs=msgs_next, poss=poss_next, chs=chs_next, fl=fl)
            loss, q = super().train(obs=obs, target_q=target_q, prob=act_prob, acts=acts,
                                    masks=masks, mean_msg=mean_msg, msgs=msgs, poss=poss, chs=chs, alt=0)

            if i % self.update_every == 0:
                self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def test_buffer(self):
        self.replay_buffer.test_buffer()

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "gqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "gqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))
