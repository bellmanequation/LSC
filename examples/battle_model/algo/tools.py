import numpy as np
import tensorflow as tf
import os
import pdb


class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'


class Buffer:
    def __init__(self):
        pass

    def push(self, **kwargs):
        raise NotImplementedError


class MetaBuffer(object):
    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):
        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def reset_new(self, start, value):
        self.data[start:] = value


class EpisodesBufferEntry:
    """Entry for episode buffer"""

    def __init__(self):
        self.views = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.terminal = False

    def append(self, view, feature, action, reward, alive, probs=None):
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if probs is not None:
            self.probs.append(probs)
        if not alive:
            self.terminal = True


class EpisodesBuffer(Buffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """

    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view, feature = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        alives = kwargs['alives']
        ids = kwargs['ids']

        if self.use_mean:
            probs = kwargs['prob']

        buffer = self.buffer
        index = np.random.permutation(len(view))

        for i in range(len(ids)):
            i = index[i]
            entry = buffer.get(ids[i])
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[ids[i]] = entry

            if self.use_mean:
                entry.append(view[i], feature[i], acts[i],
                             rewards[i], alives[i], probs=probs[i])
            else:
                entry.append(view[i], feature[i], acts[i],
                             rewards[i], alives[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()


class AgentMemory(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, msg_bits,
                 msg_n, use_mean=False, use_msg=False,
                 use_mean_msg=False, needPoss=False, needChs=False):
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean
        self.use_msg = use_msg
        self.use_mean_msg = use_mean_msg
        self.needPoss = needPoss
        self.needChs = needChs
        self.is_end = False

        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        if self.use_msg:
            self.msgs = MetaBuffer((msg_bits, msg_n[0], msg_n[1]), max_len)
        if self.use_mean_msg:
            self.mean_msg = MetaBuffer((msg_bits,), max_len)
        if self.needPoss:
            self.poss = MetaBuffer((2,), max_len)
        if self.needChs:
            self.chs = MetaBuffer((), max_len, dtype='int32')

    def append(self, obs0, feat0, act, reward, alive,
               prob=None, msgs=None, mean_msg=None, poss=None, chs=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=np.bool))

        if self.use_mean:
            self.prob.append(np.array([prob]))
        if self.use_msg:
            self.msgs.append(np.array([msgs]))
        if self.use_mean_msg:
            self.mean_msg.append(np.array([mean_msg]))
        if self.needPoss:
            self.poss.append(np.array([poss]))
        if self.needChs:
            self.chs.append(np.array([chs]))

    def pull(self):
        res = {
            'obs0': self.obs0.pull(),
            'feat0': self.feat0.pull(),
            'act': self.actions.pull(),
            'poss': None if not self.needPoss else self.poss.pull(),
            'chs': None if not self.needChs else self.chs.pull(),
            'rewards': self.rewards.pull(),
            'terminals': self.terminals.pull(),
            'prob': None if not self.use_mean else self.prob.pull(),
            'msgs': None if not self.use_msg else self.msgs.pull(),
            'mean_msg': None if not self.use_mean_msg else self.mean_msg.pull()
        }

        return res


class DecodeBuffer(object):
    def __init__(self, map_shape, feature_shape, max_len):
        self.agent = dict()
        self.max_len = max_len
        self.map_shape = map_shape
        self.feature_shape = feature_shape
        self.map = MetaBuffer(map_shape, max_len)
        self.feat = MetaBuffer(feature_shape, max_len)

    def _flush(self, **kwargs):
        self.map.append(kwargs['map'])
        self.feat.append(kwargs['feat'])

    def test_buffer(self):
        if self.nb_entries/400 > 10:
            self.agent = dict()

    def push(self, **kwargs):
        self.agent[self]


class MemoryGroup(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, batch_size,
                 sub_len, msg_bits=3, msg_n=10, use_mean=False,
                 use_msg=False, use_mean_msg=False, needPoss=False, needChs=False):
        self.agent = dict()
        self.max_len = max_len
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.feat_shape = feat_shape
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.use_msg = use_msg
        self.use_mean_msg = use_mean_msg
        self.needPoss = needPoss
        self.needChs = needChs
        self.act_n = act_n
        self.msg_bits = msg_bits
        self.msg_n = msg_n
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')

        if use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        if use_msg:
            self.msgs = MetaBuffer((msg_bits, msg_n[0], msg_n[1]), max_len)
        if use_mean_msg:
            self.mean_msg = MetaBuffer((msg_bits,), max_len)
        if needPoss:
            self.poss = MetaBuffer((2,), max_len)
        if needChs:
            self.chs = MetaBuffer((), max_len, dtype='int32')
        self._new_add = 0

    def _flush(self, **kwargs):
        self.obs0.append(kwargs['obs0'])
        self.feat0.append(kwargs['feat0'])
        self.actions.append(kwargs['act'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['terminals'])
        if self.use_mean:
            self.prob.append(kwargs['prob'])

        if self.use_msg:
            self.msgs.append(kwargs['msgs'])
        if self.use_mean_msg:
            self.mean_msg.append(kwargs['mean_msg'])
        if self.needPoss:
            self.poss.append(kwargs['poss'])
        if self.needChs:
            self.chs.append(kwargs['chs'])

        mask = np.where(kwargs['terminals'] == True, False, True)
        self.masks.append(mask)

    def test_buffer(self):
        if self.nb_entries/400 > 18:
            self.agent.clear()

    def push(self, **kwargs):
        self.agent[self.nb_entries] = AgentMemory(
            self.obs_shape, self.feat_shape, self.act_n,
            self.sub_len, self.msg_bits, self.msg_n,
            use_mean=self.use_mean, use_msg=self.use_msg,
            use_mean_msg=self.use_mean_msg, needPoss=self.needPoss,
            needChs=self.needChs)

        if self.needChs:
            l = len(kwargs['chs'])
        for i, _id in enumerate(kwargs['ids']):
            if self.needChs and i >= l:
                j = l-1
            else:
                j = i
            self.agent[self.nb_entries-1].append(
                obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i],
                act=kwargs['acts'][i], reward=kwargs['rewards'][i],
                alive=kwargs['alives'][i],
                prob=kwargs['prob'][i] if self.use_mean else None,
                msgs=kwargs['msgs'][i] if self.use_msg else None,
                mean_msg=kwargs['mean_msg'][i] if self.use_mean_msg else None,
                poss=kwargs['poss'][i] if self.needPoss else None,
                chs=kwargs['chs'][j] if self.needChs else None)
        self._new_add += 1

    def tight(self):
        self.agent[self.nb_entries-1].is_end = True

    def sample(self):
        idx = np.random.choice(self.nb_entries-1, size=1)
        idx = idx[0]
        while self.agent[idx].is_end and np.sum(
                self.agent[idx].terminals.pull()) != len(self.agent[idx].terminals.pull()):
            idx = np.random.choice(self.nb_entries-1, size=1)
            idx = idx[0]

        next_idx = (idx + 1) % self.nb_entries
        obs = self.agent[idx].obs0.pull()
        obs_next = self.agent[next_idx].obs0.pull()
        feature = self.agent[idx].feat0.pull()
        feature_next = self.agent[next_idx].feat0.pull()
        actions = self.agent[idx].actions.pull()
        rewards = self.agent[idx].rewards.pull()

        dones = self.agent[idx].terminals.pull()
        masks = np.where(dones == True, False, True)
        rt = [obs, feature, actions, obs_next,
              feature_next, rewards, dones, masks]

        if self.use_mean:
            act_prob = self.agent[idx].prob.pull()
            act_next_prob = self.agent[next_idx].prob.pull()
            rt.append(act_prob)
            rt.append(act_next_prob)
        else:
            rt.append(None)
            rt.append(None)

        if self.use_msg:
            msgs = self.agent[idx].msgs.pull()
            msgs_next = self.agent[next_idx].msgs.pull()
            rt.append(msgs)
            rt.append(msgs_next)
        else:
            rt.append(None)
            rt.append(None)

        if self.use_mean_msg:
            mean_msg = self.agent[idx].mean_msg.pull()
            mean_msg_next = self.agent[next_idx].mean_msg.pull()
            rt.append(mean_msg)
            rt.append(mean_msg_next)
        else:
            rt.append(None)
            rt.append(None)
        if self.needPoss:
            poss = self.agent[idx].poss.pull()
            poss_next = self.agent[next_idx].poss.pull()
            rt.append(poss)
            rt.append(poss_next)
        else:
            rt.append(None)
            rt.append(None)

        if self.needChs:
            chs = self.agent[idx].chs.pull()
            chs_next = self.agent[next_idx].chs.pull()
            rt.append(chs)
            rt.append(chs_next)
        else:
            rt.append(None)
            rt.append(None)
        return rt

    def get_batch_num(self):
        print('\n[INFO] Length of buffer and new add:',
              len(self.agent), self._new_add)
        res = self._new_add * 2
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.agent)


class SummaryObj:

    """
    Define a summary holder
    """

    def __init__(self, log_dir, log_name, n_group=1):
        self.name_set = set()
        self.gra = tf.Graph()
        self.n_group = n_group

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with self.gra.as_default():
            self.sess = tf.Session(graph=self.gra, config=sess_config)
            self.train_writer = tf.summary.FileWriter(
                log_dir + "/" + log_name, graph=tf.get_default_graph())
            self.sess.run(tf.global_variables_initializer())

    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """

        with self.gra.as_default():
            for name in name_list:
                if name in self.name_set:
                    raise Exception(
                        "You cannot define different operations with same name: `{}`".format(name))
                self.name_set.add(name)
                setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='Agent_{}_{}'.format(i, name))
                                     for i in range(self.n_group)])
                setattr(self, name + "_op", [tf.summary.scalar('Agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                             for i in range(self.n_group)])

    def write(self, summary_dict, step):
        """Write summary related to a certain step

        Parameters
        ----------
        summary_dict: dict, summary value dict
        step: int, global step
        """

        assert isinstance(summary_dict, dict)
        for key, value in summary_dict.items():
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):
                for i in range(self.n_group):
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[i], feed_dict={
                        getattr(self, key)[i]: value[i]}), global_step=step)
            else:
                self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[0], feed_dict={
                    getattr(self, key)[0]: value}), global_step=step)


class Runner(object):
    def __init__(self, sess, env, handles, map_size, max_steps, models, MsgModels,
                 play_handle, render_every=None, save_every=None, tau=None,
                 log_name=None, log_dir=None, model_dir=None, train=False,
                 len_nei=40, rewardtype='self', crp=False,
                 crps=[False, False], is_selfplay=False, is_fix=False):
        """Initialize runner

        Parameters
        ----------
        sess: tf.Session
            session
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        models: list
            contains models
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        model_dir: str
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.MsgModels = MsgModels
        self.max_steps = max_steps
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train
        self.len_nei = len_nei
        self.rewardtype = rewardtype
        self.is_fix = is_fix
        self.crp = crp
        self.crps = crps
        self.update_cnt = 0
        self.is_selfplay = is_selfplay
        if self.train:
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

            summary_items = ['ave_agent_reward', 'total_reward', 'kill', "Sum_Reward",
                             "Kill_Sum", "var", "oppo-ave-agent-reward", 'oppo-total-reward', 'oppo-kill']
            self.summary.register(summary_items)  # summary register
            self.summary_items = summary_items

            assert isinstance(sess, tf.Session)
            assert self.models[0].name_scope != self.models[1].name_scope
            self.sess = sess
            l_vars, r_vars = self.models[0].vars, self.models[1].vars
            if self.MsgModels[0]:
                l_m_vars, r_m_vars = self.MsgModels[0].vars, self.MsgModels[1].vars
                self.m_sp_op = [tf.assign(r_m_vars[i], (1. - tau) * l_m_vars[i] + tau * r_m_vars[i])
                                for i in range(len(l_m_vars))]

            if self.is_selfplay:
                self.sp_op = [tf.assign(r_vars[i], (1. - tau) * l_vars[i] + tau * r_vars[i])
                              for i in range(len(l_vars))]

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def run(self, variant_eps, iteration, win_cnt=None, total_rewards=None):
        info = {'main': None, 'opponent': None}
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.,
                        'oppo-ave-agent-reward': 0., 'oppo-total-reward': 0., 'oppo-kill': 0}
        info['opponent'] = {'ave_agent_reward': 0.,
                            'total_reward': 0., 'kill': 0}

        max_nums, nums, agent_r_records, total_rewards = self.play(
            env=self.env, n_round=iteration, map_size=self.map_size, max_steps=self.max_steps, handles=self.handles,
            models=self.models, MsgModels=self.MsgModels, print_every=50, eps=variant_eps,
            render=(iteration + 1) % self.render_every if self.render_every > 0 else False, train=self.train,
            len_nei=self.len_nei, rewardtype=self.rewardtype, crp=self.crp,
            crps=self.crps, selfplay=self.is_selfplay, is_fix=self.is_fix)

        for i, tag in enumerate(['main', 'opponent']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['kill'] = max_nums[i] - nums[1 - i]
            info[tag]['ave_agent_reward'] = agent_r_records[i]
        info['main']['oppo-ave-agent-reward'] = info['opponent']['ave_agent_reward']
        info['main']['oppo-kill'] = info['opponent']['kill']
        info['main']['oppo-total-reward'] = info['opponent']['total_reward']

        if self.train:
            print('\n[INFO] {}'.format(info['main']))
            if info['main']['total_reward'] > info['opponent']['total_reward'] and self.is_selfplay:
                print(Color.INFO.format('\n[INFO] Begin self-play Update ...'))
                self.sess.run(self.sp_op)
                if self.MsgModels[0]:
                    self.sess.run(self.m_sp_op)
                print(Color.INFO.format('[INFO] Self-play Updated!\n'))

                print(Color.INFO.format('[INFO] Saving model ...'))
                self.update_cnt += 1

                if self.update_cnt % 7 == 0 and self.update_cnt > 300 and iteration > 1200:
                    self.models[0].save(
                        self.model_dir + '-0', str(self.len_nei)+'-'+str(
                            iteration)+self.rewardtype+('crp' if self.crp else 'nom')+('w' if self.MsgModels[0] else 'nw'))
                    self.models[1].save(
                        self.model_dir + '-1', str(self.len_nei)+'-'+str(
                            iteration)+self.rewardtype+('crp' if self.crp else 'nom')+('w' if self.MsgModels[0] else 'nw'))
                    if self.MsgModels[0]:
                        self.MsgModels[0].save(self.model_dir + '-msg0', str(self.len_nei)+'-'+str(
                            iteration)+self.rewardtype+('crp' if self.crp else 'nom')+'w')
                        self.MsgModels[1].save(self.model_dir + '-msg1', str(self.len_nei)+'-'+str(
                            iteration)+self.rewardtype+('crp' if self.crp else 'nom')+'w')
                self.summary.write(info['main'], iteration)
            elif not self.is_selfplay and not self.is_fix:
                print(Color.INFO.format('\n[INFO] Begin self-play Update ...'))
                print(Color.INFO.format('[INFO] Self-play Updated!\n'))

                print(Color.INFO.format('[INFO] Saving model ...'))
                self.update_cnt += 1
                self.summary.write(info['main'], iteration)
            elif not self.is_selfplay and self.is_fix:
                print(Color.INFO.format('\n[INFO] Begin self-play Update ...'))
                print(Color.INFO.format('[INFO] Self-play Updated!\n'))

                print(Color.INFO.format('[INFO] Saving model ...'))
                self.update_cnt += 1
                if self.update_cnt % 7 == 0 and self.update_cnt > 500:
                    self.models[0].save(
                        self.model_dir + '-0', str(self.len_nei)+'-'+str(iteration % 5) +
                        self.rewardtype+('crp' if self.crp else 'nom')+('w' if self.MsgModels[0] else 'nw'))
                    if self.MsgModels[0]:
                        self.MsgModels[0].save(self.model_dir + '-msg0', str(self.len_nei)+'-'+str(
                            iteration % 5)+self.rewardtype+('crp' if self.crp else 'nom')+'w')
                self.summary.write(info['main'], iteration)

        else:
            print('\n[INFO] {0} \n {1}'.format(info['main'], info['opponent']))
            if info['main']['kill'] > info['opponent']['kill']:
                win_cnt['main'] += 1
            elif info['main']['kill'] < info['opponent']['kill']:
                win_cnt['opponent'] += 1
            else:
                win_cnt['main'] += 1
                win_cnt['opponent'] += 1

        return info['main']['ave_agent_reward']
