
import os
import copy
import numpy as np
import wandb
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#from collections import deque

from dm_control import viewer
import utills.logger as logger
from utills.trajectoryBuffer import *
import utills.rl_utills as rl_utills
from agents.core import Actor, Critic

from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, env, config):
        self.benchmark = config.Option.benchmark
        self._env = env
        self._logger = None
        """argument to self value"""
        self.render = config.Option.render
        self.img = None

        torch.manual_seed(config.Option.seed)
        np.random.seed(config.Option.seed)
        self.wandb = config.Option.wandb

        #Hyperparameters depend on algorithms
        self.set_own_hyper(config.Learning)

        self.log_dir = config.Log.log_dir
        self.log_interval = config.Log.log_interval

        self.model_dir = config.Model.model_dir
        self.save_interval = config.Model.save_interval

        self.max_iter = config.Learning.max_iter
        self.batch_size = config.Learning.batch_size

        self.total_sample_size = config.Learning.total_sample_size
        self.test_iter = config.Learning.test_iter

        self.gamma = config.Learning.gamma
        self.lamda = config.Learning.lamda
        self.actor_lr = config.Learning.actor_lr
        self.critic_lr = config.Learning.critic_lr


        self.state_dim = None
        self.action_dim = None

        if self.benchmark == "dm_control":
            self.state_dim = 0
            for k, v in self._env.observation_spec().items():
                shape = list(v.shape)
                try:
                    self.state_dim += np.prod(shape)
                except:
                    self.state_dim += 1
            self.action_dim = self._env.action_spec().shape[0]
            print("State spec : ",self._env.observation_spec())
            print("Action spec: ",self._env.action_spec())
            print("state_size : ", self.state_dim )
            print("action_size : ", self.action_dim)
        elif self.benchmark == "gym":
            self.state_dim = self._env.observation_space.shape[0]
            self.action_dim = self._env.action_space.shape[0]
            print("state_size : ", self.state_dim )
            print("action_size : ", self.action_dim)

        self.dev = None
        if config.gpu:
            self.dev = torch.device("cuda:0")
        else:
            self.dev = torch.device("cpu")

        self._actor = Actor(self.state_dim, self.action_dim, config).to(self.dev)
        self._critic = Critic(self.state_dim, config).to(self.dev)

        self.actor_optim = optim.Adam(self._actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self._critic.parameters(), lr=self.critic_lr)

        self.history = None
        self.global_episode = 0

        super().__init__()

    def test_interact(self, model_path, random=False):
        """load trained parameters"""
        if not random:
            self._actor.load_state_dict(torch.load(model_path))

        if self.benchmark == "dm_control":
            if random:
                def random_policy(time_step):
                    del time_step  # Unused.
                    return np.random.uniform(low=self._env.action_spec().minimum,
                                             high=self._env.action_spec().maximum,
                                             size=self._env.action_spec().shape)
                viewer.launch(self._env, policy=random_policy)
            else:
                def source_policy(time_step):
                    s = None
                    for k, v in time_step.observation.items():
                        if s is None:
                            s = v.flatten()
                        else:
                            s = np.hstack([s, v])
                    s_3d = np.reshape(s, [1, self.state_dim])
                    mu, std = self._actor(torch.Tensor(s_3d).to(self.dev))
                    action = self._actor.get_action(mu, std)

                    return action

                viewer.launch(self._env, policy=source_policy)
        elif self.benchmark == "gym":
            for ep in range(self.test_iter):
                score = 0
                done = False
                state = self._env.reset()
                state = np.reshape(state, [1, self.state_dim])
                while not done:
                    mu, std = self._actor(torch.Tensor(state).to(self.dev))
                    action = self._actor.get_action(mu, std)

                    if random:
                        next_state, reward, done, info = self._env.step(np.random.randn(self.action_dim))
                    else:
                        next_state, reward, done, info = self._env.step(action)
                    self._env.render()

                    score = self.gamma*score + reward
                    next_state = np.reshape(next_state, [1, self.state_dim])
                    state = next_state
                print(f"test iter : {ep}\tscore : {score}")

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def _update(self, iter):
        pass

    @abstractmethod
    def set_own_hyper(self, config):
        pass

    def _rollout(self):
        """
        rollout until sample num is larger than max samples per iteration
        and last episode is finished.
        """

        sample_num = 0
        return_buffer = []
        ep_len_buffer = []

        while sample_num < self.total_sample_size:
            #initialize episode
            steps = 0
            discouted_sum_reward = 0

            time_step = None #dm_control
            s = None
            done = False

            if self.benchmark == "dm_control":
                time_step = self._env.reset()
                s, _ , __ = self.history.covert_time_step_data(time_step)
            elif self.benchmark == "gym":
                s = self._env.reset()

            s_3d = np.reshape(s, [1, self.state_dim])
            #print(time_step.last())
            while not done:
                tic = time.time()

                mu, std = self._actor(torch.Tensor(s_3d).to(self.dev))
                action = self._actor.get_action(mu, std)

                s_ = None
                r = 0.0
                m = 0.0
                #Action to enviroment and get next state, reward, done(bool)
                if self.benchmark == "dm_control":
                    time_step = self._env.step(action)
                    s_, r, m = self.history.covert_time_step_data(time_step)
                    done = time_step.last()
                elif self.benchmark == "gym":
                    s_, r, done, info = self._env.step(action)
                    r = r.item(0)
                    m = 0.0 if done else 1.0

                self.history.store_history(action, s_3d, r, m)

                s = s_
                s_3d = np.reshape(s, [1, self.state_dim])



                if self.render:
                    self._render(tic, steps)

                steps += 1
                discouted_sum_reward = discouted_sum_reward*self.gamma + r

            return_buffer.append(discouted_sum_reward)
            ep_len_buffer.append(steps)
            sample_num += steps
        return_buffer = np.array(return_buffer)
        ep_len_buffer = np.array(ep_len_buffer)
        return sample_num, np.mean(return_buffer), np.std(return_buffer), np.mean(ep_len_buffer)

    def _render(self, tic, steps):
        if self.benchmark == "dm_control":
            max_frame = 90

            width = 640
            height = 480
            video = np.zeros((1000, height, 2 * width, 3), dtype=np.uint8)
            video[steps] = np.hstack([self._env.physics.render(height, width, camera_id=0),
                                     self._env.physics.render(height, width, camera_id=1)])

            if steps==0:
                self.img = plt.imshow(video[steps])
            else:
                self.img.set_data(video[steps])
            toc = time.time()
            clock_dt = toc-tic
            plt.pause(max(0.01, 0.03 - clock_dt))  # Need min display time > 0.0.
            plt.draw()
        elif self.benchmark == "gym":
            self._env.render()


    def save_model(self, iter, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

        ckpt_path_a = dir + str(iter)+'th_model_a.pth.tar'
        ckpt_path_c = dir + str(iter)+'th_model_c.pth.tar'
        torch.save(self._actor.state_dict(), ckpt_path_a)
        torch.save(self._critic.state_dict(), ckpt_path_c)
