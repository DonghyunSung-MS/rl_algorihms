
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
import utills.trajectoryBuffer as trajBuff
import utills.rl_utills as rl_utills
from agents.ppo.core import PPOActor, PPOCritic


class PPOAgent:
    def __init__(self, env, args, seed):
        self._env = env
        self._logger = None
        """argument to self value"""
        self.render = args.render
        self.img = None
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.log_dir = args.log_dir
        self.log_interval = args.log_interval

        self.model_dir = args.model_dir
        self.save_interval = args.save_interval

        self.max_iter = args.max_iter
        self.batch_size = args.batch_size
        self.model_update_num = args.model_update_num
        self.total_sample_size = args.total_sample_size

        self.gamma = args.gamma
        self.lamda = args.lamda
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.clip_param = args.clip_param

        self.state_dim = sum([v.shape[0] for k, v in self._env.observation_spec().items()])
        self.action_dim = self._env.action_spec().shape[0]
        print("State spec : ",self._env.observation_spec())
        print("Action spec: ",self._env.action_spec())

        self.dev = None
        if args.gpu:
            self.dev = torch.device("cuda:0")
        else:
            self.dev = torch.device("cpu")

        self._actor = PPOActor(self.state_dim, self.action_dim, args).to(self.dev)
        self._critic = PPOCritic(self.state_dim, args).to(self.dev)

        self.actor_optim = optim.Adam(self._actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self._critic.parameters(), lr=self.critic_lr)

        self.history = None
        self.global_episode = 0
    def test_interact(self, model_path):
        self._actor.load_state_dict(torch.load(model_path))

        def source_policy(time_step):
            s = None
            for k, v in time_step.observation.items():
                if s is None:
                    s = v
                else:
                    s = np.hstack([s, v])
            s_3d = np.reshape(s, [1, self.state_dim])
            mu, std = self._actor(torch.Tensor(s_3d).to(self.dev))
            action = self._actor.get_action(mu, std)

            return action

        viewer.launch(self._env, policy=source_policy)

    def train(self):
        log_file = os.path.join(self.log_dir,"log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file)

        start_time = time.time()
        total_samples = 0
        for iter in range(self.max_iter):
            self.history = trajBuff.Trajectory()
            sample_num, avg_train_reward, avg_train_return, avg_steps = self._rollout()
            #print(len(self.history.states))
            total_samples += sample_num
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours
            if (iter+1)%self.log_interval==0:
                self._logger.log_tabular("Iteration", iter+1)
                self._logger.log_tabular("Wall_Time", wall_time)
                self._logger.log_tabular("Samples", total_samples)
                self._logger.log_tabular("Avg_reward_iter", avg_train_reward)
                self._logger.log_tabular("Avg_Return_iter", avg_train_return)
                self._logger.log_tabular("Avg_ep_len_iter", avg_steps)

                wandb.log({"Iteration": iter+1,
                           "Wall_Time": wall_time,
                           "Samples": total_samples,
                           "Avg_reward_iter": avg_train_reward,
                           "Avg_Return_iter": avg_train_return,
                           "Avg_ep_len_iter": avg_steps})

            if (iter+1)%self.save_interval==0:
                self.save_model(iter, self.model_dir)
            self._update(iter)


    def _update(self, iter):
        """update network parameters"""
        states = self.history.states
        actions = self.history.actions
        rewards = self.history.rewards
        masks = self.history.masks

        states = torch.Tensor(states).squeeze(1)
        actions = torch.Tensor(actions).squeeze(1)
        rewards = torch.Tensor(rewards).unsqueeze(1)
        masks = torch.Tensor(masks).unsqueeze(1)

        #print("states_shape",states[0].shape)
        #print("states_tensor_shape",torch.Tensor(states).shape)
        #print("actions_shape",actions.shape)
        #print("rewards_shape",rewards.shape)
        #print("masks_shape",masks.shape)

        old_values = self._critic(torch.Tensor(states).to(self.dev))

        rewards2go, advantages = rl_utills.calculate_gae(masks, rewards, old_values, self)

        mu, std = self._actor(torch.Tensor(states).to(self.dev))
        old_policy_log = self._actor.get_log_prob(actions.to(self.dev), mu, std)

        mse = torch.nn.MSELoss()

        num_sample = len(rewards)

        arr = np.arange(num_sample)
        num = 0
        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(self.model_update_num):
            np.random.shuffle(arr)
            for i in range(num_sample//self.batch_size):
                mini_batch_index = arr[self.batch_size*i : self.batch_size*(i+1)]
                mini_batch_index = torch.LongTensor(mini_batch_index).to(self.dev)

                states_samples = torch.Tensor(states)[mini_batch_index].to(self.dev)
                actions_samples = torch.Tensor(actions)[mini_batch_index].to(self.dev)
                advantages_samples = advantages[mini_batch_index].to(self.dev)
                rewards2go_samples = rewards2go[mini_batch_index].to(self.dev)

                old_values_samples = old_values[mini_batch_index].detach()

                new_values_samples = self._critic(states_samples)
                #print("new",new_values_samples.shape)
                #print("old",old_values_samples.shape)
                #print("rewards2go_samples",rewards2go_samples.shape)
                #Monte
                critic_loss = mse(new_values_samples, rewards2go_samples)
                #Surrogate Loss

                actor_loss, ratio = rl_utills.surrogate_loss(self._actor, old_policy_log.detach(),
                                   advantages_samples, states_samples,  actions_samples,
                                   mini_batch_index)
                #print(actor_loss.shape)
                ratio_clipped = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
                actor_loss = -torch.min(actor_loss,ratio_clipped*advantages_samples).mean()
                num += 1
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

                loss = actor_loss + 0.5 * critic_loss
                # update actor & critic
                self.critic_optim.zero_grad()
                loss.backward(retain_graph=True)
                self.critic_optim.step()

                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

        if (iter+1)%self.log_interval==0:
            self._logger.log_tabular("Actor_loss", total_actor_loss/num)
            self._logger.log_tabular("Critic_loss", total_critic_loss/num)
            self._logger.print_tabular()
            self._logger.dump_tabular()


    def _rollout(self):
        """rollout utill sample num is larger thatn max samples per iter"""

        sample_num = 0
        episode = 0
        avg_train_return = 0
        avg_steps = 0
        while sample_num < self.total_sample_size:
            steps = 0
            total_reward_per_ep = 0
            time_step = self._env.reset()
            s, _ , __ = self.history.covert_time_step_data(time_step)
            s_3d = np.reshape(s, [1, self.state_dim])
            while not time_step.last():
                tic = time.time()
                mu, std = self._actor(torch.Tensor(s_3d).to(self.dev))
                action = self._actor.get_action(mu, std)
                time_step = self._env.step(action)
                s_, r , m = self.history.covert_time_step_data(time_step)
                self.history.store_history(action, s_3d, r, m)
                s = s_
                s_3d = np.reshape(s, [1, self.state_dim])
                total_reward_per_ep += r

                if self.render:
                    self._render(tic, steps)

                steps += 1

            episode += 1
            self.global_episode += 1
            wandb.log({"episode":self.global_episode,
                       "Ep_total_reward": total_reward_per_ep,
                       "Ep_Avg_reward": total_reward_per_ep / steps,
                       "Ep_len": steps})

            sample_num = self.history.get_trajLength()

        avg_steps = sample_num / episode
        sum_reward_iter = self.history.calc_return()
        avg_train_return = sum_reward_iter / episode
        avg_train_reward = sum_reward_iter / steps
        return sample_num, avg_train_reward, avg_train_return, avg_steps

    def _render(self, tic, steps):
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

    def save_model(self, iter, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

        ckpt_path_a = dir + str(iter)+'th_model_a.pth.tar'
        ckpt_path_c = dir + str(iter)+'th_model_c.pth.tar'
        torch.save(self._actor.state_dict(), ckpt_path_a)
        torch.save(self._critic.state_dict(), ckpt_path_c)
