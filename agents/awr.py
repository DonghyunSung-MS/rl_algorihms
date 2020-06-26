from agents.rl_agent import *

class AWRAgent(Agent):
    def set_own_hyper(self, configs):
        #off-policy learning
        self.buffer_size = configs.buffer_size
        #temperature
        self.beta = configs.beta

        #max weight
        self.max_weight = configs.max_weight

        #how many gradient steps per iteration
        self.model_update_num_critic = configs.model_update_num_critic
        self.model_update_num_actor = configs.model_update_num_actor

    def train(self):
        log_file = os.path.join(self.log_dir,"log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file)
        self.history = ReplayBuffer(self.buffer_size)
        start_time = time.time()
        total_samples = 0
        for iter in range(self.max_iter):
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

                if self.wandb:
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
        states = list(self.history.states)
        actions = list(self.history.actions)
        rewards = list(self.history.rewards)
        masks = list(self.history.masks)



        states = torch.Tensor(states).squeeze(1)
        actions = torch.Tensor(actions).squeeze(1)
        rewards = torch.Tensor(rewards).reshape(-1,1)
        masks = torch.Tensor(masks).unsqueeze(1)

        #print("states_shape",states[0].shape)
        #print("states_tensor_shape",torch.Tensor(states).shape)
        #print("actions_shape",actions.shape)
        #print("rewards_shape",rewards.shape)
        #print("masks_shape",masks.shape)

        old_values = self._critic(torch.Tensor(states).to(self.dev))

        _, advantages = rl_utills.calculate_gae(masks, rewards, old_values, self)

        target_values = advantages.to(self.dev) + old_values

        mse = torch.nn.MSELoss()

        num_sample = len(rewards)

        arr = np.arange(num_sample)
        num = 0
        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(self.model_update_num_critic):
            np.random.shuffle(arr)
            for i in range(num_sample//self.batch_size):
                mini_batch_index = arr[self.batch_size*i : self.batch_size*(i+1)]
                mini_batch_index = torch.LongTensor(mini_batch_index).to(self.dev)

                states_samples = torch.Tensor(states)[mini_batch_index].to(self.dev)
                target_values_samples = target_values[mini_batch_index].to(self.dev)

                old_values_samples = self._critic((states_samples).to(self.dev))

                critic_loss = mse(target_values_samples.detach(), old_values_samples)

                total_critic_loss += critic_loss

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        #use updated critic
        updated_values = self._critic(torch.Tensor(states).to(self.dev))
        _, updated_advantages = rl_utills.calculate_gae(masks, rewards, updated_values, self)

        for _ in range(self.model_update_num_actor):
            np.random.shuffle(arr)
            for i in range(num_sample//self.batch_size):
                mini_batch_index = arr[self.batch_size*i : self.batch_size*(i+1)]
                mini_batch_index = torch.LongTensor(mini_batch_index).to(self.dev)

                states_samples = torch.Tensor(states)[mini_batch_index].to(self.dev)
                actions_samples = torch.Tensor(actions)[mini_batch_index].to(self.dev)
                updated_advantages_samples = updated_advantages[mini_batch_index].to(self.dev)

                weights = torch.clamp(updated_advantages_samples/self.beta, max=self.max_weight)
                #print(weights)

                mu, std = self._actor(states_samples)
                policy_log = self._actor.get_log_prob(actions_samples, mu, std)

                actor_loss = -(weights * policy_log).mean()

                total_actor_loss += actor_loss

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()


        if (iter+1)%self.log_interval==0:
            self._logger.log_tabular("Actor_loss", total_actor_loss.item()/self.model_update_num_actor)
            self._logger.log_tabular("Critic_loss", total_critic_loss.item()/self.model_update_num_critic)
            self._logger.print_tabular()
            self._logger.dump_tabular()
        if self.wandb:
            wandb.log({"Actor_loss": total_actor_loss.item()/self.model_update_num_actor,
		       "Critic_loss": total_critic_loss.item()/self.model_update_num_critic
		      })
