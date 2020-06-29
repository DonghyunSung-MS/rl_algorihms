from agents.rl_agent import *



class AWRAgent(Agent):
    def set_own_hyper(self, configs):
        self.ADV_EPS = 1e-5
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

            sample_num, mean_train_return, std_train_return, mean_ep_len = self._rollout()


            #print(len(self.history.states))
            total_samples += sample_num
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours


            actor_loss_mean, critic_loss_mean = self._update()


            #loggging
            if (iter+1)%self.log_interval==0:
                self._logger.log_tabular("Iteration", iter+1)
                self._logger.log_tabular("Wall Time", wall_time)
                self._logger.log_tabular("Samples", total_samples)
                self._logger.log_tabular("Return std", std_train_return)
                self._logger.log_tabular("Return mean", mean_train_return)
                self._logger.log_tabular("Episode length", mean_ep_len)
                self._logger.log_tabular("Actor loss", actor_loss_mean)
                self._logger.log_tabular("Critic loss", critic_loss_mean)
                self._logger.print_tabular()
                self._logger.dump_tabular()
                if self.wandb:
                    wandb.log({"Iteration": iter+1,
                               "Wall_Time": wall_time,
                               "Samples": total_samples,
                               "Return std": std_train_return,
                               "Return mean": mean_train_return,
                               "Episode length": mean_ep_len,
                               "Actor loss": actor_loss_mean,
                               "Critic loss": critic_loss_mean})
            #check point
            if (iter+1)%self.save_interval==0:
                self.save_model(iter, self.model_dir)

    def _update(self):
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

        retrun2go, advantages = rl_utills.calculate_gae(masks, rewards, old_values, self)

        #TD(lambda) Estimate
        target_values = advantages.to(self.dev) + old_values

        #Monte Calro Estimate
        #target_values = retrun2go

        mse = torch.nn.MSELoss()

        num_sample = len(rewards)


        num = 0
        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(self.model_update_num_critic):
            mini_batch_index = np.random.choice(num_sample, self.batch_size, replace=False)
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
            mini_batch_index = np.random.choice(num_sample, self.batch_size, replace=False)
            mini_batch_index = torch.LongTensor(mini_batch_index).to(self.dev)

            states_samples = torch.Tensor(states)[mini_batch_index].to(self.dev)
            actions_samples = torch.Tensor(actions)[mini_batch_index].to(self.dev)

            updated_advantages_samples = updated_advantages[mini_batch_index].to(self.dev)

            """
            exponential of adv tend to explode
            Try normalizer of sample advantages
            following author's implementation
            """

            normalized_advantatges_samples = (updated_advantages_samples - updated_advantages_samples.mean())/(updated_advantages_samples.std()+self.ADV_EPS)

            weights = torch.clamp(torch.exp(normalized_advantatges_samples/self.beta), max=self.max_weight)
            #print(min(torch.exp(normalized_advantatges_samples/self.beta)).item())
            mu, std = self._actor(states_samples)
            policy_log = self._actor.get_log_prob(actions_samples, mu, std)

            actor_loss = -(weights * policy_log).mean()

            total_actor_loss += actor_loss

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        return total_actor_loss.item()/self.model_update_num_actor, total_critic_loss.item()/self.model_update_num_critic
