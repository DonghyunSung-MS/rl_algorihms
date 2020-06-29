from agents.rl_agent import *

class PPOAgent(Agent):
    def set_own_hyper(self, configs):
        #how much clip policy ratio
        self.clip_param = configs.clip_param

        #how many gradient steps per iteration
        self.model_update_num = configs.model_update_num

    def train(self):
        log_file = os.path.join(self.log_dir,"log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file)

        start_time = time.time()
        total_samples = 0
        for iter in range(self.max_iter):
            self.history = Trajectory()

            sample_num, mean_train_return, std_train_return, mean_ep_len = self._rollout()

            total_samples += sample_num
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours

            actor_loss_mean, critic_loss_mean = self._update()

            #loggging
            if (iter+1)%self.log_interval==0:
                self._logger.log_tabular("Iteration", iter+1)
                self._logger.log_tabular("Wall_Time", wall_time)
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
        states = self.history.states
        actions = self.history.actions
        rewards = self.history.rewards
        masks = self.history.masks

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
                self.actor_optim.zero_grad()
                loss.backward()
                self.critic_optim.step()
                self.actor_optim.step()

        return total_actor_loss/num, total_critic_loss/num
