
import torch
import torch.nn as nn
import torch.optim as optim

from humanoid.algo.ppo.rollout_storage import RolloutStorage

class GRPO:
    def __init__(self,
                 actor,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 group_size=4,
                 kl_beta=0.01, 
                 **kwargs
                 ):

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        

        self.group_size = group_size
        self.kl_beta = kl_beta
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.train(False)
    
    def train_mode(self):
        self.actor_critic.train(True)

    def act(self, obs, critic_obs):

        self.transition.actions = self.actor_critic.act(obs).detach()
        
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):

        returns = torch.zeros_like(self.storage.rewards)
        
        
        R = torch.zeros_like(self.storage.rewards[0])
        
        for step in reversed(range(self.storage.num_transitions_per_env)):


            non_terminal = 1.0 - self.storage.dones[step].float()
            R = self.storage.rewards[step] + self.gamma * R * non_terminal
            returns[step] = R
            
        self.storage.returns.copy_(returns)
        
        
        total_steps, total_envs, _ = returns.shape
        

        if self.group_size <= 0 or self.group_size >= total_envs:
             flat_returns = returns.view(-1, 1)
             mean = flat_returns.mean()
             std = flat_returns.std()
             self.storage.advantages = (returns - mean) / (std + 1e-6)
        else:
            num_groups = total_envs // self.group_size
            
            returns_grouped = returns.view(total_steps, num_groups, self.group_size, 1)
            
            group_mean = returns_grouped.mean(dim=2, keepdim=True)
            group_std = returns_grouped.std(dim=2, keepdim=True)
            
            advantages_grouped = (returns_grouped - group_mean) / (group_std + 1e-6)
            
            self.storage.advantages.copy_(advantages_grouped.view(total_steps, total_envs, 1))

    def update(self):
        mean_kl_loss = 0
        mean_surrogate_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                
                with torch.inference_mode():
                     pass

                kl_div = torch.log(sigma_batch / old_sigma_batch + 1.e-5) + \
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / \
                        (2.0 * torch.square(sigma_batch)) - 0.5
                kl_div = torch.sum(kl_div, axis=-1)
                
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                
                surrogate = torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                
                
                kl_penalty = self.kl_beta * kl_div
                
                
                objective = torch.min(surrogate, surrogate_clipped) - kl_penalty
                
                loss = -objective.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_surrogate_loss += loss.item()
                mean_kl_loss += kl_div.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_surrogate_loss /= num_updates
        mean_kl_loss /= num_updates
        self.storage.clear()

        return mean_surrogate_loss, mean_kl_loss
