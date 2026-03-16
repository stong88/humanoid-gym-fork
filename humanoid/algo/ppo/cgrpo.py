# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans, DBSCAN

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

class CGRPO:
    def __init__(self,
                 actor_critics,
                 num_policies=8,
                 num_kmeans_groups=2,
                 dbscan_eps=0.5,
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
                 ):

        self.device = device

        self.num_policies = num_policies
        self.num_kmeans_groups = num_kmeans_groups
        self.dbscan_eps = dbscan_eps
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        assert len(actor_critics) == self.num_policies
        self.actor_critics = actor_critics
        for actor_critic in self.actor_critics:
            actor_critic.to(self.device)
        
        self.storage = None # initialized later
        self.optimizers = [
            optim.Adam(self.actor_critics[i].parameters(), lr=learning_rate)
            for i in range(num_policies)
        ]
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        assert num_envs % self.num_policies == 0, "Number of envs must be divisible by num policies"
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        for actor_critic in self.actor_critics:
            actor_critic.test()
    
    def train_mode(self):
        for actor_critic in self.actor_critics:
            actor_critic.train()

    def act(self, obs, critic_obs):  # all (num_envs, ...)
        split_obs = obs.chunk(self.num_policies, dim=0)
        split_critic_obs = critic_obs.chunk(self.num_policies, dim=0)

        # Compute the actions and values
        self.transition.actions = torch.cat([
            self.actor_critics[i].act(o)
            for i, o in enumerate(split_obs)
        ], dim=0).detach()
        self.transition.values = torch.cat([
            self.actor_critics[i].evaluate(co)
            for i, co in enumerate(split_critic_obs)
        ], dim=0).detach()
        self.transition.actions_log_prob = torch.cat([
            self.actor_critics[i].get_actions_log_prob(a)
            for i, a in enumerate(self.transition.actions.chunk(self.num_policies, dim=0))
        ], dim=0).detach()
        # NOTE -- I believe action_mean and action_sigma are only used in computation for the KL
        # learning rate computation (which we're omitting), so no huge downstream impact of doing this
        self.transition.action_mean = torch.cat([
            self.actor_critics[i].action_mean
            for i in range(self.num_policies)
        ]).mean().detach()
        self.transition.action_sigma = torch.cat([
            self.actor_critics[i].action_std
            for i in range(self.num_policies)
        ]).mean().detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):  # rewards, dones (num_envs,)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        for actor_critic in self.actor_critics:
            actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):  # (num_envs, ...)
        # Compute returns
        # ----------------
        returns = torch.zeros_like(self.storage.rewards)
        intermediate_returns = torch.zeros_like(self.storage.rewards[0])
        
        for step in reversed(range(self.storage.num_transitions_per_env)):
            non_terminal = 1.0 - self.storage.dones[step].float()
            intermediate_returns = self.storage.rewards[step] + self.gamma * intermediate_returns * non_terminal
            returns[step] = intermediate_returns

        self.storage.returns = returns

        # Conduct grouping
        # -----------------
        policy_labels = self._compute_cgrpo_kmeans()  # (num_policies,)
        
        num_policies = len(policy_labels)
        num_envs = self.storage.num_envs
        assert num_envs % num_policies == 0
        num_envs_per_policy = num_envs // num_policies
        
        group_labels_all_envs = policy_labels.repeat_interleave(num_envs_per_policy)  # (num_envs,)
        assert group_labels_all_envs.shape[0] == num_envs  # Requires num_envs % num_policies == 0

        
        advantages = []
        for i in range(self.num_kmeans_groups):
            group_indices = (group_labels_all_envs == i).nonzero().squeeze()  # (num_envs_in_group,) -- dims will change per k-means group

            group_returns = returns[:, group_indices]  # (num_timesteps_per_env, num_envs_in_group)
            normalized_returns = (group_returns - group_returns.mean()) / (group_returns.std() + 1e-8)
            advantages.append(normalized_returns)
            
            # TODO -- is this needed? Not if this is already done above in computing returns right?
            # normalized_summed_returns = torch.zeros(normalized_returns.shape, device=self.device)
            # for i in reversed(range(normalized_returns.shape[0] - 1)):
            #     normalized_summed_returns[i] = normalized_returns[i] + normalized_returns[i + 1]
        self.advantages = torch.cat(advantages, dim=1)  # (num_timesteps, num_envs)

        assert self.advantages.shape[1] == num_envs


        # last_values= torch.cat([
        #     self.actor_critics[i].evaluate(o)
        #     for i, o in enumerate(last_critic_obs.chunk(self.num_policies, dim=0))
        # ], dim=0).detach()
        # self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def _compute_cgrpo_kmeans(self):
        features = torch.zeros(self.num_policies, 3)  # recall we exclude KL divergence from \phi's
        
        returns = self.storage.returns.mean(dim=0)  # (num_envs, 1)
        returns = returns.chunk(self.num_policies, dim=0)  # tuple of tensors (num_envs/num_policies, 1)
        returns = torch.tensor([r.mean() for r in returns])  # (num_policies,)
        features[:, 0] = returns

        entropies = torch.tensor([  # (num_policies,)
            self.actor_critics[i].entropy.mean()
            for i in range(self.num_policies)
        ])
        features[:, 1] = entropies

        variances = torch.tensor([ # (num_policies,)
            self.actor_critics[i].distribution.variance.mean()
            for i in range(self.num_policies)
        ])
        features[:, 2] = variances
        
        kmeans = KMeans(n_clusters=self.num_kmeans_groups, random_state=0).fit(features.numpy())
        return torch.from_numpy(kmeans.labels_)  # (num_policies,)
    
    # def _compute_cgrpo_state_clusters(self):
    #     features = torch.cat((  # (num_timesteps_per_env, num_envs, observation_dim + action_dim + reward_dim)
    #         self.storage.observations,
    #         self.storage.actions,
    #         self.storage.rewards
    #     ), dim=-1).cpu().detach().reshape(self.storage.num_transitions_per_env * self.storage.num_envs, -1)

    #     clustering = DBSCAN(eps=self.dbscan_eps).fit(features.numpy())
    #     clustering.labels_


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        generator = self.storage.cgrpo_mini_batch_generator(self.num_policies, self.num_kmeans_groups, self.num_mini_batches, self.num_learning_epochs)
        # Per group...
        for policy_indices_batch, obs_batch, _, actions_batch, _, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                actions_log_prob_batch = torch.zeros_like(old_actions_log_prob_batch)

                for i in range(self.num_policies):
                    policy_indices = policy_indices_batch == i
                    self.actor_critics[i].act(obs_batch[policy_indices], masks=None, hidden_states=None)
                    actions_log_prob_batch[policy_indices] = self.actor_critics[i].get_actions_log_prob(actions_batch[policy_indices])

                # for i, actor_critic_index in enumerate(policy_indices_batch):
                #     # Note that `masks` and `hidden_states` are hardcoded in rollout_storage to be None, so this is functionally equivalent
                #     self.actor_critics[actor_critic_index].act(obs_batch[i], masks=None, hidden_states=None)


                mu_batch = torch.stack([self.actor_critics[i].action_mean for i in policy_indices_batch], dim=0).mean(dim=0)
                sigma_batch = torch.stack([self.actor_critics[i].action_std for i in policy_indices_batch], dim=0).mean(dim=0)
                entropy_batch = torch.stack([self.actor_critics[i].entropy.mean() for i in policy_indices_batch], dim=0).mean(dim=0)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for optimizer in self.optimizers:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = self.learning_rate

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                for i in range(self.num_policies):
                    self.optimizers[i].zero_grad()
                loss.backward()
                for i in range(self.num_policies):
                    nn.utils.clip_grad_norm_(self.actor_critics[i].parameters(), self.max_grad_norm)
                    self.optimizers[i].step()

                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return 0, mean_surrogate_loss  # keep 0 as placeholder for mean_value_loss for compatability with original PPO return type