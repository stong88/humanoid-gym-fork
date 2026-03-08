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

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

class CGRPO:
    def __init__(self,
                 actor_critics,
                 num_policies=6,  # CGRPO-specific
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

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.num_policies = num_policies

        # PPO components
        assert len(actor_critics) == self.num_policies
        self.actor_critics = actor_critics
        for actor_critic in self.actor_critics:
            actor_critic.to(self.device)
        self.storages = [] # initialized later
        self.optimizers = [optim.Adam(self.actor_critics[i].parameters(), lr=learning_rate) for i in range(self.num_policies)]
        self.transitions = [RolloutStorage.Transition() for _ in range(self.num_policies)]

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
        for _ in range(self.num_policies):
            self.storages.append(
                RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
            )

    def test_mode(self):
        for i in range(self.num_policies):
            self.actor_critics[i].test()
    
    def train_mode(self):
        for i in range(self.num_policies):
            self.actor_critics[i].train()

    def act(self, env_index, obs, critic_obs):
        # Compute the actions and values
        self.transitions[env_index].actions = self.actor_critics[env_index].act(obs).detach()
        self.transitions[env_index].values = self.actor_critics[env_index].evaluate(critic_obs).detach()
        self.transitions[env_index].actions_log_prob = self.actor_critics[env_index].get_actions_log_prob(self.transitions[env_index].actions).detach()
        self.transitions[env_index].action_mean = self.actor_critics[env_index].action_mean.detach()
        self.transitions[env_index].action_sigma = self.actor_critics[env_index].action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transitions[env_index].observations = obs
        self.transitions[env_index].critic_observations = critic_obs
        return self.transitions[env_index].actions
    
    def process_env_step(self, env_index, rewards, dones, infos):
        self.transitions[env_index].rewards = rewards.clone()
        self.transitions[env_index].dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transitions[env_index].rewards += self.gamma * torch.squeeze(self.transitions[env_index].values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storages[env_index].add_transitions(self.transitions[env_index])
        self.transitions[env_index].clear()
        self.actor_critics[env_index].reset(dones)
    
    def compute_returns(self, env_index, last_critic_obs):
        last_values = self.actor_critics[env_index].evaluate(last_critic_obs).detach()
        self.storages[env_index].compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        # As a sanity check, we arbitrarily choose one specific policy to update and ensure that everything's
        # working correctly. Later, we'll change this to actually implement cgrpo
        ARBITRARY_ENV_INDEX = 0

        generator = self.storages[ARBITRARY_ENV_INDEX].mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critics[ARBITRARY_ENV_INDEX].act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critics[ARBITRARY_ENV_INDEX].get_actions_log_prob(actions_batch)
                value_batch = self.actor_critics[ARBITRARY_ENV_INDEX].evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critics[ARBITRARY_ENV_INDEX].action_mean
                sigma_batch = self.actor_critics[ARBITRARY_ENV_INDEX].action_std
                entropy_batch = self.actor_critics[ARBITRARY_ENV_INDEX].entropy

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
                        
                        for param_group in self.optimizers[ARBITRARY_ENV_INDEX].param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizers[ARBITRARY_ENV_INDEX].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critics[ARBITRARY_ENV_INDEX].parameters(), self.max_grad_norm)
                self.optimizers[ARBITRARY_ENV_INDEX].step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        for i in range(self.num_policies):
            self.storages[i].clear()

        return mean_value_loss, mean_surrogate_loss
