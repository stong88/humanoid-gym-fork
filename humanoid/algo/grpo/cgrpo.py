import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from sklearn.cluster import DBSCAN, KMeans
except ImportError as exc:
    raise ImportError(
        "CGRPO requires scikit-learn for KMeans/DBSCAN clustering. "
        "Install with: pip install scikit-learn"
    ) from exc

from humanoid.algo.ppo.rollout_storage import RolloutStorage


class CGRPO:
    """Continuous GRPO without a critic/value function.

    Key properties:
    - Uses trajectory clustering (KMeans) for policy grouping.
    - Uses state clustering (DBSCAN) for state-relative advantages.
    - Uses group-normalized, group-clipped policy updates.
    - Adds temporal smoothness and inter-group diversity regularization.
    """

    def __init__(
        self,
        actor,
        num_learning_epochs=4,
        num_mini_batches=8,
        clip_param=0.2,
        gamma=0.994,
        lam=0.92,
        value_loss_coef=0.0,
        entropy_coef=0.001,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        use_clipped_value_loss=False,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        group_size=-1,
        kl_beta=0.004,
        temporal_smoothness_coef=0.01,
        diversity_coef=0.003,
        num_clusters=8,
        dbscan_eps=0.45,
        dbscan_min_samples=6,
        clip_var_scale=0.3,
        min_clip_param=0.1,
        max_clip_param=0.3,
        state_feature_dim=16,
        max_state_cluster_samples=1024,
        ref_mix_alpha=0.04,
        topk_reference=0.2,
        **kwargs,
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

        # Critic-free: keep placeholders for runner compatibility only.
        self.value_loss_coef = 0.0
        self.use_clipped_value_loss = False

        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.group_size = group_size
        self.kl_beta = kl_beta
        self.temporal_smoothness_coef = temporal_smoothness_coef
        self.diversity_coef = diversity_coef

        self.num_clusters = max(2, int(num_clusters))
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = max(2, int(dbscan_min_samples))
        self.clip_var_scale = float(clip_var_scale)
        self.min_clip_param = float(min_clip_param)
        self.max_clip_param = float(max_clip_param)

        self.state_feature_dim = int(state_feature_dim)
        self.max_state_cluster_samples = int(max_state_cluster_samples)
        self.ref_mix_alpha = float(ref_mix_alpha)
        self.topk_reference = float(topk_reference)

        self.state_proj = None
        self.sample_group_ids = None
        self.sample_clips = None

        self.reference_actor = copy.deepcopy(self.actor_critic.actor).to(self.device)
        for p in self.reference_actor.parameters():
            p.requires_grad = False

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.train(False)

    def train_mode(self):
        self.actor_critic.train(True)

    def act(self, obs, critic_obs):
        self.transition.actions = self.actor_critic.act(obs).detach()
        # Critic-free placeholder for storage compatibility.
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

    def _ensure_state_projection(self, obs_dim):
        if self.state_proj is None or self.state_proj.shape[0] != obs_dim:
            proj = torch.randn(obs_dim, self.state_feature_dim, device=self.device)
            proj = proj / torch.clamp(torch.norm(proj, dim=0, keepdim=True), min=1e-6)
            self.state_proj = proj

    def _discounted_returns(self):
        returns = torch.zeros_like(self.storage.rewards)
        running = torch.zeros_like(self.storage.rewards[0])
        for step in reversed(range(self.storage.num_transitions_per_env)):
            non_terminal = 1.0 - self.storage.dones[step].float()
            running = self.storage.rewards[step] + self.gamma * running * non_terminal
            returns[step] = running
        return returns

    def _compute_policy_groups(self, returns):
        t, n, _ = returns.shape
        ret_env = returns.mean(dim=0).squeeze(-1)

        sigma = torch.clamp(self.storage.sigma, min=1e-4)
        entropy = (0.5 + 0.5 * torch.log(2.0 * torch.pi * torch.square(sigma) + 1e-8)).sum(dim=-1).mean(dim=0)
        act_var = self.storage.actions.var(dim=0, unbiased=False).mean(dim=-1)

        with torch.inference_mode():
            flat_obs = self.storage.observations.flatten(0, 1)
            ref_mu = self.reference_actor(flat_obs).view(t, n, -1)

        cur_mu = self.storage.mu
        ref_sigma = torch.ones_like(sigma)
        kl_ref = torch.log(ref_sigma / sigma) + (
            torch.square(sigma) + torch.square(cur_mu - ref_mu)
        ) / (2.0 * torch.square(ref_sigma)) - 0.5
        kl_ref = kl_ref.sum(dim=-1).mean(dim=0)

        feat = torch.stack([ret_env, entropy, act_var, kl_ref], dim=-1)
        feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + 1e-6)

        if self.group_size <= 0:
            k = self.num_clusters
        else:
            k = max(2, n // self.group_size)
        k = min(k, n)

        feat_np = feat.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        group_ids = torch.from_numpy(kmeans.fit_predict(feat_np)).to(self.device, dtype=torch.long)

        return group_ids, ret_env

    def _assign_state_clusters(self, state_features):
        n = state_features.shape[0]
        sample_n = min(n, self.max_state_cluster_samples)
        sample_idx = torch.randperm(n, device=self.device)[:sample_n]
        sample = state_features[sample_idx].detach().cpu().numpy()

        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        sample_labels = dbscan.fit_predict(sample)

        valid = sample_labels >= 0
        if not np.any(valid):
            return torch.full((n,), -1, dtype=torch.long, device=self.device)

        unique = np.unique(sample_labels[valid])
        centers = []
        center_ids = []
        for cid in unique:
            mask = sample_labels == cid
            centers.append(sample[mask].mean(axis=0))
            center_ids.append(int(cid))

        centers = torch.from_numpy(np.asarray(centers, dtype=np.float32)).to(self.device)
        d = torch.cdist(state_features, centers, p=2.0)
        nearest = torch.argmin(d, dim=1)
        min_d = torch.min(d, dim=1).values

        mapped = torch.tensor(center_ids, device=self.device, dtype=torch.long)
        labels = torch.where(min_d <= self.dbscan_eps, mapped[nearest], torch.full_like(nearest, -1))
        return labels

    def compute_returns(self, last_critic_obs):
        returns = self._discounted_returns()
        self.storage.returns.copy_(returns)

        t, n, obs_dim = self.storage.observations.shape
        group_ids_env, env_scores = self._compute_policy_groups(returns)

        self._ensure_state_projection(obs_dim)
        flat_obs = self.storage.observations.flatten(0, 1)
        state_features = torch.tanh(flat_obs @ self.state_proj)
        state_cluster_ids = self._assign_state_clusters(state_features)

        flat_returns = returns.flatten(0, 1).squeeze(-1)
        adv = torch.zeros_like(flat_returns)

        valid = state_cluster_ids >= 0
        if valid.any():
            max_cid = int(state_cluster_ids[valid].max().item())
            sums = torch.zeros(max_cid + 1, device=self.device)
            counts = torch.zeros(max_cid + 1, device=self.device)
            sums.scatter_add_(0, state_cluster_ids[valid], flat_returns[valid])
            counts.scatter_add_(0, state_cluster_ids[valid], torch.ones_like(flat_returns[valid]))
            means = sums / torch.clamp(counts, min=1.0)
            adv[valid] = flat_returns[valid] - means[state_cluster_ids[valid]]
            adv[~valid] = flat_returns[~valid] - flat_returns.mean()
        else:
            adv = flat_returns - flat_returns.mean()

        env_ids = torch.arange(n, device=self.device).repeat(t)
        sample_group_ids = group_ids_env[env_ids]

        num_groups = int(sample_group_ids.max().item()) + 1
        for gid in range(num_groups):
            mask = sample_group_ids == gid
            if mask.any():
                a = adv[mask]
                adv[mask] = (a - a.mean()) / (a.std(unbiased=False) + 1e-6)

        self.storage.advantages.copy_(adv.view(t, n, 1))
        self.sample_group_ids = sample_group_ids

        group_clips = torch.full((num_groups,), self.clip_param, device=self.device)
        for gid in range(num_groups):
            env_mask = group_ids_env == gid
            if env_mask.any():
                gadv = self.storage.advantages[:, env_mask, :].reshape(-1)
                gstd = torch.sqrt(torch.clamp(torch.var(gadv, unbiased=False), min=0.0))
                dyn = self.clip_param * (1.0 + self.clip_var_scale * gstd)
                group_clips[gid] = torch.clamp(dyn, self.min_clip_param, self.max_clip_param)
        self.sample_clips = group_clips[sample_group_ids]

        k_top = max(1, int(n * self.topk_reference))
        top_idx = torch.topk(env_scores, k=k_top).indices
        score = torch.clamp(env_scores[top_idx].mean(), min=0.0)
        alpha = torch.clamp(self.ref_mix_alpha * (1.0 + 0.1 * score), max=0.2)
        with torch.inference_mode():
            for ref_p, cur_p in zip(self.reference_actor.parameters(), self.actor_critic.actor.parameters()):
                ref_p.mul_(1.0 - alpha).add_(alpha * cur_p)

    def _iter_minibatches(self):
        t, n, _ = self.storage.observations.shape
        batch_size = t * n
        mini_batch_size = batch_size // self.num_mini_batches

        obs = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        old_logp = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)
        adv = self.storage.advantages.flatten(0, 1).squeeze(-1)

        next_obs = torch.roll(self.storage.observations, shifts=-1, dims=0)
        next_obs[-1] = self.storage.observations[-1]
        next_obs = next_obs.flatten(0, 1)

        indices = torch.randperm(self.num_mini_batches * mini_batch_size, device=self.device)
        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                s = i * mini_batch_size
                e = (i + 1) * mini_batch_size
                bidx = indices[s:e]
                yield (
                    obs[bidx],
                    next_obs[bidx],
                    actions[bidx],
                    old_logp[bidx],
                    old_mu[bidx],
                    old_sigma[bidx],
                    adv[bidx],
                    self.sample_clips[bidx],
                    self.sample_group_ids[bidx],
                )

    def update(self):
        mean_surrogate_loss = 0.0
        mean_kl_loss = 0.0
        num_updates = 0

        for (
            obs_b,
            next_obs_b,
            actions_b,
            old_logp_b,
            old_mu_b,
            old_sigma_b,
            adv_b,
            clip_b,
            group_id_b,
        ) in self._iter_minibatches():
            self.actor_critic.act(obs_b)
            logp = self.actor_critic.get_actions_log_prob(actions_b)
            mu = self.actor_critic.action_mean
            sigma = torch.clamp(self.actor_critic.action_std, min=1e-4)
            entropy = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_m = torch.sum(
                        torch.log(sigma / torch.clamp(old_sigma_b, min=1e-4) + 1e-5)
                        + (torch.square(old_sigma_b) + torch.square(old_mu_b - mu))
                        / (2.0 * torch.square(sigma))
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl_m > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_m < self.desired_kl / 2.0:
                        self.learning_rate = min(3e-4, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.learning_rate

            ratio = torch.exp(logp - old_logp_b)
            ratio_clip = torch.clamp(ratio, 1.0 - clip_b, 1.0 + clip_b)
            surr = torch.min(ratio * adv_b, ratio_clip * adv_b)

            kl = torch.sum(
                torch.log(sigma / torch.clamp(old_sigma_b, min=1e-4) + 1e-5)
                + (torch.square(old_sigma_b) + torch.square(old_mu_b - mu))
                / (2.0 * torch.square(sigma))
                - 0.5,
                dim=-1,
            )

            mu_next = self.actor_critic.actor(next_obs_b)
            smooth_pen = F.mse_loss(mu, mu_next)

            uniq = torch.unique(group_id_b)
            centroids = []
            for gid in uniq:
                mask = group_id_b == gid
                if mask.any():
                    centroids.append(mu[mask].mean(dim=0))
            if len(centroids) > 1:
                centroids = torch.stack(centroids, dim=0)
                c = F.normalize(centroids, dim=-1)
                sim = c @ c.t()
                div_pen = (sim.sum() - torch.diagonal(sim).sum()) / (sim.numel() - sim.shape[0])
            else:
                div_pen = torch.zeros(1, device=self.device).squeeze(0)

            surrogate_loss = -surr.mean()
            loss = (
                surrogate_loss
                + self.kl_beta * kl.mean()
                - self.entropy_coef * entropy.mean()
                + self.temporal_smoothness_coef * smooth_pen
                + self.diversity_coef * div_pen
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_surrogate_loss += surrogate_loss.item()
            mean_kl_loss += kl.mean().item()
            num_updates += 1

        self.storage.clear()
        if num_updates > 0:
            mean_surrogate_loss /= num_updates
            mean_kl_loss /= num_updates

        # Keep runner-compatible signature: (value_loss, surrogate_loss)
        return 0.0, mean_surrogate_loss
