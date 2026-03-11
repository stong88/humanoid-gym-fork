import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from humanoid.algo.ppo.rollout_storage import RolloutStorage


class CGRPO:

    def __init__(
        self,
        actor,
        num_learning_epochs=4,
        num_mini_batches=8,
        clip_param=0.2,
        gamma=0.994,
        lam=0.92,
        value_loss_coef=1.0,
        entropy_coef=0.001,
        learning_rate=3e-5,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        group_size=-1,
        kl_beta=0.008,
        temporal_smoothness_coef=0.005,
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
        state_rel_adv_coef=0.25,
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
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

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
        self.state_rel_adv_coef = float(state_rel_adv_coef)

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
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def _kmeans(self, x, k, iters=8):
        n = x.shape[0]
        if n <= k:
            labels = torch.arange(n, device=x.device)
            centers = x
            if n < k:
                pad = x[torch.randint(0, n, (k - n,), device=x.device)]
                centers = torch.cat([x, pad], dim=0)
            return labels, centers

        init = torch.randperm(n, device=x.device)[:k]
        centers = x[init]
        labels = torch.zeros(n, dtype=torch.long, device=x.device)

        for _ in range(iters):
            d = torch.cdist(x, centers, p=2.0)
            labels = torch.argmin(d, dim=1)
            for i in range(k):
                m = labels == i
                if m.any():
                    centers[i] = x[m].mean(dim=0)
        return labels, centers

    def _ensure_state_projection(self, obs_dim):
        if self.state_proj is None or self.state_proj.shape[0] != obs_dim:
            proj = torch.randn(obs_dim, self.state_feature_dim, device=self.device)
            proj = proj / torch.clamp(torch.norm(proj, dim=0, keepdim=True), min=1e-6)
            self.state_proj = proj

    def _dbscan_sample(self, x):
        n = x.shape[0]
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=x.device), torch.empty(0, x.shape[1], device=x.device)

        dist = torch.cdist(x, x, p=2.0)
        nbr = dist <= self.dbscan_eps
        visited = torch.zeros(n, dtype=torch.bool, device=x.device)
        labels = torch.full((n,), -1, dtype=torch.long, device=x.device)
        cid = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            pts = torch.where(nbr[i])[0]
            if pts.numel() < self.dbscan_min_samples:
                continue
            labels[i] = cid
            queue = pts.tolist()
            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    jpts = torch.where(nbr[j])[0]
                    if jpts.numel() >= self.dbscan_min_samples:
                        queue.extend(jpts.tolist())
                if labels[j] < 0:
                    labels[j] = cid
            cid += 1

        if cid == 0:
            return labels, torch.empty(0, x.shape[1], device=x.device)

        centers = []
        for i in range(cid):
            centers.append(x[labels == i].mean(dim=0))
        return labels, torch.stack(centers, dim=0)

    def _assign_state_clusters(self, feat):
        n = feat.shape[0]
        sn = min(n, self.max_state_cluster_samples)
        sidx = torch.randperm(n, device=self.device)[:sn]
        sample = feat[sidx]
        _, centers = self._dbscan_sample(sample)
        if centers.numel() == 0:
            return torch.full((n,), -1, dtype=torch.long, device=self.device)

        d = torch.cdist(feat, centers, p=2.0)
        nearest = torch.argmin(d, dim=1)
        min_d = torch.min(d, dim=1).values
        return torch.where(min_d <= self.dbscan_eps, nearest, torch.full_like(nearest, -1))

    def _compute_policy_groups(self):
        t, n, _ = self.storage.observations.shape

        ret_env = self.storage.returns.mean(dim=0).squeeze(-1)
        ent = (0.5 + 0.5 * torch.log(2.0 * torch.pi * torch.square(self.storage.sigma) + 1e-8)).sum(dim=-1).mean(dim=0)
        a_var = self.storage.actions.var(dim=0, unbiased=False).mean(dim=-1)

        with torch.inference_mode():
            obs = self.storage.observations.flatten(0, 1)
            ref_mu = self.reference_actor(obs).view(t, n, -1)

        cur_mu = self.storage.mu
        cur_sigma = torch.clamp(self.storage.sigma, min=1e-4)
        ref_sigma = torch.ones_like(cur_sigma)
        kl_ref = torch.log(ref_sigma / cur_sigma) + (
            torch.square(cur_sigma) + torch.square(cur_mu - ref_mu)
        ) / (2.0 * torch.square(ref_sigma)) - 0.5
        kl_ref = kl_ref.sum(dim=-1).mean(dim=0)

        feat = torch.stack([ret_env, ent, a_var, kl_ref], dim=-1)
        feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + 1e-6)

        if self.group_size <= 0:
            k = self.num_clusters
        else:
            k = max(2, n // self.group_size)
        k = min(k, n)

        gids, centers = self._kmeans(feat, k)
        return gids, centers, ret_env

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

        t, n, obs_dim = self.storage.observations.shape
        gids_env, _, env_scores = self._compute_policy_groups()

        self._ensure_state_projection(obs_dim)
        flat_obs = self.storage.observations.flatten(0, 1)
        sfeat = torch.tanh(flat_obs @ self.state_proj)
        scids = self._assign_state_clusters(sfeat)

        gae_adv = self.storage.advantages.flatten(0, 1).squeeze(-1)
        flat_ret = self.storage.returns.flatten(0, 1).squeeze(-1)

        state_rel = torch.zeros_like(flat_ret)
        valid = scids >= 0
        if valid.any():
            max_c = int(scids[valid].max().item())
            sums = torch.zeros(max_c + 1, device=self.device)
            cnts = torch.zeros(max_c + 1, device=self.device)
            sums.scatter_add_(0, scids[valid], flat_ret[valid])
            cnts.scatter_add_(0, scids[valid], torch.ones_like(flat_ret[valid]))
            means = sums / torch.clamp(cnts, min=1.0)
            state_rel[valid] = flat_ret[valid] - means[scids[valid]]
            state_rel[~valid] = flat_ret[~valid] - flat_ret.mean()
        else:
            state_rel = flat_ret - flat_ret.mean()

        adv = (1.0 - self.state_rel_adv_coef) * gae_adv + self.state_rel_adv_coef * state_rel

        env_ids = torch.arange(n, device=self.device).repeat(t)
        sample_gids = gids_env[env_ids]

        ng = int(sample_gids.max().item()) + 1
        for gid in range(ng):
            m = sample_gids == gid
            if m.any():
                a = adv[m]
                adv[m] = (a - a.mean()) / (a.std(unbiased=False) + 1e-6)

        self.storage.advantages.copy_(adv.view(t, n, 1))
        self.sample_group_ids = sample_gids

        gclips = torch.full((ng,), self.clip_param, device=self.device)
        for gid in range(ng):
            em = gids_env == gid
            if em.any():
                gadv = self.storage.advantages[:, em, :].reshape(-1)
                gstd = torch.sqrt(torch.clamp(torch.var(gadv, unbiased=False), min=0.0))
                dyn = self.clip_param * (1.0 + self.clip_var_scale * gstd)
                gclips[gid] = torch.clamp(dyn, self.min_clip_param, self.max_clip_param)
        self.sample_clips = gclips[sample_gids]

        k_top = max(1, int(n * self.topk_reference))
        top_idx = torch.topk(env_scores, k=k_top).indices
        score = torch.clamp(env_scores[top_idx].mean(), min=0.0)
        alpha = torch.clamp(self.ref_mix_alpha * (1.0 + 0.1 * score), max=0.2)
        with torch.inference_mode():
            for rp, cp in zip(self.reference_actor.parameters(), self.actor_critic.actor.parameters()):
                rp.mul_(1.0 - alpha).add_(alpha * cp)

    def _iter_minibatches(self):
        t, n, _ = self.storage.observations.shape
        bs = t * n
        mbs = bs // self.num_mini_batches

        obs = self.storage.observations.flatten(0, 1)
        critic_obs = (
            self.storage.privileged_observations.flatten(0, 1)
            if self.storage.privileged_observations is not None
            else obs
        )
        actions = self.storage.actions.flatten(0, 1)
        old_logp = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)
        adv = self.storage.advantages.flatten(0, 1).squeeze(-1)
        ret = self.storage.returns.flatten(0, 1)
        old_v = self.storage.values.flatten(0, 1)

        next_obs = torch.roll(self.storage.observations, shifts=-1, dims=0)
        next_obs[-1] = self.storage.observations[-1]
        next_obs = next_obs.flatten(0, 1)

        idx = torch.randperm(self.num_mini_batches * mbs, device=self.device)
        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                s = i * mbs
                e = (i + 1) * mbs
                b = idx[s:e]
                yield (
                    obs[b],
                    critic_obs[b],
                    next_obs[b],
                    actions[b],
                    old_logp[b],
                    old_mu[b],
                    old_sigma[b],
                    adv[b],
                    ret[b],
                    old_v[b],
                    self.sample_clips[b],
                    self.sample_group_ids[b],
                )

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        num_updates = 0

        for (
            obs_b,
            critic_obs_b,
            next_obs_b,
            actions_b,
            old_logp_b,
            old_mu_b,
            old_sigma_b,
            adv_b,
            ret_b,
            old_v_b,
            clip_b,
            group_id_b,
        ) in self._iter_minibatches():
            self.actor_critic.act(obs_b)
            logp = self.actor_critic.get_actions_log_prob(actions_b)
            mu = self.actor_critic.action_mean
            sigma = torch.clamp(self.actor_critic.action_std, min=1e-4)
            entropy = self.actor_critic.entropy
            value = self.actor_critic.evaluate(critic_obs_b)

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
            cents = []
            for gid in uniq:
                m = group_id_b == gid
                if m.any():
                    cents.append(mu[m].mean(dim=0))
            if len(cents) > 1:
                cents = torch.stack(cents, dim=0)
                c = F.normalize(cents, dim=-1)
                sim = c @ c.t()
                div_pen = (sim.sum() - torch.diagonal(sim).sum()) / (sim.numel() - sim.shape[0])
            else:
                div_pen = torch.zeros(1, device=self.device).squeeze(0)

            if self.use_clipped_value_loss:
                v_clip = old_v_b + (value - old_v_b).clamp(-clip_b.unsqueeze(-1), clip_b.unsqueeze(-1))
                v_loss = torch.max((value - ret_b).pow(2), (v_clip - ret_b).pow(2)).mean()
            else:
                v_loss = (value - ret_b).pow(2).mean()

            surrogate_loss = -surr.mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * v_loss
                + self.kl_beta * kl.mean()
                - self.entropy_coef * entropy.mean()
                + self.temporal_smoothness_coef * smooth_pen
                + self.diversity_coef * div_pen
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += v_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            num_updates += 1

        self.storage.clear()
        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
        return mean_value_loss, mean_surrogate_loss
