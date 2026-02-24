import torch

class DiscreteActionsWrapper:
    # creates a wrapper around the env to allow us to go from the discrete action space back to the continuous
    def __init__(self, env, num_bins = 7, low = -1.0, high = 1.0):
        self.env = env

        # create the bins 
        self.num_bins = int(num_bins)
        self.bins = torch.linspace(low, high, self.num_bins, device=env.device)

        self.device = env.device
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.cfg = env.cfg
        self.dt = env.dt
        self.gym = getattr(env, "gym", None)
        self.sim = getattr(env, "sim", None)
        self.envs = getattr(env, "envs", None)

    def __getattr__(self, name):
        return getattr(self.env, name)

    # goes from action index to the action in the continuous state space
    def step(self, action_index):
        if action_index.dtype != torch.long:
            action_index = action_index.long()
        action_index = torch.clamp(action_index, 0, self.num_bins - 1)
        action_continuous_state_space = self.bins[action_index]
        return self.env.step(action_continuous_state_space)

    def reset(self):
        return self.env.reset()