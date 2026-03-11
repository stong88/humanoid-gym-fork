from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def train(args):

    args.task = "humanoid_cgrpo"
    print(f"Training with task: {args.task}")

    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
