import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import RacingEnv
from track import TRACKS

NUM_ENVS = 8
MODEL_DIR = "models"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="generalist")
    parser.add_argument("--track", default=None, choices=list(TRACKS))
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def build_model(args):
    vec_env = make_vec_env(
        RacingEnv,
        n_envs=NUM_ENVS,
        seed=args.seed,
        env_kwargs={"track": args.track},
        vec_env_cls=SubprocVecEnv,
    )
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=1024,
        batch_size=256,
        ent_coef=0.005,
        seed=args.seed,
        verbose=1,
    )


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = build_model(args)
    model.learn(total_timesteps=args.steps)

    save_path = os.path.join(MODEL_DIR, args.name)
    model.save(save_path)
    print(f"saved {save_path}.zip")


if __name__ == "__main__":
    main()
