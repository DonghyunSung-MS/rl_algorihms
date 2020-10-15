import argparse
import wandb
import toml
from dotmap import DotMap

from agents.ppo import *
from agents.awr import *

import dm_control.suite as suite
import gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--policy", type=str, default="0")
    args = parser.parse_args()

    config = DotMap(toml.load(args.config))

    env = None
    if config.Option.benchmark == "dm_control":
        env = suite.load(config.Option.env, config.Option.task)
    elif config.Option.benchmark == "gym":
        env = gym.make(config.Option.env)
        env.seed(config.Option.seed)

    agent = None

    if config.Option.algorithm == "ppo":
        agent = PPOAgent(env, config)
    elif config.Option.algorithm == "awr":
        agent = AWRAgent(env, config)

    if config.Option.wandb:
        wandb.init(project="custom-rl-algorithms-test")
        wandb.config.update(config.toDict())

    #get N.N parameters
    model_path = config.Model.model_dir + args.policy + "th_model_a.pth.tar"
    agent.test_interact(model_path, random=False)

if __name__=="__main__":
    main()
