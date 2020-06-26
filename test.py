import gym
import argparse
import dm_control.suite as suite

from rl_configs import *
from agents.ppo import *
from agents.awr import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="gym")
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--task", type=str, default="swingup")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--wandb', action='store_true', help='wand on' )
    parser.add_argument('--algo', type=str, default="awr")
    args = parser.parse_args()

    #select benchmark
    env = None
    if args.benchmark=="dm_control":
        env = suite.load(args.env, args.task)
        env._task.__init__(desired_speed=0.1, random = args.seed)
    elif args.benchmark=="gym":
        env = gym.make(args.env)
        env.seed(args.seed)

    configs = None
    agent = None
    #select algorithms
    if args.algo == "ppo":
        configs = PPO[args.env]
        agent = PPOAgent(env, configs, args)
    elif args.algo == "awr":
        configs = AWR[args.env]
        agent = AWRAgent(env, configs, args)

    if args.wandb:
        wandb.init(project="custom-rl-algorithms-test")
        wandb.config.update(configs)

    #get N.N parameters
    model_path = configs.model_dir+'989th_model_a.pth.tar'

    agent.test_interact(model_path, random=False)

if __name__=="__main__":
    main()
