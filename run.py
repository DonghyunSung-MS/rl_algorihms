import argparse
import wandb
import agents.ppo.rl_agent as rl_agent
import dm_control.suite as suite
from rl_configs import RL_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="acrobot")
    parser.add_argument("--seed",type=int,default=1)
    args = parser.parse_args()
    wandb.init(project="custom-rl-algorithms-test")

    configs = RL_CONFIGS[args.env] #presetting prameters for each enviroment.
    wandb.config.update(configs)
    env = suite.load("acrobot","swingup")
    env._task.__init__(False,random= args.seed)


    imit_agent = rl_agent.PPOAgent(env, configs, args.seed)
    imit_agent.train()

if __name__=="__main__":
    main()
