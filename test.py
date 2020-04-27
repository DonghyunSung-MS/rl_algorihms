import argparse
import dm_control.suite as suite

from rl_configs import RL_CONFIGS
import agents.ppo.rl_agent as rl_agent



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="pendulum")
    parser.add_argument("--seed",type=int,default=1)
    args = parser.parse_args()

    configs = RL_CONFIGS[args.env] #presetting prameters for each enviroment.

    env = suite.load("pendulum","swingup")
    env._task.__init__(random= args.seed)
    model_path = configs.model_dir+'989th_model_a.pth.tar'

    imit_agent = rl_agent.PPOAgent(env, configs, args.seed)
    imit_agent.test_interact(model_path)

if __name__=="__main__":
    main()
