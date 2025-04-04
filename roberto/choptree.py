import gym
import minerl
from stable_baselines3 import PPO #type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv #type: ignore
from stable_baselines3.common.evaluation import evaluate_policy #type: ignore
from stable_baselines3.common.monitor import Monitor #type: ignore

from obs_wrapper import FlattenObservationWrapper
from action_wrapper import DictToMultiDiscreteWrapper 
from reward_wrapper import LogRewardWrapper

class ForceMineRLRenderWrapper(gym.Wrapper):
    def render(self, mode='human'):
        # Directly call the underlying environment’s original render
        # ignoring the compatibility layer.
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped.render(mode=mode)
        return super().render(mode=mode)

def make_env():
    env = gym.make("MineRLObtainDiamondShovel-v0")
    # env = Monitor(env) 

    env = DictToMultiDiscreteWrapper(env)
    env = FlattenObservationWrapper(env)
    env = LogRewardWrapper(env)
    # env = ForceMineRLRenderWrapper(env)
    return env


def main():
    env = DummyVecEnv([make_env])
    
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save("minerl_multidiscrete_log_ppo")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Eval results: mean={mean_reward}, std={std_reward}")

    # quick test rollout
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, rew, done, info = env.step(action)
        # env.render()
        if done[0]:
            obs = env.reset()
    
    env.close()

if __name__ == '__main__':
    main()
