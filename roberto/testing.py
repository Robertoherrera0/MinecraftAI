import gym
import minerl
from stable_baselines3 import PPO

from obs_wrapper import FlattenObservationWrapper
from action_wrapper import DictToMultiDiscreteWrapper
from reward_wrapper import LogRewardWrapper

def make_env(debug=False):
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env = DictToMultiDiscreteWrapper(env)
    env = FlattenObservationWrapper(env, inv_items=["log"])
    env = LogRewardWrapper(env, debug=debug)
    return env

def main():
    print("Loading environment and model...")
    env = make_env(debug=False)
    model = PPO.load("ppo_choptree_model")

    obs = env.reset()
    max_logs = 0

    print("Starting rendering...")
    for step in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        env.render()

        logs = obs["inv"][0]
        max_logs = max(max_logs, logs)

        print(f"[STEP {step}] Reward: {reward:.2f} | Logs: {logs}")

        if done:
            print("Done. Resetting environment...\n")
            obs = env.reset()

    print(f"\nMax logs collected: {max_logs}")
    env.close()

if __name__ == "__main__":
    main()
