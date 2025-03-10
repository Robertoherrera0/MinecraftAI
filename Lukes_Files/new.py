import minerl
import gym

def test_minerl():
    try:
        env = gym.make("MineRLNavigateDense-v0")
        obs = env.reset()

        print("MineRL environment loaded successfully!")

        # Take a few random actions to verify functionality
        for _ in range(5):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, _ = env.step(action)
            print(f"Step Reward: {reward}, Done: {done}")
            if done:
                env.reset()

        env.close()
        print("MineRL test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_minerl()
