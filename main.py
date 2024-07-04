import torch
import wandb

from Reward import RewardPredictor
from env import CrossEnv
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

def show(render=False, model="ppo_cross"):
    # 加载模型并测试
    model = PPO.load(model)
    mini_env = gym.make('MiniGrid-Unlock-v0', render_mode="human")
    env = CrossEnv(mini_env, type="cross_3")
    obs, _ = env.reset()
    times = 10
    while times > 0:
        action, _ = model.predict(obs)
        obs, reward, done, info, _ = env.step(action)
        if render:
            env.render()
        if done:
            drop_times, step_nums = env.analyse()
            print(f"drop_times: {drop_times}, step_nums: {step_nums}")
            obs, _ = env.reset()
            times -= 1

def test(env_name, env_type, model, render=False, times=50):
    total_times = times
    mini_env = gym.make(env_name)
    env = CrossEnv(mini_env, use_reward_predictor=False, type=env_type)
    reward_predictor = RewardPredictor(3, 64, 1)
    reward_predictor.load_state_dict(torch.load("reward_predictor_crl_reward.pth"))
    obs, _ = env.reset()
    success_times = 0
    rewards = 0
    now_step = 0
    costs = 0
    total_step_num = 0
    while times > 0:
        action, _ = model.predict(obs)
        # action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)
        if now_step > 500:
            done = True
            now_step = 0
            reward = 0
            env.force_done()
        else:
            now_step += 1
        if render:
            env.render()
        if done:
            now_step = 0
            if reward > 0:
                success_times += 1
            #     print(f"success_times: {success_times}, times: {times}")
            # else:
            #     print(f"fail_times, {reward}")
            success = 1 if reward > 0 else 0
            if env_type == "lava":
                drop_times, step_nums, is_have_cross_lava, is_drop_into_lava = env.analyse2lava()
                rewards += reward_predictor(torch.tensor([success, step_nums, is_have_cross_lava, is_drop_into_lava], dtype=torch.float32)).item()
            else:
                drop_times, step_nums = env.analyse()
                if times == 1:
                    print(f"drop_times: {drop_times}, step_nums: {step_nums}")
                rewards += reward_predictor(torch.tensor([success, drop_times, step_nums], dtype=torch.float32)).item()
                costs += env.analyse2cost()
                total_step_num += step_nums
            obs, _ = env.reset()
            times -= 1
        print(success_times / total_times, rewards / total_times, total_step_num / total_times)
    return success_times / total_times, rewards / total_times


class EvaluateCallback(BaseCallback):
    def __init__(self, model, eval_freq, render=False, verbose=1):
        super(EvaluateCallback, self).__init__()
        self.env_name = "MiniGrid-LavaGapS7-v0"
        self.env_type = "lava"
        self.eval_results_success = []
        self.eval_results_reward = []
        self.eval_drop_times = []
        self.stepNum = []
        self.model = model
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"evaluating model at step {self.n_calls}")
            success_rate, reward = test(self.env_name, self.env_type, self.model, render=False)
            self.eval_results_success.append(success_rate)
            self.eval_results_reward.append(reward)
            self.stepNum.append(self.n_calls)
            print(f"step: {self.n_calls}, success_rate: {success_rate}, reward: {reward}")
        return True

    def _on_training_end(self) -> None:
        import pandas as pd

        data = pd.DataFrame({"Step": self.stepNum, "Success_rate": self.eval_results_success, "Reward": self.eval_results_reward})
        data.to_csv("./difference/test_change_lava.csv")

def collect_drop_times(times=1000):
    total_times = times
    mini_env = gym.make('MiniGrid-Unlock-v0')
    env = CrossEnv(mini_env, use_reward_predictor=True, type="cross_3")
    obs, _ = env.reset()
    drop_times = []
    costs = 0
    rewards = 0
    model = PPO.load("unlock_crl_times_0.4")
    turn_down = 0
    while times > 0:
        action, _ = model.predict(obs)
        # action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)
        # turn_down += 1
        # if turn_down > 500:
        #     print(env.drop_times)
        #     done = True
        #     turn_down = 0
        #     env.force_done()
        #     reward = env.getNowReward()
        if done:
            turn_down = 0
            drop_time, step_num = env.analyse()
            drop_times.append(drop_time)
            costs += env.analyse2cost()
            rewards += reward
            print(f"drop_times: {drop_time}, times: {times}")
            obs, _ = env.reset()
            times -= 1
    # 找平均值
    print(sum(drop_times) / total_times)
    print(costs / total_times)
    print(rewards / total_times)
    return sum(drop_times) / total_times


if __name__ == '__main__':
    mini_env = gym.make('MiniGrid-LavaGapS7-v0')
    env_type = "lava"
    # env = CrossEnv(mini_env, use_reward_predictor=False, type=env_type)
    #     # # wandb.init(project="cross")
    #     # callback = EvaluateCallback(model=None, eval_freq=10000)
    #     # model = PPO("MlpPolicy", env, verbose=2)
    #     # # callback = WandbCallback(
    #     # #     gradient_save_freq=10,
    #     # # )
    #     # success_rates = []
    #     # rewards = []
    #     # steps = []
    #     # model.learn(total_timesteps=250 * 2000, callback=callback)
    # model = PPO.load("unlock_three_times.zip")
    # test("MiniGrid-Unlock-v0", "cross_3", model, render=False, times=100)
    # model.save("QWen_lava_mixtral")
    # show(render=True, model="unlock_two_times")
    # collect_drop_times(times=5)
    # data = np.load("./data/sample_trajectory_QWen_unlock.npy")
    # labels = np.load("./data/labels_QWen_unlock.npy")
    # reward_predictor = RewardPredictor(3, 64, 1)
    # reward_predictor.load_state_dict(torch.load("reward_predictor_mixtral_unlock.pth"))
    # index_list = []
    # success_rate, reward = test("MiniGrid-Unlock-v0", "unlock", model, render=False)
    # print(success_rate, reward)
    # success_rate, reward= test("MiniGrid-LavaGapS7-v0", env_type, model, render=False)
    # success_rates.append(success_rate)
    # rewards.append(reward)
    # steps.append(250 * 2000)
    # model.save("lavaS07_llm")
    # # 写入文件
    # import pandas as pd
    #
    # data = pd.DataFrame({"Step": steps, "Success_rate": success_rates, "Reward": rewards})
    # data.to_csv("./data/lava_llm_QWen.csv")
