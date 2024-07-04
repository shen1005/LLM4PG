import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from enum import Enum, unique
import torch
from Reward import RewardPredictor


@unique
class Act(Enum):
    left = 0
    right = 1
    forward = 2
    pickup = 3
    drop = 4
    toggle = 5
    done = 6

@unique
class Obj(Enum):
    unseen = 0
    empty = 1
    wall = 2
    floor = 3
    door = 4
    key = 5
    ball = 6
    box = 7
    goal = 8
    lava = 9
    agent = 10

# 相对于lava所造成的夹角
@unique
class Direction(Enum):
    up = 0
    right = 1
    down = 2
    left = 3

@unique
class env_type(Enum):
    lava = 'MiniGrid-LavaGapS7-v0'
    cross = 'MiniGrid-Unlock-v0'

    '''这里的情况比较复杂。首先分几方面考虑：
    1.如果左前方和右前方都是墙，首先确定的是一定在墙附近。那么可以分成面朝墙和侧对墙。侧对墙一定无法达成这类情况，必然是面对墙，而面对墙必然前面不是空白
    2.如果左前方右前方都是岩浆，侧对岩浆一定不可能，因为岩浆只有一条，只能面对岩浆，而面对岩浆则一定会有前面不是空白
    3.如果一边是墙，一边是岩浆则比较棘手。这种情形一定是同时侧对岩浆和墙。其中，这几种情形很棘手：
    ##############
    #🔺          #
    #   ~~~~~~~~~#          
    #            #
    #            #
    #            #
    ##############
    所以目前的想法是，除了限制条件是左前右前都是岩浆，前方空白之外，还要添加约束：
    左前右前的任意一个岩浆他的前和后都是空白
    为什么说它是对的？主要从两个角度来确认：即不漏不错
    不漏：首先在正确的朝向下，他的岩浆前面必然是空，不可能是墙，因为智能体最终要穿过去;其次，后面必然不能是墙，否则会出现墙穿过岩浆或者智能体在墙内情况
    不错：在上图筛选出的那个条件下，岩浆侧身的时候演讲带只有一处空白，没有别的空白，不可能满足我的约束条件。
    '''

def is_lava_empty(state, idx):
    if state[idx] == Obj.lava.value and state[idx + 1] == Obj.empty.value and state[idx - 1] == Obj.empty.value:
        return True
    return False

def is_lava_gap_towards(state):
    if (state[19] == Obj.lava.value or state[19] == Obj.wall.value) and (state[33] == Obj.lava.value or state[33] == Obj.wall.value) and state[26] == Obj.empty.value and (is_lava_empty(state, 19) or is_lava_empty(state, 33)):
        return True
    return False

# TODO 有bug， 如果左边是墙，右边是顺着面朝方向的岩浆，这个时候也会返回True
def is_in_lava_gap(state):
    if (state[20] == Obj.lava.value or state[20] == Obj.wall.value) and (state[34] == Obj.lava.value or state[34] == Obj.wall.value):
        return True
    return False


class CrossEnv(gym.Env):
    def __init__(self, env, use_reward_predictor=False, type="cross", gamma=0.1):
        super(CrossEnv, self).__init__()
        self.mini_env = env
        self.action_space = env.action_space
        min_value = np.min(env.observation_space['image'].low)
        max_value = np.max(env.observation_space['image'].high)
        if type =="cross_3":
            self.observation_space = gym.spaces.Box(low=min_value,
                                                    high=max_value,
                                                    dtype=env.observation_space['image'].dtype,
                                                    shape=(env.observation_space['image'].shape[0]*env.observation_space['image'].shape[1] + 2,))
        else:
            self.observation_space = gym.spaces.Box(low=min_value,
                                                    high=max_value,
                                                    dtype=env.observation_space['image'].dtype,
                                                    shape=(env.observation_space['image'].shape[0]*env.observation_space['image'].shape[1],))
        self.now_state = None
        self.has_key = False            # 表示智能体是不是拿到了钥匙
        self.drop_times = 0             # 丢下钥匙的次数
        self.stepList = []              # 已经走过的steps
        self.num_steps = 0
        self.actions_dict = {
            'w': self.mini_env.actions.forward,
            'a': self.mini_env.actions.left,
            'd': self.mini_env.actions.right,
            's': self.mini_env.actions.pickup,
            ' ': self.mini_env.actions.toggle,
            'q': self.mini_env.actions.done,
            'e': self.mini_env.actions.drop
        }
        self.use_reward_predictor = use_reward_predictor
        print("use_reward_predictor: ", use_reward_predictor)
        if use_reward_predictor:
            self.reward_model = RewardPredictor(4, 64, 1)
            self.reward_model.load_state_dict(torch.load("reward_predictor.pth"))
            # self.reward_model.load_state_dict(torch.load("reward_predictor_mixtral_unlock.pth"))

        '''for lava env, some parameters are different from cross env'''
        self.env_type = type
        self.direction = Direction.up
        self.lava_state = 0 # 区别于cross env的now_state
        self.is_have_cross_lava = False     # 是否已经穿过lava
        self.down_times = 0
        self.is_drop_into_lava = False

        self.cost = 0

    def ob2state(self, obs):
        array = obs['image']
        # 要把这个7*7*3的array变成7*7, 只取每个3的第一个值
        state = np.array(array[:, :, 0]).flatten()
        # 再在后面加上钥匙的状态
        if self.env_type == "cross_3":
            state = np.append(state, self.drop_times)
            state = np.append(state, self.has_key)
            # if self.drop_times == 3:
                #print("--------get 3 times---------")
        return state

    def reset(self, **kwargs):
        obs = self.mini_env.reset()[0]
        state = self.ob2state(obs)
        self.now_state = state
        self.has_key = False
        self.drop_times = 0
        self.stepList = []
        self.num_steps = 0

        self.direction = Direction.up
        self.lava_state = 0  # 区别于cross env的now_state
        self.is_have_cross_lava = False  # 是否已经穿过lava
        self.down_times = 0
        self.is_drop_into_lava = False
        state = self.ob2state(obs)
        self.cost = 0
        return state, {}

    def render(self):
        return self.mini_env.render()

    # 输入现在的state和action, 看看是不是让钥匙状态发生了改变
    def key_change(self, state, action):
        if action == Act.drop.value and self.has_key:
            self.has_key = False
            self.stepList.append(self.num_steps)
            self.num_steps = 0
            self.drop_times += 1
        if action == Act.pickup.value and state[26] == Obj.key.value:
            self.has_key = True
            self.stepList.append(self.num_steps)
            self.num_steps = 0

    def lava_change(self, state, action):
        if is_lava_gap_towards(state) and action == Act.forward.value:
            if self.lava_state != 0:
                print("error")
            assert self.lava_state == 0     # 预防一下bug
            self.lava_state = 1
            self.direction = Direction.up.value if self.is_have_cross_lava is False else Direction.down.value
        elif self.lava_state == 1 and action == Act.forward.value:
            '''这里要注意万一旁边是墙，向前走就相当于没动'''
            if self.direction == Direction.up.value:
                self.is_have_cross_lava = True
                self.lava_state = 0
            elif self.direction == Direction.down.value:
                self.is_have_cross_lava = False
                self.lava_state = 0
                self.down_times += 1
            else:
                pass
        elif self.lava_state == 1 and action == Act.left.value:
            '''向左转， 这里也要根据当前的朝向调整方向'''
            if self.direction == Direction.up.value:
                self.direction = Direction.left.value
            elif self.direction == Direction.down.value:
                self.direction = Direction.right.value
            elif self.direction == Direction.left.value:
                self.direction = Direction.down.value
            elif self.direction == Direction.right.value:
                self.direction = Direction.up.value
        elif self.lava_state == 1 and action == Act.right.value:
            '''向右转同理'''
            if self.direction == Direction.up.value:
                self.direction = Direction.right.value
            elif self.direction == Direction.down.value:
                self.direction = Direction.left.value
            elif self.direction == Direction.left.value:
                self.direction = Direction.up.value
            elif self.direction == Direction.right.value:
                self.direction = Direction.down.value
        if state[26] == Obj.lava.value and action == Act.forward.value:
            self.is_drop_into_lava = True


    def step(self, action):
        # 注意这里的前后顺序：pick up 的步骤算到了拿取钥匙了的步骤里
        self.num_steps += 1
        key_before = self.has_key
        if self.env_type == "lava":
            self.lava_change(self.now_state, action)
        else:
            self.key_change(self.now_state, action)
        obs, reward, done, info, _ = self.mini_env.step(action)
        state = self.ob2state(obs)
        self.now_state = state
        if done:
            self.stepList.append(self.num_steps)
            self.num_steps = 0
            success = 1 if reward > 0 else 0
            if self.use_reward_predictor and (self.env_type == "cross" or self.env_type == "cross_3"):
                reward = self.reward_model(torch.tensor([success, self.drop_times, sum(self.stepList)], dtype=torch.float32)).item()
                self.cost = (self.drop_times - 3) ** 2
                #reward = reward - 0.2 * self.cost
                # print(f"reward: {reward}")
                # print(f"{self.use_reward_predictor}, {self.env_type}")
                # exit(0)
            elif self.use_reward_predictor and self.env_type == "lava":
                reward = self.reward_model(torch.tensor([success, sum(self.stepList), self.is_have_cross_lava, self.is_drop_into_lava], dtype=torch.float32)).item()
            else:
                #reward = 0 if reward <= 0 else reward
                self.cost = (self.drop_times - 3) ** 2
                # reward = reward - 0.9 * self.cost
                reward = -sum(self.stepList) if reward <= 0 else reward
                # reward = reward
        return state, reward, done, info, {}

    def getNowReward(self):
        return self.reward_model(torch.tensor([0, self.drop_times, sum(self.stepList)], dtype=torch.float32)).item()

    def force_done(self):
        self.stepList.append(self.num_steps)
        self.cost = (self.drop_times - 3) ** 2
        self.num_steps = 0

    def analyse(self):
        step_num = sum(self.stepList)
        return self.drop_times, step_num

    def analyse2lava(self):
        return self.down_times, sum(self.stepList), self.is_have_cross_lava, self.is_drop_into_lava

    def analyse2cost(self):
        return self.cost

def sampleData(env_name='MiniGrid-Unlock-v0', type="cross", times=500):
    render = False
    resultWriter = []
    if render:
        mini_env = gym.make(env_name, render_mode="human")
    else:
        mini_env = gym.make(env_name)
    env = CrossEnv(mini_env, type=type)
    obs, _ = env.reset()
    times = times
    step_num = 0
    while times > 0:
        # action = env.actions_dict[input("请输入动作: ")]
        action = env.action_space.sample()
        # noinspection PyTupleAssignmentBalance
        obs, reward, done, info, _ = env.step(action)
        step_num += 1
        if reward != 0:
            print("reward: ", reward)
            print(action)
            print(env.analyse())
        if render:
            env.render()
        if done or step_num > 500:
            step_num = 0
            env.force_done()
            if reward == 0:
                result = 0
            elif reward > 0:
                result = reward
            else:
                # raise Exception("reward < 0")
                result = reward
            # 轨迹
            if env.env_type == "lava":
                drop_times, step_nums, is_have_cross_lava, is_drop_into_lava = env.analyse2lava()
                resultWriter.append([result, step_nums, is_have_cross_lava, is_drop_into_lava])
            else:
                drop_times, step_nums = env.analyse()
                resultWriter.append([result, drop_times, step_nums])
            obs = env.reset()[0]
            env.force_done()
            print("done, ------------------------------------")
            times -= 1
            print(f"剩余次数: {times}")
    # 排序
    resultWriter.sort(key=lambda x: x[0])
    # 变成np.array
    resultWriter = np.array(resultWriter)
    # 写入文件
    print(resultWriter)
    with open("result.txt", "w") as f:
        for i in resultWriter:
            if env.env_type == "lava":
                f.write(f"{i[0]} {i[1]} {i[2]} {i[3]}\n")
            else:
                f.write(f"{i[0]} {i[1]} {i[2]}\n")
    # 存成npy
    np.save("result.npy", resultWriter)

def operate_by_hand():
    env = gym.make('MiniGrid-Unlock-v0', render_mode="human")
    env = CrossEnv(env)
    obs, _ = env.reset()
    while True:
        action = input("请输入动作: ")
        action = env.actions_dict[action]
        print(action)
        obs, reward, done, info, _ = env.step(action)
        env.render()
        if done:
            print(reward, info)
            break


if __name__ == '__main__':
    env_name = "MiniGrid-Unlock-v0"
    #sampleData(env_name=env_name, type="cross", times=5000)
    data = np.load("result.npy")
    # 第一个元素如果是正数，变成1，如果是负数，变成0
    data[:, 0] = np.where(data[:, 0] > 0, 1, 0)
    with open("data.txt", "w") as f:
        for i in data:
            f.write(f"{i[0]} {i[1]} {i[2]}\n")
    np.save("data2.npy", data)





