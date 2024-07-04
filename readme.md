# 使用大模型塑造奖励函数，在minigrid中训练
## 用法
### 1.收集随机智能体数据
please run env.py to collect data

### 2.调用大模型收集偏好
please run llm.py to collect preference

### 3.根据数据训练奖励函数
please run Reward.py to train reward function

### 4.使用训练好的奖励函数训练智能体
please run main.py to train agent