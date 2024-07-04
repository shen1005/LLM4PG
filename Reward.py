import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class RewardPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RewardPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        #nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        #nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc3(x))
        x = self.fc2(x)
        # x = -F.sigmoid(x)
        return x

def train():
    # wandb.init(project="reward_predictor")
    writer = SummaryWriter("reward_predictor")
    data = np.load("./data/sample_trajectory_QWen.npy")  
    labels = np.load("./data/labels_QWen.npy")  
    # labels = np.load("./data/labels.npy") 
    # 总共250组, 200训练，50测试
    train_data = data[:200]
    train_labels = labels[:200]
    test_data = data[250:]
    test_labels = labels[250:]
    # 转化为tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    # 初始化模型
    reward_predictor = RewardPredictor(4, 64, 1)
    optimizer = Adam(reward_predictor.parameters(), lr=1e-3)
    # 训练
    epochs = 5000
    for epoch in range(epochs):
        reward1 = reward_predictor(train_data[:, 0])
        reward2 = reward_predictor(train_data[:, 1])
        pref = torch.softmax(torch.cat((reward1, reward2), dim=1), dim=1)
        loss = F.cross_entropy(pref, train_labels.long()) + 0.001 * (reward1 ** 2).mean() + 0.001 * (reward2 ** 2).mean()
        # wandb.log({"loss": loss.item()})
        writer.add_scalar("loss", loss.item(), epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")
    # 测试
    reward1 = reward_predictor(test_data[:, 0])
    reward2 = reward_predictor(test_data[:, 1])
    pref = torch.softmax(torch.cat((reward1, reward2), dim=1), dim=1)
    test_loss = F.cross_entropy(pref, test_labels.long())
    print(f"test_loss: {test_loss.item()}")
    # 保存模型
    torch.save(reward_predictor.state_dict(), "reward_predictor.pth")
    writer.close()


if __name__ == '__main__':
    #train()
    model = RewardPredictor(3, 64, 1)
    model.load_state_dict(torch.load("reward_predictor_crl_reward.pth"))
    # model.eval()
    # data = np.load("data.npy")
    # # 变成tensor输入到模型
    # data = torch.tensor(data, dtype=torch.float32)
    # # 得到reward
    # reward = model(data)
    # with open("reward.txt", "w") as f:
    #     f.write(str(reward))
    # # 画图
    # import matplotlib.pyplot as plt
    # plt.plot(reward.detach().numpy())
    # plt.show()