from numpy.lib.function_base import select
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from other.Show import Show
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, lr, input_dims, h1_dims, h2_dims, actions, action_lock, 
                    log_std_min, log_std_max, init_w = 3e-3) -> None:

        super(Actor, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.h1_dims = h1_dims
        self.h2_dims = h2_dims
        self.actions = actions
        self.action_lock = action_lock
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.h1 = nn.Linear(self.input_dims, self.h1_dims)
        self.h2 = nn.Linear(self.h1_dims, self.h2_dims)

        self.mean_linear = nn.Linear(self.h2_dims, self.actions)
        self.mean_linear.weight.data.uniform_(-init_w,init_w)
        self.mean_linear.bias.data.uniform_(-init_w,init_w)

        self.log_std_linear = nn.Linear(self.h2_dims,self.actions)
        self.log_std_linear.weight.data.uniform_(-init_w,init_w)
        self.log_std_linear.bias.data.uniform_(-init_w,init_w)

        self.optim = optim.Adam(self.parameters(), lr = self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):

        state = T.Tensor(observation).to(self.device)
        x = T.relu(self.h1(state))
        x = T.relu(self.h2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class PolicyGradient:
    def __init__(self, lr, gamma, input_dims, h1_dims, h2_dims, actions, action_lock, log_std_min, log_std_max) -> None:
        super(PolicyGradient,self).__init__()

        self.gamma = gamma
        self.loss_list = []
        self.reward_storage = []
        self.action_storage = []
        self.policy = Actor(lr, input_dims, h1_dims, h2_dims, actions, action_lock, log_std_min, log_std_max)
    
    def choose_action(self, observation):
        state = T.FloatTensor(observation).to(self.policy.device)
        mean, log_std = self.policy.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        action      = normal.sample()
        log_probs = normal.log_prob(action)
        log_prob = T.mean(log_probs)
        self.action_storage.append(log_prob)

        action = T.tanh(action)
        action = T.clamp(action,-self.policy.action_lock,self.policy.action_lock)
        action  = action.detach().cpu().numpy()
        return action

    def store_rewards(self, reward):
        self.reward_storage.append(reward)

    def learn(self,n_episodes,temp_signal):
        self.policy.optim.zero_grad()
        self.reward_storage = np.array(self.reward_storage)
        signal = 0
        # Decrease the variance by the baseline - average value of the reward
        for i in range(len(self.reward_storage)):
            if temp_signal[i] == 1:
                self.reward_storage[signal:i+1] = self.reward_storage[signal:i+1] - np.mean(self.reward_storage[signal:i+1])

        G = []
        A = []
        loss = 0
        for i in range(len(self.reward_storage)):
            # G = np.zeros_like(self.reward_storage, dtype=np.float)
            G_sum = 0
            discount = 1
            for j in range(i, len(self.reward_storage)):
                G_sum += self.reward_storage[j] * discount
                discount *= self.gamma
                if temp_signal[j] == 1:
                    break
            G.append(G_sum)
            A.append(self.action_storage[i])
            if temp_signal[i] == 1:
                mean = np.mean(G)
                std  = np.std(G) if np.std(G) > 0 else 1
                G = (G - mean) / std
                G = T.tensor(G, dtype = T.float64).to(self.policy.device)
                for g, logprob in zip(G,A):
                    loss += -g * logprob
                A = []
                G = []

        loss /= n_episodes
        self.loss_list.append(loss)
        loss.backward()
        self.action_storage = []
        self.reward_storage = []
        self.policy.optim.step()

# input parameters
# run(args.env_chosse,args.lr,args.gamma,args.input_dims,args.h1_dims,args.h1_dims,args.actions,args.action_lock,
#         args.epoch_nums,args.sample_nums,args.t_horizens,args.env, args.path)

def run(env_choose, lr, gamma, input_dims, h1_dims, h2_dims, actions, action_lock, log_std_min, log_std_max,
            epoch_nums, sample_nums, t_horizens, env, path):
    
    agent = PolicyGradient(lr, gamma, input_dims, h1_dims, h2_dims, actions, action_lock, log_std_min, log_std_max)
    score_history = []
    score = 0
    t_horizens_history = []
    # 一共有n_epochs次theta参数更新
    for epoch in range(epoch_nums):
        temp_score = []
        temp_signal = []
        temp_horizens = []

        
        # 每次采 n_episodes 个样本
        for i in range(sample_nums):
            
            done = False
            score = 0
            t = 0
            observation = env.reset()
            # 每个样本有t_horizens个时期
            for t in range(t_horizens):
                if env_choose != 'MazeNavigation' and epoch > 480:
                    env.render()
                # 从神经网络采集 t 时期的 action，并保存动作的概率值
                action = agent.choose_action(observation)
                # 从环境中获取 reward 值与新的状态 observation 值
                observation_, reward, done, _ = env.step(action)
                # 存储 reward 值
                agent.store_rewards(reward)
                # 更新环境
                observation = observation_
                # 累加当前时期的 reward 值
                score += reward
                if done:
                    break
                if t != (t_horizens-1):
                    temp_signal.append(0)
            
            if (epoch+1) % 20 == 0:
                print('epoch: ',epoch+1,'episode: ',i+1, 'score %.3f' % score)
            # 作为每一个样本的间隔标志
            temp_signal.append(1)
            # 总的无加权reward值
            temp_score.append(score)
            # 记录步数
            temp_horizens.append(t+1)
        
        # 记录 n_episodes 个样本的无加权reward平均值
        score_history.append(np.mean(temp_score))
        # 使用 1 / n_episodes 的值作为梯度，更新theta值
        agent.learn(n_episodes=sample_nums, temp_signal=temp_signal)
        # 记录平均步数
        t_horizens_history.append(np.mean(temp_horizens))
    
    if env_choose != 'MazeNavigation':
        env.close()
    # Reward & loss 函数值变化图
    title = f"reward-loss-lr-{lr}-gamma-{gamma}-sample_nums-{sample_nums}-t_horizens-{t_horizens}"
    filename = title + '.jpg'
    filename = path + filename

    Show(t_horizens_history, score_history, agent.loss_list, Name1='t_horizens', Name2='reward', Name3 = 'loss',
                three=True, fileName=filename, title=title)