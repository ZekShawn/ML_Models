import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from other.Show import Show
from torch.distributions import Normal as N


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


class Critic(nn.Module):
    def __init__(self, lr, critic_input_dims, h1_dims, h2_dims, critic_values, log_std_min, 
                    log_std_max, init_w = 3e-3) -> None:

        super(Critic, self).__init__()
        self.lr = lr
        self.critic_input_dims = critic_input_dims
        self.h1_dims = h1_dims
        self.h2_dims = h2_dims
        self.critic_values = critic_values
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.h1 = nn.Linear(self.critic_input_dims, self.h1_dims)
        self.h2 = nn.Linear(self.h1_dims, self.h2_dims)

        self.mean_linear = nn.Linear(self.h2_dims, self.critic_values)
        self.mean_linear.weight.data.uniform_(-init_w,init_w)
        self.mean_linear.bias.data.uniform_(-init_w,init_w)

        self.log_std_linear = nn.Linear(self.h2_dims,self.critic_values)
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


class ActorCritic:
    def __init__(self, lr, gamma, input_dims, critic_input_dims, h1_dims, h2_dims, actions, critic_values, action_lock, log_std_min, log_std_max) -> None:
        super(ActorCritic,self).__init__()

        self.gamma = gamma
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.action_storage = []
        self.reward_storage = []
        self.state_storage = []
        self.value_storage = []
        self.advantage_storage = []
        self.actor = Actor(lr, input_dims, h1_dims, h2_dims, actions, action_lock, log_std_min, log_std_max)
        self.critic = Critic(lr, critic_input_dims, h1_dims, h2_dims, critic_values, log_std_min, log_std_max)
    

    def value_action(self, observation, reward):

        state = list(observation)
        state.append(reward)
        state = T.FloatTensor(state).to(self.critic.device)
        mean, log_std = self.critic.forward(state)
        std = log_std.exp()
        normal = N(mean,std)
        value = normal.sample()
        value = value.detach().cpu().numpy()

        return value[0]


    def choose_action(self, observation, value):

        state = list(observation)
        state.append(value)
        state = T.FloatTensor(state).to(self.actor.device)
        mean, log_std = self.actor.forward(state)
        std = log_std.exp()
        normal = N(mean, std)
        z      = normal.sample()
        action = T.tanh(z)
        log_probs = normal.log_prob(z)
        log_prob = T.mean(log_probs)
        self.action_storage.append(log_prob)
        action = T.clamp(action, -self.actor.action_lock, self.actor.action_lock)
        action  = action.detach().cpu().numpy()
        return action


    def store(self, reward, observation, value):
        self.reward_storage.append(reward)
        self.state_storage.append(observation)
        self.value_storage.append(value)


    def value_loss(self):
        loss = 0
        for v,t in zip(self.value_storage, self.reward_storage):
            loss += (v - t) ** 2
        loss /= len(self.reward_storage)

        loss = T.tensor(loss, dtype=T.float64, requires_grad=True)
        return loss


    def advantage_function(self, temp_signal):

        G = []
        A = []
        A_next = []
        loss = 0
        for i in range(len(self.value_storage)):
            # G = np.zeros_like(self.reward_storage, dtype=np.float)
            G_sum = 0
            discount = 1
            for j in range(i, len(self.value_storage)):
                G_sum += self.value_storage[j] * discount
                discount *= self.gamma
                if temp_signal[j] == 1:
                    break
            G.append(G_sum)
            if temp_signal[i] == 1:
                mean = np.mean(G)
                std  = np.std(G) if np.std(G) > 0 else 1
                G = (G - mean) / std
                G = T.tensor(G, dtype = T.float64).to(self.actor.device)
                for g in G:
                    A.append(g)
                G = []
        
        for i in range(len(A)):
            if temp_signal[i] == 1:
                A_next.append(self.value_action(self.state_storage[i], self.reward_storage[i]))
            else:
                A_next.append(A[i+1])
        
        A = np.array(A)
        A_next = np.array(A_next)
        A = self.gamma * A_next - A
        for i in range(len(A)):
            A[i] += self.reward_storage[i]

        A = T.tensor(A, dtype=T.float64, requires_grad=True)
        A_next = T.tensor(A_next, dtype=T.float64, requires_grad=True)
        return A
        
    def action_loss(self, temp_signal, sample_nums):

        a_function = self.advantage_function(temp_signal)
        loss = 0
        for i in range(len(a_function)):
            loss -= self.action_storage[i] * a_function[i]
        loss /= sample_nums

        return loss

    def learn(self, v_loss, a_loss):
        self.actor.optim.zero_grad()
        self.critic.optim.zero_grad()
        self.actor_loss_list.append(a_loss)
        self.critic_loss_list.append(v_loss)
        v_loss.backward()
        a_loss.backward()
        self.actor.optim.step()
        self.critic.optim.step()
    
    def reset(self):
        self.action_storage = []
        self.reward_storage = []
        self.state_storage = []
        self.value_storage = []
        self.advantage_storage = []


# run(args.env_chosse, args.lr, args.gamma, args.input_dims, args.h1_dims, args.h1_dims, args.actions, args.action_lock, 
#         args.log_std_min, args.log_std_max, args.epoch_nums, args.sample_nums, args.t_horizens, args.env, args.path)
def run(env_choose, lr, gamma, input_dims, critic_input_dims, h1_dims, h2_dims, actions, critic_values, action_lock, log_std_min, log_std_max, epoch_nums, sample_nums, t_horizens, env, path):

    score_history = []
    t_horizens_history = []
    agent = ActorCritic(lr, gamma, input_dims, critic_input_dims, h1_dims, h2_dims, actions, critic_values, action_lock, log_std_min, log_std_max)
    
    # 一共有 n_epochs 次theta参数更新
    for epoch in range(epoch_nums):
        t = 0
        score = 0
        temp_score = []
        temp_signal = []
        temp_horizens = []


        # Step 1 Collect a set of trajectories on current policy
        # 每次采 n_episodes 个样本
        for i in range(sample_nums):
            t = 0
            score = 0
            value = 0
            reward = 0
            done = False
            temp_value = 0
            observation = env.reset()

            
            # 每个样本有t_horizens个时期
            for t in range(t_horizens):
                if env_choose != 'MazeNavigation' and epoch > 490:
                    env.render()
                
                # 1.1 Value the current state
                value = agent.value_action(observation, reward)

                # 1.2 Actor chooses the action
                action = agent.choose_action(observation, value)

                # 1.3 Env return forward state and reward
                observation, reward, done, _ = env.step(action)

                # 1.4 Store the reward and the state
                agent.store(reward, observation, value)

                # 1.5 Store any other info 
                score += reward
                temp_value += value

                # 1.6 Add the signal whether done
                if done:
                    break
                if t != (t_horizens-1):
                    temp_signal.append(0)


            # 输出每轮的信息
            if (epoch+1) % 20 == 0:
                print('epoch: ',epoch+1,'sampple: ',i+1, 'reward %.3f' % score, 'value %.3f' % (temp_value/t))
            # 作为每一个样本的间隔标志
            temp_signal.append(1)
            # 总的无加权reward值
            temp_score.append(score)
            # 记录步数
            temp_horizens.append(t+1)
        
        # Step 2 Calculate the value_loss and action_loss
        value_loss = agent.value_loss()
        action_loss = agent.action_loss(temp_signal, sample_nums)

        # Step 3 Update the model
        agent.learn(value_loss, action_loss)
        agent.reset()

        # Step 4 Record info
        # 记录 sample_nums 个样本的无加权reward平均值
        score_history.append(np.mean(temp_score))
        # 记录平均步数
        t_horizens_history.append(np.mean(temp_horizens))
    
    # Reward & loss 函数值变化图
    title = f"reward-loss-lr-{lr}-gamma-{gamma}-sample_nums-{sample_nums}-t_horizens-{t_horizens}"
    filename = path + title
    filename += '.jpg'
    Show(t_horizens_history, score_history, Name1='t_horizens', Name2='reward',
                double=True, fileName=filename, title=title)