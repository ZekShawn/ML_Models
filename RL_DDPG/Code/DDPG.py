import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from other.Show import Show


#Hyperparameters
lr_a            = 0.0005
lr_q            = 0.001
gamma           = 0.99
train_steps     = 10
tau             = 0.005                        # for target network soft update
env_choose      = 'MazeNavigation'             # 'HalfCheetah-v2' ,  'MazeNavigation'
q_dims          = 1
t_horizens      = 2500
load_model      = False
print_interval  = 1
is_learn        = True

if env_choose != 'MazeNavigation':
    import gym
    env          = gym.make(env_choose)
    buffer_limit = 20000
    epoch_nums   = 20000
    batch_size   = 250        # make the batch_size be large
    train_point  = 5000       # make the train_point be large
    s_dims       = 17
    a_dims       = 6
    a_lock       = 1
else:
    from other.Maze import ENV
    env          = ENV()
    buffer_limit = 5000
    epoch_nums   = 500
    batch_size   = 5
    train_point  = 1000
    s_dims       = 2
    a_dims       = 2
    a_lock       = 0.1
    t_horizens   //= 50


# File saving
separate = '\\'
if os.name == 'posix':
    separate = '/'
title           = f'lr_a{lr_a}-lr_q{lr_q}-gamma{gamma}-batch_size{batch_size}-train_steps{train_steps}'
filename        = f'data{separate}{env_choose}{separate}{title}.jpg'
save_actor      = f'save{separate}{env_choose}-actor.pt'
save_critic     = f'save{separate}{env_choose}-critic.pt'


class ANet(nn.Module):
    def __init__(self):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, a_dims)


    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * a_lock # Multipled by a_lock
        
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(s_dims, 64)
        self.fc_a = nn.Linear(a_dims,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,q_dims)


    def forward(self, x, a):
        x = torch.as_tensor(x, dtype=torch.float)
        a = torch.as_tensor(a, dtype=torch.float)
        h1 = F.relu(self.fc_s(x))
        h1 = torch.squeeze(h1)
        h2 = F.relu(self.fc_a(a))
        h2 = torch.squeeze(h2)
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)

        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)


    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class ReplayBuffer():
    def __init__(self) -> None:
        self.buffer = []
    

    def size(self):
        return len(self.buffer)


    def storage(self,a,r,s,s_,done):

        if self.size() <= buffer_limit:
            row_buffer = []
            row_buffer.append(a.detach().numpy())
            row_buffer.append(r)
            row_buffer.append(s)
            row_buffer.append(s_)
            row_buffer.append(done)
            self.buffer.append(row_buffer)
    

    def sample(self):
        mini_batch = random.sample(self.buffer, batch_size)
        a_lst, r_lst, s_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for batch in mini_batch:
            a, r, s, s_, done = batch
            a_lst.append(a)
            r_lst.append([r])
            s_lst.append(s)
            s_prime_lst.append(s_)
            done = 0.0 if done else 1.0 
            done_mask_lst.append([done])

        return torch.as_tensor(a_lst, dtype=torch.float), torch.as_tensor(r_lst, dtype=torch.float), \
                torch.as_tensor(s_lst, dtype=torch.float), torch.as_tensor(s_prime_lst, dtype=torch.float), \
                torch.as_tensor(done_mask_lst, dtype=torch.float)


def train(memory, actor, critic, target_actor, target_critic, actor_optim, critic_optim):
    a, r, s, s_, done = memory.sample()

    target = r + gamma * target_critic(s_, target_actor(s_)) * done
    q_loss = F.smooth_l1_loss(critic(s,a), target)
    critic_optim.zero_grad()
    q_loss.backward()
    critic_optim.step()
    
    a_loss = -critic(s,actor(s)).mean() # That's all for the policy loss.
    actor_optim.zero_grad()
    a_loss.backward()
    actor_optim.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def learn():
    score = 0
    score_memory = []
    memory = ReplayBuffer()
    if load_model:
        actor, target_actor = torch.load(save_actor), torch.load(save_actor)
        critic, target_critic = torch.load(save_critic), torch.load(save_critic)
    else:
        actor, target_actor = ANet(), ANet()
        critic, target_critic = QNet(), QNet()
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    actor_optim = optim.Adam(actor.parameters(), lr=lr_a)
    critic_optim = optim.Adam(critic.parameters(), lr=lr_q)
    

    for epoch in range(epoch_nums):
        s = env.reset()
        done = False
        step = 0


        # learning from the buffer
        while (not done) and (step < t_horizens):
            a = actor(s)
            a += ou_noise()[0]              # Add the Gause noise
            s_, r, done, _ = env.step(a.detach().numpy())
            memory.storage(a,r,s,s_,done)
            s = s_
            step += 1
        

        if memory.size() > train_point:
            for _ in range(train_steps):
                train(memory, actor, critic, target_actor, target_critic, actor_optim, critic_optim)
                soft_update(actor, target_actor)
                soft_update(critic, target_critic)
        

        # policy action
        s = env.reset()
        done = False
        step = 0
        while (not done) and (step < t_horizens):
            if (epoch + 1) >= (epoch_nums - 2):
                env.render()
            a = target_actor(s)
            s, r, done, _ = env.step(a.detach().numpy())
            score += r
            step  += 1


        if (epoch+1) % print_interval == 0:
            print(f"Learning Epoch: {epoch+1}, average scores: {score / print_interval}")
            score_memory.append(score / print_interval)
            score = 0

    torch.save(target_actor, save_actor)
    torch.save(target_critic, save_critic)
    Show(score_memory, Name1='Score', fileName=filename, title=title)
    env.close()


def run():
    actor = torch.load(save_actor)
    for _ in range(epoch_nums):
        s = env.reset()
        done = False

        while (not done):
            env.render()
            a = actor(s)
            s, _, done, _ = env.step(a.detach().numpy())


if __name__ == '__main__':
    if is_learn:
        learn()
    else:
        run()