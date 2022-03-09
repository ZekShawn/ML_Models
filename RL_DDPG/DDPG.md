## What is DDPG ?

DDPG, its full name is 'Deep Deterministic Policy Gradient'. We have learned the 'Policy Gradient' before, and 'Deep' means to use the deep neural network to fit the Q function.

$\theta = \theta + \alpha \triangledown_{\theta}J_{\theta}$	(1)

$\triangledown_{\theta} J_{\theta} = \frac{1}{N} \sum_{i=1}^{N} (\sum_{t=1}^{T} \triangledown_{\theta} log \pi_{\theta}(a_t^i|s_t^i)(\sum_{t'=t}^{T} \gamma^{t'-1} Q(s_{t'},a_{t'})))$	(2)

What is the 'Deterministic'? We use the outputs of network to fit the mean and variance of normal distribution. And we sample actions from this distribution, it's not deterministic. Now, we just view the outputs as the actions, it is deterministic.

Let's review the DQN, the process shows it can not handle the continuous control problem. That's for we should get the maximize Q function with the optimizing action $a_i$ to arrive sate $s_{i'}$.  

<img src="https://pic2.zhimg.com/80/v2-f79600fe97c508e7984bd222ef3f587d_720w.jpg" style="zoom: 67%;" />

So, how we solve this problem? Actually it's easy to get experience from DQN. We use the deep neural network to fit the Q function. 

But the goal of the DDPG is to increase the grad. Because of the network actor is finding the maximized value. So it's a off-line strategy without the importance sampling.

<img src="https://pic1.zhimg.com/80/v2-300bc62b29fa41b5ee5a11d8ae5ca128_720w.jpg" style="zoom:67%;" />

Critic Network:

- We use the critic network to evaluate the Q function in current state and action.
- Input elements: action, state.
- Update method: 
- <img src="https://pic1.imgdb.cn/item/5fc0c83315e77190846760ba.png" alt="image-20201127173444201" style="zoom: 33%;" />

Actor Network:

- We use the actor network to get the action.
- Input elements: current state.
- Update method:
- <img src="https://pic1.imgdb.cn/item/5fc0c86715e7719084677af2.png" alt="image-20201127173536387" style="zoom: 25%;" />

The logistical code step as follow:

<img src="https://pic1.imgdb.cn/item/5fc0c42115e771908464c5a2.png" alt="image-20201127171715954" style="zoom:67%;" />

And the the process as below (maybe not use TD-error):

<img src="https://pic4.zhimg.com/v2-cca7a5ac0cab40cd63ad9fd3679ba333_1440w.jpg?source=172ae18b" style="zoom:67%;" />

## Where the difference between DDPG and PG or AC?

Compare with PG:

1. PG uses the actor network to fit the distribution of action, is not deterministic. While ‘DDPG’ use the actor network to get the deterministic action which can maximize the Q function. 
2. PG uses the  accumulative rewards to evaluate the advantage of next state compared with current state. While ‘DDPG’ use the another network which called critic to evaluate the action.
3. PG uses the same neural network to learn and make policy decision (on-policy). While ‘DDPG’ use the actor-critic to learn and target actor-critic network to make policy decision (off-policy) .
4. Of course, its update methods of parameters are different, refer to the forward theory classify.
5. PG sample a batch size sequential info set from the actor network, while ‘DDPG’ get from rely buffer uniform randomly.

Compare with AC:

1. Update method of AC, which is different:<img src="https://pic1.imgdb.cn/item/5fc0dd6c15e77190846db17c.png" style="zoom:33%;" />
2. AC use same net work to learn, critic and fit. It is an on-policy algorithm while ‘DDPG’ is not.<img src="https://pic1.imgdb.cn/item/5fc0dcea15e77190846d8f9f.png" style="zoom: 33%;" />
3. AC get the sequential samples from the network, while DDPG get uniform randomly from rely buffer.

## How about the code of DDPG?

**Problem 1: Maze Navigation, we need to train the robot to get the right route from start to goal.**

<img src="https://pic.imgdb.cn/item/5fb3f045b18d62711324ebc8.png" alt="image-20201117234605775" style="zoom: 50%;" />

So, there are some parameters be definite in the below:

```python
#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
train_times  = 10000
each_train_times = 10
batch_size   = 50
buffer_limit = 5000
tau          = 0.005 # for target network soft update
```

The environment settings as below:

```bash
# Maze Nvigation
action_dims			= 2
state_dims			= 2
out_of_border_reward= -100
goal_area			= [0.4,0.5]^2
goal_area_reward	= 100
map_area			= [-0.5,0.5]^2
start_point			= [0,0]
other_area_in_map	= -1
action_lock			= [-0.1,0.1]^2
```

The result of ‘DDPG’ in Maze Navigation (it has trained 10000 epochs actually, and record the average reward every 20 steps):

<img src="https://pic1.imgdb.cn/item/5fc261a2d590d4788a803a92.jpg" alt="save-lr_a0.0005-lr_q0.001-gamma0.99-batch_size5-train_steps10" style="zoom: 18%;" />

It seems great, and it's really smoothly. I feel awesome with my code.

Really strange thing is I always get reward number: -104. Which means the robot run 4 steps in map area and steps out at the fifth step. Another strange thing is the robot thinks this is the best result in the mechanism of ‘DDPG’. How's the world? Maybe the inner goal area will be more friendly for the robot.

**Problem 2: We need to make the robot walk forward.**

I don't know the full informations about this environment. But I know the max location is infinite and the min is negative infinite. The max action step is 1, the min action step value is -1. And the observation of environment are 17 axis which action are 6. And I find a really funny video online, which used ‘PPO’ to train the robot take forward by the environment ‘HalfCheetah-v2’. And he used more than 0.2 million steps to train it. At that before, the robot even seemly don't take actions.

Here's the link: [PPO in HalfCheetah-v2 training.](https://sites.google.com/view/onlineretailing/) 

The result of ‘DDPG’ in ‘HalfCheetah-v2’ which coded by myself:

<img src="https://pic.imgdb.cn/item/5fccfa5e394ac523789bfad5.png" alt="b005ff05fc9dd18c7b7f2f7e37bb73d" style="zoom: 33%;" />

Awesome, it's a really great result for me. I see the robot runs the first.
