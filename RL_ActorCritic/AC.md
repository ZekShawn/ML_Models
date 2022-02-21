# MF20150100-邱祥智-AC

## Let's come back the Policy Gradient.

### What is Policy Gradient？

It's just like this, it's core of PG. This is how we update the $\theta$ of the network. 

$\theta = \theta + \alpha \triangledown_{\theta}J_{\theta}$	(1)

$\triangledown_{\theta} J_{\theta} = \frac{1}{N} \sum_{i=1}^{N} (\sum_{t=1}^{T} \triangledown_{\theta} log \pi_{\theta}(a_t^i|s_t^i)(\sum_{t'=1}^{T} \gamma^{t'-1} r(s_t^i,a_t^i)))$	(2)

So, how I understand these parameters in the paradigm, like below:

$\theta$ : which we don't need to focus on how to update it in the code, just need to know those are parameters in the neural network.

$\triangledown_{\theta}J_{\theta}$ : means to get the grad of $J_\theta$ according to parameter $\theta$.

i & N: i is the current episode and N is the number of episodes.

t & T: t is the current period and the T is the periods of an episode.

$a_t^i$: the action took in period t and episode i.

$s_t^i$ : the state in period t and episode i.

 $\triangledown_{\theta} log \pi_{\theta}(a_t^i|s_t^i)$ : the probability of $a_t^i$ is took in $s_t^i$, and get it's logarithm.

$\gamma$ : the discount of next period reward

$\gamma^{t'-1} r(s_t^i,a_t^i)$ : the future reward with discount.

$\sum_{t'=1}^{T} \gamma^{t'-1} r(s_t^i,a_t^i))$ : the sum of all discounted rewards of an episode.

Actually, it works by MC sampling method to make the probability of the great route be the max. Like this, make the neural network more likely to output the right continuous actions in the specific period.

![](https://upload.wikimedia.org/wikipedia/commons/7/74/Normal_Distribution_PDF.svg)

And, to make it more correctly with the reality, we just need the backward rewards to evaluate and update the neural network. The final paradigm like this:

$\triangledown_{\theta} J_{\theta} = \frac{1}{N} \sum_{i=1}^{N} (\sum_{t=1}^{T} \triangledown_{\theta} log \pi_{\theta}(a_t^i|s_t^i)(\sum_{t'=t}^{T} \gamma^{t'-t} r(s_t^i,a_t^i)))$	(3)

### What is Actor-Critic?

We use the neural network to fit the value function to evaluate the advantage of states. Like this:

$A^{\pi}(s_i,a_i)=r(s_i,a_a)+\gamma V^{\pi}_{\phi}(s_i')-V_{\phi}^{\pi}(s_i)$ 	(4)

And the update paradigm is like this:

$\triangledown_{\theta}J(\theta) \approx \sum_i \triangledown_{\theta}log\pi_{\theta}(a_i|s_i)A^{\pi}(s_i,a_i)$ 	(5)

It seems like Q-Learning.

### Ho about the code?

**Problem 1: Maze Navigation, we need to train the robot to get the right route from start to goal.**

<img src="https://pic1.imgdb.cn/item/5fbfa29415e77190841b34de.png" alt="image-20201126204156433" style="zoom:50%;" />

So, there are some parameters be definite in the below:

```python
class Args:
    def __init__(self) -> None:
        super(Args,self).__init__()
        self.env_chosse = 'MazeNavigation'      # default: MazeNavigation
        self.algorithm = 'PG'                   # default: PG
        self.env = None                         # default: None
        self.lr = 0.001							# learning rate
        self.gamma = 0.99						# discount rate of future reward
        self.input_dims = 2						# input dimensions of actor nerual network 
        self.h1_dims = 16						# hidden dimensions
        self.h2_dims = 16						# hidden dimensions
        self.actions = 2						# action dimensions, x_axis changes
        self.action_lock = 0.1					# max action value
        self.log_std_min = -20					# restrict the sigma
        self.log_std_max = 2					# restrict the sigma
        self.epoch_nums = 5000					# trainning rounds
        self.sample_nums = 3					# sample nums, MC method
        self.t_horizens = 20					# time horizens
        self.path = None                        # default: None
        self.critic_input_dims = 3				# critic input dimensions
        self.critic_values = 1					# critic critics the actions
```

And use the `nn.Linear` to create the full network layer, `torch.distributions.Normal` is the shape to fit the actions distribution. And its out is be limited to [-0.1, 0.1]. If the robot's location is in the first quadrant (the refer zero point is start, and the goal is [0.4-0.5, 0.4-0.5]), robot get the 0 reward, and other location get -1 reward.

After adjust the environments' parameters:

Terminal field and reward: $[0.4, 0.5]^2$,  20

Start point and reward: [0, 0], 0

Reward apart from terminal field of the map: -1

Reward when out of the border: -5

And the result of AC is:

<img src="https://pic1.imgdb.cn/item/5fbfa2cb15e77190841b4185.jpg" alt="ACreward-loss-lr-0.001-gamma-0.99-sample_nums-3-t_horizens-20" style="zoom:18%;" />

Seems great. Actually, the more simple the more efficient. And it can be a great explorer and learner. 

**Notions:** If you want to get the same great result, maybe you need to set the reward of out of border lower than the reward of arriving to terminal field.

**Problem 2: We need to make the robot walk forward.**

I don't know the full informations about this environment. But I know the max location is infinite and min is negative infinite. And the max action step is 1, the min action step value is -1. And the observation of environment are 17 axis which action are 6.

<img src="https://pic1.imgdb.cn/item/5fbf98fe15e77190841883e0.png" alt="image-20201126200102958" style="zoom:25%;" />

So, just different input dimensions and output dimensions, we get the bad results...Maybe we need more time horizons and large batch size. But unfortunately there's only CPU on laptop.

<img src="https://pic1.imgdb.cn/item/5fbfadb515e77190841e9fb5.jpg" alt="ACreward-loss-lr-0.001-gamma-0.99-sample_nums-4-t_horizens-500" style="zoom:18%;" />

OK, that's all, I'm being more negative in the record, and more concentrate on the reshow of the algorithm...I'm sorry for bad code style and bad result of the second robot.