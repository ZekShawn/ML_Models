# MF20150100-邱祥智-PG&NPG

## There's no  good results, but I have some thinkings.

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

### What is Natural Policy Gradient?

We may should to sample a large set of trajectories if we train the model by the vanilla PG because of the large variance taken by the reward. Surely we can decrease the variance according to make the reward cut the **baseline**.  The baseline can be the average value of reward of an episode. Like the follow paradigm:

$\triangledown_{\theta} J_{\theta} = \frac{1}{N} \sum_{i=1}^{N} (\sum_{t=1}^{T} \triangledown_{\theta} log \pi_{\theta}(a_t^i|s_t^i)(\sum_{t'=t}^{T} \gamma^{t'-t} r(s_t^i,a_t^i)-b))$ 	(4)

$b = \frac{1}{T} \sum_{t=1}^T r(s_t,a_t)$ 	(5)

At the before, we use the $\triangledown_{\theta} J_{\theta}$ to update the $\theta$ . There's no doubt it's a feasible method but not efficient. The optimizer need to find the direction and distance of the new parameters. We view the $\triangledown_{\theta} J_{\theta}$ as the distance and the update direction. But for the $\theta$ it self, it's a distribution parameters, not so involved with $\triangledown_{\theta} J_{\theta}$ . We need to know how to calculate the distance. Maybe the KL divergence can help us:

$D_{KL}=\int p(x)log \frac{p(x)}{q(x)}d(x) = \frac{1}{2}d^T \triangledown_{\theta'}^2D_{KL}(\pi_{\theta}||\pi_{\theta'})|_{\theta'=\theta}d = \frac{1}{2}d^TF(\theta)d$ 	(6)

And our target is:

$\theta \leftarrow \theta + d^*$	(7)

We need to find the $d^*$ , by making the $J(\theta+d)$ :

$d^* = \mathop{argmax}\limits_{d} J(\theta+d), s.t.D_{KL}(\pi_{\theta}||\pi_{\theta+d}\leq \epsilon)$ 	(8)

Finally we need to solve the paradigm 8, the results is:

$\theta_{k+1}=\theta_k + \sqrt{\frac{2\epsilon}{g_k^TH_kg_k}}H^{-1}_kg_k$ 	(9)

$H_K$: the Hessian matrix of $\theta$ at the point K

$g_K$: the Policy gradient of $\theta$ at the point K

### How about the code?

**Problem 1: Maze Navigation, we need to train the robot to get the right route from start to goal.**

<img src="https://pic.imgdb.cn/item/5fb3f045b18d62711324ebc8.png" alt="image-20201117234605775" style="zoom: 50%;" />

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

The result is great than I expected, that's in below(PG before):

<img src="https://pic.imgdb.cn/item/5fb3f2b4b18d62711325ff57.jpg" alt="Lr-0.001-Gamma-0.99-Reward-Loss" style="zoom: 25%;" />

After adjust the environments' parameters:

Terminal field and reward: $[0.4, 0.5]^2$,  20

Start point and reward: [0, 0], 0

Reward apart from terminal field of the map: -1

Reward when out of the border: -100

And the result of PG is:

<img src="https://pic1.imgdb.cn/item/5fbf951715e771908417692d.jpg" alt="PGreward-loss-lr-0.001-gamma-0.99-sample_nums-3-t_horizens-20" style="zoom:18%;" />

And the result of NPG is:

<img src="https://pic1.imgdb.cn/item/5fbf96c115e771908417d381.jpg" alt="NPGreward-loss-lr-0.001-gamma-0.99-sample_nums-3-t_horizens-20" style="zoom:18%;" />

Seems badly? Yes, there are some special code difficulties in calculating the matrix, **I know how to get the grad and update the neural network, nothing know about how to do it in hand. And the grad Hessian matrix is more difficult to calculate than the previous.**

**Notions:** If you want to get the same result of PG in Maze Navigation, you may need to set the reward of out of the border larger than the reward of arriving terminal yield.

**Problem 2: We need to make the robot walk forward.**

I don't know the full informations about this environment. But I know the max location is infinite and min is negative infinite. And the max action step is 1, the min action step value is -1. And the observation of environment are 17 axis which action are 6.

<img src="https://pic1.imgdb.cn/item/5fbf98fe15e77190841883e0.png" alt="image-20201126200102958" style="zoom:25%;" />

So, just different input dimensions and output dimensions, we get the bad results(PG)...

<img src="https://pic1.imgdb.cn/item/5fbf9ac515e771908418fc19.jpg" alt="PGreward-loss-lr-0.001-gamma-0.99-sample_nums-4-t_horizens-50" style="zoom:18%;" />

And the result of NPG:

<img src="https://pic1.imgdb.cn/item/5fbfaf9115e77190841f4c06.jpg" alt="NPGreward-loss-lr-0.001-gamma-0.99-sample_nums-4-t_horizens-50" style="zoom:18%;" />

If the result of PG is partly incorrectly, I can be surely say the result of NPG is unbelievable. So, there's something wrong in updating model. Maybe the KL divergence matrix is wrongly calculated.