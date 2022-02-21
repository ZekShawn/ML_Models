# default settings

class Args:
    def __init__(self) -> None:
        super(Args,self).__init__()
        self.env_chosse = 'MazeNavigation'      # default: MazeNavigation
        self.algorithm = 'PG'                   # default: PG
        self.env = None                         # default: None

        self.lr = 0.001
        self.gamma = 0.99

        self.input_dims = 2
        self.h1_dims = 16
        self.h2_dims = 16
        self.actions = 2
        self.action_lock = 0.1
        self.log_std_min = -20
        self.log_std_max = 2

        self.epoch_nums = 2500
        self.sample_nums = 3
        self.t_horizens = 20

        self.path = None                        # default: None

        self.critic_input_dims = 3
        self.critic_values = 1


args = Args()
args.env_chosse = 'HalfCheetah-v2'          # optional: HalfCheetah-v2 , MazeNavigation
args.algorithm = 'NPG'                       # optional: PG, NPG, AC


if args.env_chosse == 'MazeNavigation':
    from other.Maze import ENV
    args.env = ENV()
else:
    import gym
    args.env = gym.make(args.env_chosse)
    args.lr = 0.001
    args.gamma = 0.99

    args.input_dims = 17
    args.h1_dims = 256
    args.h2_dims = 256
    args.actions = 6
    args.action_lock = 1
    args.log_std_min = -20
    args.log_std_max = 2

    args.epoch_nums = 500
    args.sample_nums = 4
    args.t_horizens = 50

    args.critic_input_dims = 18
    args.critic_values = 1


if args.algorithm != 'AC':
    if args.algorithm == 'PG':
        from algorithms.PG import run
    else:
        from algorithms.NPG import run
    args.path = 'data\\' + args.env_chosse
    args.path += '\\'
    args.path += args.algorithm
    run(args.env_chosse, args.lr, args.gamma, args.input_dims, args.h1_dims, args.h2_dims, args.actions, args.action_lock, 
        args.log_std_min, args.log_std_max, args.epoch_nums, args.sample_nums, args.t_horizens, args.env, args.path)
else:
    from algorithms.AC import run
    args.input_dims += 1
    args.path = 'data\\' + args.env_chosse
    args.path += '\\'
    args.path += args.algorithm
    run(args.env_chosse, args.lr, args.gamma, args.input_dims, args.critic_input_dims, args.h1_dims, args.h2_dims, 
        args.actions, args.critic_values, args.action_lock, args.log_std_min, args.log_std_max, args.epoch_nums, 
        args.sample_nums, args.t_horizens, args.env, args.path)