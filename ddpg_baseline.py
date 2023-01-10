'''
DDPG
'''

# utils
import argparse
import time
import math
import random
import matplotlib.pyplot as plt

# Simulate environments
import gym
import pybullet_envs

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

TIME_STAMPS = time.strftime("%Y%m%d-%H%M%S", time.localtime())

'''Parse: control training or testing'''
parser = argparse.ArgumentParser(description='Train DDPG on OpenAI Gym')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1,  help='random seed')
'''Parse: Trainenv'''
'''Choose from ['Pendulum-v1', 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'MountainCarContinuous-v0', 'HopperBulletEnv-v0'][4]'''
parser.add_argument('--gym_id', type=str, default='Pendulum-v0', help='OpenAI Gym environment ID')

'''Parse: hyperparameters'''
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--tau', type=float, default=5e-3, help='soft update')
parser.add_argument('--target_update_delay', type=int, default=2, help='Target network update delay')
parser.add_argument('--buffer_size', type=int, default=1e6, help='replay buffer size')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
parser.add_argument('--episodes_limit', type=int, default=2000, help='episodes limit')
parser.add_argument('--record_steps_limit', type=int, default=100, help='record steps limit')
parser.add_argument('--warm_up_steps_limit', type=int, default=0, help='warm_up steps to accumulate replay buffer')
parser.add_argument('--noise_scale', type=float, default=0.0, help='noise')

args = parser.parse_args()


'''Initialize the gym game environment'''
experiment_name = f"{args.gym_id}"
args.path = experiment_name
writer = SummaryWriter(f"tensorboard/{experiment_name}/{TIME_STAMPS}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))


'''Experience Replay Buffer'''
class ReplayBuffer:
    def __init__(self, capacity):
        """Initialize a ReplayBuffer object."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        '''Use Ring Buffer to maximize the use of memory'''
        self.position = int((self.position + 1) % self.capacity)  
    
    def sample(self, batch_size):
        '''Randomly choose a batch of experiences from the replay buffer'''
        batch = random.sample(self.buffer, batch_size)
        '''Element stack'''
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

'''Actor Network'''
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        '''Actor Network is consisted of three fully connected layers'''
        self.action_dim=output_dim
        '''output dim = dim of action'''
        self.linear1 = nn.Linear(input_dim, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state):
        activation=F.relu
        x = activation(self.linear1(state)) 
        x = activation(self.linear2(x))
        x = self.linear3(x) 
        '''For simplicity, no activation and range formalization for output layer'''

        return x

    def select_action(self, state, noise_scale=0.1):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu, used in warm_up stage
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        normal = Normal(0, 1)
        action = self.forward(state)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action+=noise
        return action.detach().cpu().numpy()[0]

    def sample_action(self, action_range=1.0):
        '''select random actions, return .cpu, used in training stage'''
        normal = Normal(0, 1)
        random_action=action_range*normal.sample( (self.action_dim,) )

        return random_action.cpu().numpy()


    def evaluate_action(self, state, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action = self.forward(state)
        # action = torch.tanh(action)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action+=noise
        return action

'''Critic Network'''
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        '''Critic Network is consisted of three fully connected layers'''
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        '''Use relu activation function'''
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

'''DDPG Algorithm framework'''
class DDPG():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim):
        '''Initialize the DDPG algorithm'''
        '''Initialize the replay buffer, Q network, target Q network, policy network, target policy network'''
        self.replay_buffer = replay_buffer
        self.qnet = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.target_qnet = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)

        '''Print the basic parameters'''
        print('Q network: ', self.qnet)
        print('Policy network: ', self.policy_net)

        '''Initialize the target Q network and target policy network'''
        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)

        '''Define the loss function and optimizer'''
        self.q_criterion = nn.MSELoss()
        q_lr=args.lr
        policy_lr = args.lr
        print("Current lr is",q_lr, policy_lr)
        self.update_cnt=0

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_soft_update(self, net, target_net, soft_tau):
        '''Soft update the target network'''
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_( 
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def update(self, batch_size, gamma=0.99, soft_tau=1e-2, target_update_delay=2,):
        '''Main function to update the network'''
        self.update_cnt+=1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predict_q = self.qnet(state, action) # for q 
        new_next_action = self.target_policy_net.evaluate_action(next_state)  # for q
        new_action = self.policy_net.evaluate_action(state) # for policy
        target_q = reward+(1-done)*gamma*self.target_qnet(next_state, new_next_action)  # for q

        '''Train critic network'''
        q_loss = self.q_criterion(predict_q, target_q.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        '''Train actor network'''
        policy_loss = -self.qnet(state, new_action).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # print("[debug]hi")
        self.policy_optimizer.step()

            
        '''Delay update the target network'''
        if self.update_cnt%target_update_delay==0:
            self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    '''Basic function for saving and loading model'''
    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path+'_q')
        torch.save(self.target_qnet.state_dict(), path+'_target_q')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'_q'))
        self.target_qnet.load_state_dict(torch.load(path+'_target_q'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.qnet.eval()
        self.target_qnet.eval()
        self.policy_net.eval()

'''Basic plotting network'''
def plot(data, data_type):
    plt.figure(figsize=(20,5))
    plt.plot(data)
    plt.savefig('./pic/ddpg_{}_{}.pdf'.format(TIME_STAMPS, data_type))
    # plt.show()
    plt.clf()

'''Device Inializing'''
GPU = True
device_idx = 2
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("[Debug]: Using Device:",device)


if __name__ == '__main__':
    env = gym.make(args.gym_id)
    print('[Debug]Using Env: ', args.gym_id, env.action_space, env.observation_space)

    '''Set random seed'''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 512
    explore_steps = 0  # for random exploration

    replay_buffer_size=1e6
    '''Initialize replay buffer'''
    replay_buffer = ReplayBuffer(replay_buffer_size)
    model_path='./model/ddpg'
    torch.autograd.set_detect_anomaly(True)
    alg = DDPG(replay_buffer, state_dim, action_dim, args.hidden_dim)

    if args.train:
        '''Training mode'''
        warm_up_step   = 0
        rewards=[]
        actor_loss_list=[]
        critic_loss_list=[]

        for episode in range (args.episodes_limit):
            q_loss_list=[]
            policy_loss_list=[]
            state = env.reset()
            episode_reward = 0

            for step in range(args.record_steps_limit):
                '''Warm up steps for policy nets'''
                if warm_up_step > args.warm_up_steps_limit:
                    action = alg.policy_net.select_action(state)
                else:
                    action = alg.policy_net.sample_action(action_range=1.)
                next_state, reward, done, _ = env.step(action)
                '''Push the data into replay buffer'''
                replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                warm_up_step += 1
                
                if len(replay_buffer) > args.batch_size:
                    '''Update the network'''
                    q_loss, policy_loss = alg.update(args.batch_size, args.gamma, args.tau, args.target_update_delay)
                    q_loss_list.append(q_loss)
                    policy_loss_list.append(policy_loss)
                
                if done:
                    break

            '''Basic plotting and saving model every 20 episodes'''
            if episode % 20 == 0:
                plot(rewards, 'reward')
                plot(actor_loss_list, 'actor_loss')
                plot(critic_loss_list, 'critic_loss')
                alg.save_model(model_path)
            
            print('Eps: ', episode, '| Reward: ', episode_reward, '| Loss: ', np.average(q_loss_list), np.average(policy_loss_list))
            
            '''Save the rewards and losses for plotting'''
            rewards.append(episode_reward)
            actor_loss_list.append(np.average(policy_loss_list))
            critic_loss_list.append(np.average(q_loss_list))
            '''Save the rewards and losses for tensorboard'''
            writer.add_scalar("rewards", np.average(episode_reward), episode)
            writer.add_scalar("losses/actor_loss", np.average(policy_loss_list), episode)
            writer.add_scalar("losses/critic_loss", np.average(q_loss_list), episode)


    if args.test:
        '''Testing mode for ranking'''
        test_episodes = 10
        max_steps=100
        alg.load_model(model_path)

        for episode in range (test_episodes):
            q_loss_list=[]
            policy_loss_list=[]
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = alg.policy_net.select_action(state, noise_scale=args.noise_scale)  # no noise for testing
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
                
                
                if done:
                    break
 
            print('Eps: ', episode, '| Reward: ', episode_reward)

env.close()
writer.close()