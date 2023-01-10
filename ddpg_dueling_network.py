'''
DDPG-New-Network
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

# Parse: control training or testing
parser = argparse.ArgumentParser(description='Train DDPG on OpenAI Gym')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

# Parse: Trainenv
## Choose from ENV = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'MountainCarContinuous-v0', 'HopperBulletEnv-v0'][4]
parser.add_argument('--gym_id', type=str, default='Pendulum-v0', help='OpenAI Gym environment ID')

# Parse: hyperparameters
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--tau1', type=float, default=5e-3, help='soft update')
parser.add_argument('--tau2', type=float, default=4e-3, help='soft update tau2')
parser.add_argument('--target_update_delay', type=int, default=3, help='Target network update delay')
parser.add_argument('--buffer_size', type=int, default=1e6, help='replay buffer size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension')
parser.add_argument('--episodes_limit', type=int, default=1000, help='episodes limit')
parser.add_argument('--record_steps_limit', type=int, default=100, help='record steps limit')
parser.add_argument('--warm_up_steps_limit', type=int, default=0, help='warm_up steps to accumulate replay buffer')
parser.add_argument('--noise_scale', type=float, default=0.0, help='noise')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()


# Initialize the gym game environment
experiment_name = f"{args.gym_id}"
args.path = experiment_name
writer = SummaryWriter(f"tensorboard/Network/{experiment_name}/{TIME_STAMPS}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.action_dim=output_dim
        
        self.linear1 = nn.Linear(input_dim, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, output_dim) # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state):
        activation=F.relu
        x = activation(self.linear1(state)) 
        x = activation(self.linear2(x))
        # x = F.tanh(self.linear3(x)).clone() # need clone to prevent in-place operation (which cause gradients not be drived)
        x = self.linear3(x) # for simplicity, no restriction on action range

        return x

    def select_action(self, state, noise_scale=1.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # state dim: (N, dim of state)
        normal = Normal(0, 1)
        action = self.forward(state)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action+=noise
        return action.detach().cpu().numpy()[0]

    def sample_action(self, action_range=1.):
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


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DDPG():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim):
        self.replay_buffer = replay_buffer
        self.qnet = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.target_qnet_1 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.target_qnet_2 = QNetwork(state_dim+action_dim, hidden_dim).to(device)
        self.policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net_1 = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net_2 = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)

        print('Q network: ', self.qnet)
        print('Policy network: ', self.policy_net)

        for target_param, param in zip(self.target_qnet_1.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_qnet_2.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)

        self.q_criterion = nn.MSELoss()
        q_lr=args.lr
        policy_lr = args.lr
        print("Current lr is",q_lr, policy_lr)
        self.update_cnt=0

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def update(self, batch_size, gamma=0.99, soft_tau_1=1e-2, soft_tau_2=1e-2, target_update_delay=3):
        self.update_cnt+=1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predict_q = self.qnet(state, action) # for q 
        new_next_action_1 = self.target_policy_net_1.evaluate_action(next_state)  # for q
        new_next_action_2 = self.target_policy_net_2.evaluate_action(next_state)  # for q
        new_next_action = (new_next_action_1+new_next_action_2)/2
        # new_next_action = min(new_next_action_1,new_next_action_2)


        new_action = self.policy_net.evaluate_action(state) # for policy
        # predict_new_q = self.qnet(state, new_action) # for policy
        target_q_1 = reward+(1-done)*gamma*self.target_qnet_1(next_state, new_next_action)  # for q
        target_q_2 = reward+(1-done)*gamma*self.target_qnet_2(next_state, new_next_action)  # for q
        # target_q = (target_q_1+target_q_2)/2
        target_q = min(target_q_1, target_q_2)

        # train qnet
        q_loss = self.q_criterion(predict_q, target_q.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # train policy_net
        # policy_loss = -torch.mean(predict_new_q)
        policy_loss = -self.qnet(state, new_action).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # print("[debug]hi")
        self.policy_optimizer.step()

            
        # update the target_qnet
        if self.update_cnt%target_update_delay==0:
            self.target_qnet_1=self.target_soft_update(self.qnet, self.target_qnet_1, soft_tau_1)
            self.target_qnet_2=self.target_soft_update(self.qnet, self.target_qnet_2, soft_tau_2)
            self.target_policy_net_1=self.target_soft_update(self.policy_net, self.target_policy_net_1, soft_tau_2)
            self.target_policy_net_2=self.target_soft_update(self.policy_net, self.target_policy_net_2, soft_tau_2)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path+'_q')
        torch.save(self.target_qnet_1.state_dict(), path+'_target_q_1')
        torch.save(self.target_qnet_2.state_dict(), path+'_target_q_2')
        torch.save(self.policy_net.state_dict(), path+'_policy')
        torch.save(self.target_policy_net_1.state_dict(), path+'_target_policy_1')
        torch.save(self.target_policy_net_2.state_dict(), path+'_target_policy_2')
        

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'_q'))
        self.target_qnet_1.load_state_dict(torch.load(path+'_target_q_1'))
        self.target_qnet_2.load_state_dict(torch.load(path+'_target_q_2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.target_policy_net_1.load_state_dict(torch.load(path+'_target_policy_1'))
        self.target_policy_net_2.load_state_dict(torch.load(path+'_target_policy_2'))
        self.qnet.eval()
        self.target_qnet_1.eval()
        self.target_qnet_2.eval()
        self.policy_net.eval()
        self.target_policy_net_1.eval()
        self.target_policy_net_2.eval()

def plot(data, data_type):
    plt.figure(figsize=(20,5))
    plt.plot(data)
    plt.savefig('./pic/network/ddpg_{}_{}.pdf'.format(TIME_STAMPS, data_type))
    # plt.show()
    plt.clf()

# Initialize the devies
GPU = True
device_idx = 2
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("[Debug]: Using Device:",device)


if __name__ == '__main__':
    ENV = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'MountainCarContinuous-v0', 'HopperBulletEnv-v0'][4]
    env = gym.make(args.gym_id)
    print('[Debug]Using Env: ', args.gym_id, env.action_space, env.observation_space)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 512
    explore_steps = 0  # for random exploration
    # batch_size = 64

    replay_buffer_size=1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)
    model_path='./model/network/ddpg'
    torch.autograd.set_detect_anomaly(True)
    alg = DDPG(replay_buffer, state_dim, action_dim, args.hidden_dim)

    if args.train:

        # hyper-parameters
        # max_episodes  = 1000
        # max_steps   = 100
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
                if warm_up_step > args.warm_up_steps_limit:
                    action = alg.policy_net.select_action(state)
                else:
                    action = alg.policy_net.sample_action(action_range=1.)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                warm_up_step += 1
                
                if len(replay_buffer) > args.batch_size:
                    q_loss, policy_loss = alg.update(args.batch_size, args.gamma, args.tau1 ,args.tau2, args.target_update_delay)
                    q_loss_list.append(q_loss)
                    policy_loss_list.append(policy_loss)
                
                if done:
                    break
            if episode % 20 == 0:
                plot(rewards, 'reward')
                plot(actor_loss_list, 'actor_loss')
                plot(critic_loss_list, 'critic_loss')
                alg.save_model(model_path)

            print('Eps: ', episode, '| Reward: ', episode_reward, '| Loss: ', np.average(q_loss_list), np.average(policy_loss_list))
            
            rewards.append(episode_reward)
            actor_loss_list.append(np.average(policy_loss_list))
            critic_loss_list.append(np.average(q_loss_list))
            writer.add_scalar("rewards", np.average(episode_reward), episode)
            writer.add_scalar("losses/actor_loss", np.average(policy_loss_list), episode)
            writer.add_scalar("losses/critic_loss", np.average(q_loss_list), episode)


    if args.test:
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