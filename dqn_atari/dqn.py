import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

import torch.cuda.profiler as profiler
#from apex import pyprof
#pyprof.nvtx.init()

parser = argparse.ArgumentParser(description='dqn with atari')
parser.add_argument('--env-name',type=str,default='Pong-v0')
parser.add_argument('--num-epochs',type=int,default=10e3)
parser.add_argument('--lr',type=int,default=1e-2)
parser.add_argument('--log-interval',type=int,default=10) # log every 10 epochs
parser.add_argument('--batch-size',type=int,default=32)
parser.add_argument('--replay-memory',type=int,default=10000)
parser.add_argument('--is_cuda',type=int,default=1)
parser.add_argument('--render',type=int,default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.is_cuda else "cpu")

env = gym.make(args.env_name).unwrapped

#env = gym.make('CartPole-v0').unwrapped
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
Transition = namedtuple('Transition',
						('state','action','reward','next_state'))

class Log():
	def __init__(self):
		self.log_dict = {}

	def update_log(key, val, op_perform='avg', curr_n=1):
		if op_perform == 'avg':
			if key in self.log_dict:
				self.log_dict[key] += (val - self.log_dict[key])/curr_n
			else:
				self.log_dict[key] = val
		elif op_perform == 'append':
			if key in self.log_dict:
				self.log_dict[key].append(val)
			else:
				self.log_dict[key] = [val]

class ReplayMemory(object): # inheriting from parent object class
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = [] # list of s,a,r,s'
		self.position = 0

	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position+1)%self.capacity

	def sample(self,batch_size):
		return random.sample(self.memory,batch_size)

	def __len__(self):
		return len(self.memory)
	

class CNN(nn.Module):
	def __init__(self,h,w,outputs):
		super(CNN,self).__init__()
		self.conv1 = nn.Conv2d(3,16,kernel_size=5,stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32,kernel_size=5,stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,32,kernel_size=5,stride=2)
		self.bn3 = nn.BatchNorm2d(32)	

		def conv2d_size_out(size,kernel_size=5,stride=2):
			return (size - (kernel_size - 1) - 1) // stride + 1

		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw*convh*32
		self.head = nn.Linear(linear_input_size,outputs)

	def forward(self,x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0),-1))


def get_screen():
	#env.render() ## Don't need it for now
	screen = env.render(mode='rgb_array').transpose((2,0,1))
	screen_ch,screen_height,screen_width = screen.shape
	screen = screen[:,::2,::2] # should downsample
	# hard-coding centre crop for now
	#screen = screen[:,21:101,:]
	screen = np.ascontiguousarray(screen)  # do we need to rescale
	screen = torch.from_numpy(screen)
	resize = T.Compose([
		T.ToPILImage(), # because pytorch tutorial does so
		T.CenterCrop(80),
		T.ToTensor()])
	# resize and add a batch dimension (BCHW)
	return resize(screen).unsqueeze(0).to(device)		


def get_Tensor(inp, crop_dim=80): # expects inp in CxHxW
	t = np.ascontiguousarray(np.transpose(inp,(2,0,1)))
	t = t[:,::2,::2]
	remove_height = t.shape[1] - crop_dim
	remove_width = t.shape[2] - crop_dim
	start_height = int(remove_height/2)
	start_width = int(remove_width/2)
	t = t[:,start_height:start_height+crop_dim,start_width: \
					start_width+crop_dim]
	return torch.tensor(t,dtype=torch.float32).unsqueeze(0).to(device)

def display_screen_snapshot(screen):
	plt.figure()
	plt.imshow(screen.cpu().squeeze().permute(1,2,0).numpy())
	plt.show()


class Policy():
	def __init__(self,epsilon_start,epsilon_end,epsilon_decay,action_space):
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		self.action_space = action_space
	
	def select_action(self, curr_state, model, steps_done):
		# from the paper on Deep RL for Atari
		epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
						math.exp(-1. * steps_done / epsilon_decay)
	
		if random.random() > epsilon_threshold: # greedy
			# get state-action values from the dqn
			with torch.no_grad():
				state_action = model(state)
				return state_action.max(1)[1].view(1,1)
		else:
			return torch.tensor([[np.random.choice(self.action_space.n)]], \
											device=device,dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def train(memory, gamma, batch_size, policy_dqn, train_dqn, optimizer):
	if len(memory) < batch_size:
		return

	transitions = memory.sample(batch_size)
	# we now have a batch of transitions
	# next we have to find Q(s',a',w*)

	# 1. separate into tensor of states, actions
	batch = Transition(*zip(*transitions))
	
	# 2. batch - transition of states, actions, rewards, next_states
	# batch - stores as tuple of tensors. Need to concatenate
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	policy_action_values = policy_dqn(state_batch).gather(1,action_batch) # [batch_size,1]

	# 3, don't consider terminal states for target computation
	non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, \
                       batch.next_state)), device=device, dtype=torch.bool)
	non_terminal_next_states = torch.cat([s for s in batch.next_state \
                                                if s is not None])
	### A bit unclear on where no_grad would go wrong
	target_action_values = torch.zeros(batch_size, device=device)
	#target_action_values[non_terminal_mask] = target_dqn(non_terminal_next_states).max(1)[0].detach()
	target_action_values[non_terminal_mask] = policy_dqn(non_terminal_next_states).max(1)[0].detach()
	
	target_reward = reward_batch +  gamma * target_action_values
	criterion = nn.MSELoss()

	optimizer.zero_grad()
	loss = criterion(policy_action_values, target_reward.unsqueeze(1))
	loss.backward()
	optimizer.step()


if __name__=="__main__":
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	#with torch.autograd.profiler.profile(use_cuda=True) as prof:
		# change to user parseargs
	epsilon_start = 0.9
	epsilon_end = 0.05
	epsilon_decay = 200
	gamma = 0.99
	target_update = 10
	render = args.render
	record_time_step = True 
	record_epoch = False

	reset = env.reset()
	init_screen = get_Tensor(reset)
	# shape - BxCxHxW
	init_shape_h,init_shape_w = init_screen.shape[2],init_screen.shape[3]
	n_actions = env.action_space.n

	policy_dqn = CNN(init_shape_h,init_shape_w,n_actions).to(device)
	target_dqn = CNN(init_shape_h,init_shape_w,n_actions).to(device)
	# We load the policy state dict to update the target state dict
	target_dqn.load_state_dict(policy_dqn.state_dict())
	target_dqn.eval() # sets the model in eval mode - equivalent to train(False)

	optimizer = optim.RMSprop(policy_dqn.parameters(),lr=args.lr)
	memory = ReplayMemory(args.replay_memory)
	action_policy = Policy(epsilon_start,epsilon_end,epsilon_decay,env.action_space)

	steps_done = 0
	tot_rewards = []
	running_mean_reward = 0.0
	
	isRecording = False
	time_Records = []
	temp_Records = []

	for i in range(args.num_epochs):
		#print('episode %d'%i)
		reward_per_episode = 0
		last_screen = get_Tensor(env.reset())
		curr_screen = last_screen
		state = curr_screen - last_screen

		if record_epoch and not record_time_step and not isRecording:
			print("Recording Epoch times")
			isRecording = True
			start.record()
		elif record_epoch and record_time_step:
			print("Can only set recording at one level")

		#profiler.start()
		for t in count():
			if render: env.render()

			if record_time_step and not record_epoch and not isRecording:
				#print("Recording Time per step")
				isRecording = True
				start.record()		

			steps_done += 1
			action = action_policy.select_action(state,policy_dqn,steps_done)
			obs,reward,done,info = env.step(action.item())
			reward_per_episode += reward

			reward = torch.tensor([reward],device=device)

			last_screen = curr_screen
			curr_screen = get_Tensor(obs) #get_screen()
			if not done:
				next_state = curr_screen - last_screen
			else:
				next_state = None

			memory.push(state,action,reward,next_state)
			state = next_state # update state

			train(memory,gamma,args.batch_size,policy_dqn,target_dqn,optimizer)

			if isRecording and record_time_step:
				isRecording = False
				end.record()
				torch.cuda.synchronize()
				temp_Records.append(start.elapsed_time(end))		

			if done:
				episode_durations.append(t+1)
				running_mean_reward += (1/(i+1))*(reward_per_episode - running_mean_reward)
				if i % args.log_interval == 0:
					print("mean reward after episode %d (duration %d steps): %f \n"%( \
								i,t+1,running_mean_reward))
				#plot_durations() # don't plot every iteration. Plot at the end

				if temp_Records:
					time_Records.append(sum(temp_Records)/len(temp_Records))
					temp_Records = []
				break	

		#profiler.stop()
		# update the reward plot
		tot_rewards.append(reward_per_episode)

		if t % target_update == 0:
			start.record()
			target_dqn.load_state_dict(policy_dqn.state_dict())
			end.record()
			torch.cuda.synchronize()
			print(start.elapsed_time(end))

		if isRecording and record_epoch:
			isRecording = False
			end.record()
			torch.cuda.synchronize()
			time_Records.append(start.elapsed_time(end))		

	for i,elem in enumerate(time_Records):
		print("Time taken in epoch %d: %f"%(i,elem)) 
	
	# Don't plot reward every iteration
	#plot_durations()

	#plt.figure(3)
	#plt.clf()
	#plt.plot(tot_rewards)
	#plt.pause(0.001) # essential to update the same plot in real-time?

	#env.render()
	#env.close()
	#plt.ioff()
	#plt.show()
