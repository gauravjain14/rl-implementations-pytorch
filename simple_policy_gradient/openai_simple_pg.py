import gym
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torchvision.transforms as transforms
from math import log
import numpy as np

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='run simple policy gradient')
parser.add_argument('--env-name',type=str,default='PongNoFrameskip-v4')
parser.add_argument('--num-epochs',type=int,default=10e3)
parser.add_argument('--lr',type=int,default=1e-3)
parser.add_argument('--save-path',type=str,default='model-atari.pt')
parser.add_argument('--log-interval',type=int,default=10) # log every 10 epochs
parser.add_argument('--is_cuda',type=int,default=1)
parser.add_argument('--render',type=int,default=0)
parser.add_argument('--steps-per-epoch',type=int,default=5000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.is_cuda else "cpu")

class PolicyNet(nn.Module):
	def __init__(self,inp_dim,out_dim):
		super(PolicyNet,self).__init__()
		self.fc1 = nn.Linear(inp_dim,256)
		self.fc2 = nn.Linear(256,256)
		self.fc3 = nn.Linear(256,out_dim)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return torch.sigmoid(x) # don't apply any activation?

def discount_rewards(r):
	gamma = 1.0 # don't discount for now 0.99 # discount factor for reward
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		# do we need this? isn't done supposed to say game boundary?
		#if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r


# valid for openai atari only for now
def preprocess(x):
	resize = transforms.Compose([
		transforms.ToPILImage(),
		transforms.CenterCrop((80,80)),
		transforms.Grayscale(num_output_channels=1),
		transforms.ToTensor()])
	return resize(x).to(device)


if __name__=="__main__":
	env = gym.make('PongNoFrameskip-v4')
	obs_dim = env.observation_space.shape[0]
	n_acts = env.action_space.n

	x = preprocess(env.reset())
	x = torch.flatten(x)
	inp_dim = x.shape[0]
	model = PolicyNet(inp_dim,n_acts) 
	model.cuda() # move the model to cuda prior optimizer initialization

	optimizer = optim.Adam(model.parameters(),lr=args.lr)
	episode_number = 0
	running_mean_reward = 0.0

	def train_one_epoch():	
		batch_acts = []
		batch_log_probs = []
		done = False
		ep_rews = []

		curr_state = torch.flatten(preprocess(env.reset()))
		last_state = curr_state

		steps_done = 0
		# continue the trajectory until done
		#while True:
		while True:
			if args.render: env.render()
			diff_state = curr_state - last_state

			# from https://pytorch.org/docs/stable/distributions.html
			probs = model(diff_state)
			# says this is equivalen	t to multinomial
			m = Categorical(probs)
			action = m.sample()

			next_state,reward,done,info = env.step(action)
			next_state = preprocess(next_state)
			next_state = torch.flatten(next_state)

			last_state = curr_state
			curr_state = next_state

			ep_rews.append(reward)
			batch_acts.append(action)
			#batch_loss.append()
			log_probs = F.log_softmax(probs)[action] # seems like a hack
			batch_log_probs.append(-log_probs)
			steps_done += 1

			if done:
				global episode_number
				episode_number += 1
				discounted_epr = discount_rewards(np.vstack(ep_rews))
				t_discounted_epr = torch.from_numpy(discounted_epr).squeeze(1).float().to(device)
				loss = torch.mean(torch.stack(batch_log_probs) * t_discounted_epr)
				
				# backprop
				loss.backward()
					
				curr_state = torch.flatten(preprocess(env.reset()))
				last_state = curr_state
				global running_mean_reward
				running_mean_reward += (1/episode_number)*(sum(ep_rews) - running_mean_reward)

				if episode_number % args.log_interval == 0:
					print("avg reward after %d episodes: %f"%(episode_number,running_mean_reward))
					torch.save(model, args.save_path)

				ep_rews,batch_log_probs = [],[]
				done = False
		
				if steps_done > args.steps_per_epoch:
					break

		# take a single policy gradient update step
		# optimize	
		optimizer.step()
		optimizer.zero_grad() # reset grad every epoch, right?
	
	for i in range(args.num_epochs):	
		train_one_epoch()
		
