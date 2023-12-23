from datetime import datetime
from itertools import count
import math
import random
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model.DQN import DQN
import torch.optim as optim
from model.ReplayMemory import ReplayMemory, Transition
from model.config import *

sys.path.append(".")

from ple.games.flappybird import FlappyBird
from ple import PLE


class FlapMaster(): # my not so naive agent
    def __init__(self, actions):
        self.actions = actions
        # to be able to visualize concious and random actions
        self.c = 0 # concious actions 
        self.r = 0 # random actions 
        self.eps_threshold = 0

    def act_random(self): # returns a random action
        return p.getActionSet()[0] if random.random() > 0.5 else 0; 
        

    def pickAction(self, reward, obs): # returns an action based on epsilon (explore or exploit)
        global steps_done
        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                self.c = self.c + 1
                return policy_net(obs).max(1).indices.view(1, 1)
        else:
            self.r = self.r + 1
            return torch.tensor([[self.act_random()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_tensors = [r.clone().detach() for r in batch.reward]
    reward_tensors =  [tensor.view(1) if tensor.dim() == 0 else tensor for tensor in reward_tensors]
    reward_tuple = tuple(reward_tensors)
    reward_batch = torch.cat(reward_tuple)

    action_batch = torch.clamp(action_batch, 0, 1)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))
    # optimizing model steps
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
episode_durations = []

def plot_durations(show_result=False,plot_to_png = False): # plots durations
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title(f'Epsilon value: {myAgent.eps_threshold:.2f}') # to be able to track randomness
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 140:
        means = durations_t.unfold(0, 140, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(139), means))
        plt.plot(means.numpy())

    plt.text(0.05, 1.07, f'EPS_START: {EPS_START}', transform=plt.gca().transAxes,ha='center', fontsize=8)
    plt.text(0.35, 1.07, f'EPS_END: {EPS_END}', transform=plt.gca().transAxes, ha='center',fontsize=8)
    plt.text(0.65, 1.07, f'EPS_DECAY: {EPS_DECAY}', transform=plt.gca().transAxes, ha='center',fontsize=8)
    plt.text(0.95, 1.07, f'LR: {LR}', transform=plt.gca().transAxes, ha='center',fontsize=8)

    plt.pause(0.001)  # pause a bit so that plots are updated
    
    if plot_to_png:
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M")
        print(timestamp)
        filename = f'solution/plots/{timestamp}.png'
        plt.savefig(filename)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

myAgent = FlapMaster(p.getActionSet())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get number of actions from gym action space
n_actions = len(p.getActionSet())
print("p.actionSet: ",p.getActionSet())
# Get the number of state observations
# p.game.getGameState -> low level 
# p.getScreenRgb -> high level
# observations = p.game.getGameState()
observations = p.game.getGameState()
n_observations = len(observations)

print("n_actions:", n_actions, "n_observations:",n_observations);

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

nb_episodes = 1000
reward = 0.0

episode_min = 999
episode_max = 0
episode_avg = 0
passed_obstacles = 0

for episode in range(nb_episodes):
    myAgent.c = 0
    myAgent.r = 0
    low_level_obs = p.game.getGameState()
    high_level_obs = p.getScreenRGB()

    obs = low_level_obs
    obs_values = np.array(list(obs.values()), dtype=np.float32)
    state = torch.tensor(obs_values, dtype=torch.float32,
                        device=device).unsqueeze(0)
    for frame in count():        
        action = myAgent.pickAction(reward, state)
        reward = p.act(action) 
        if reward > 0:
            passed_obstacles += 1

        if p.game_over():
            # after 10 episodes begin to calculate reward offset
            # additional feedback if it performs better than average
            # additional feedback for passed obstacles  
            if episode > 10:
                # update max, min and average frame values
                episode_min = min(episode_durations)
                episode_max = max(episode_durations)
                episode_avg = sum(episode_durations) / len(episode_durations)
                offset_aggression = -0.5 * -5 # rate * (negative feedback) -> used to calculate ratio of offset to the feedback itself
                reward_offset = max(min(((frame - episode_avg) / ((episode_max - episode_min) / 2)) * offset_aggression, 2.5), -2.5)
                # reward_offset += passed_obstacles
                reward += reward_offset
                passed_obstacles = 0 # resert passed obstacles
                print(reward)    
            next_state = None
        else:
            next_obs = p.game.getGameState()
            next_obs_values = np.array(list(next_obs.values()), dtype=np.float32)
            next_state = torch.tensor(next_obs_values, dtype=torch.float32,
                            device=device).unsqueeze(0)

        # Store the transition in memory
        reward_tensor = torch.tensor(reward).unsqueeze(0) # unsqueeze to make it tensor
        memory.push(state, action, next_state, reward_tensor) 
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if p.game_over():
            episode_durations.append(frame + 1)
            if episode == nb_episodes - 1: 
                plot_durations(plot_to_png=True)
            else:
                plot_durations()
            break

    p.reset_game()
    