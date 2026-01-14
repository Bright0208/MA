import torch.optim as optim

from config import gamma, tau, lr_actor, lr_critic, hidden_dim ,device
from config import batch_size ,buffer_size
from net.actor import Actor_deploy
from net.critic import Critic
from net.replay_buffer import ReplayBuffer


class Magent:
    def __init__(self, n_agents, state_dim, action_dim,
                 hidden_dim= hidden_dim,
                 buffer_size = buffer_size,
                 batch_size=batch_size,
                 gamma=gamma,
                 tau=tau,
                 lr_actor=lr_actor,
                 lr_critic=lr_critic):

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # 所有 actor 与 target_actor
        self.actors = []
        self.target_actors = []

        for i in range(n_agents):
            actor = Actor_deploy(state_dim, action_dim, hidden_dim).to(device)
            target_actor = Actor_deploy(state_dim, action_dim, hidden_dim).to(device)
            target_actor.load_state_dict(actor.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)

        # critic
        self.critic = Critic(state_dim, action_dim, n_agents, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, n_agents, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())


        # optimizers
        self.actor_opt = [optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, n_agents)



    def select_action(self, agent_id, state):
        action = self.actors[agent_id](state)
        return action