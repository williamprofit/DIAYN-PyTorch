import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class SACAgent:
    def __init__(self, p_z, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.n_prior = self.config["n_prior"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(
            self.n_states + self.n_skills,
            self.config["n_actions"],
            self.config["action_bounds"],
            neurons_list=self.config["neurons_list"],
        ).to(self.device)

        self.q_value_network1 = QvalueNetwork(
            n_states=self.n_states + self.n_skills,
            n_actions=self.config["n_actions"],
            n_hidden_filters=self.config["n_hiddens"],
        ).to(self.device)

        self.q_value_network2 = QvalueNetwork(
            n_states=self.n_states + self.n_skills,
            n_actions=self.config["n_actions"],
            n_hidden_filters=self.config["n_hiddens"],
        ).to(self.device)

        self.value_network = ValueNetwork(
            n_states=self.n_states + self.n_skills,
            n_hidden_filters=self.config["n_hiddens"],
        ).to(self.device)

        self.value_target_network = ValueNetwork(
            n_states=self.n_states + self.n_skills,
            n_hidden_filters=self.config["n_hiddens"],
        ).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(
            n_states=self.n_prior if self.n_prior != 0 else self.n_states,
            n_skills=self.n_skills,
            n_hidden_filters=self.config["n_hiddens"],
        ).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(
            self.q_value_network1.parameters(), lr=self.config["lr"]
        )
        self.q_value2_opt = Adam(
            self.q_value_network2.parameters(), lr=self.config["lr"]
        )
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(
            self.discriminator.parameters(), lr=self.config["lr"]
        )

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state, prior=None):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")

        if prior is not None:
            prior = from_numpy(prior).float().to("cpu")

        self.memory.add(state, z, done, action, next_state, prior)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = (
            torch.cat(batch.state)
            .view(self.batch_size, self.n_states + self.n_skills)
            .to(self.device)
        )
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = (
            torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        )
        next_states = (
            torch.cat(batch.next_state)
            .view(self.batch_size, self.n_states + self.n_skills)
            .to(self.device)
        )

        if batch.prior[0] is None:
            prior = None
        else:
            prior = (
                torch.cat(batch.prior)
                .view(self.batch_size, len(batch.prior[0]))
                .to(self.device)
            )

        return states, zs, dones, actions, next_states, prior

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states, prior = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(
                states
            )
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            if prior is None:
                logits = self.discriminator(
                    torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0]
                )
            else:
                logits = self.discriminator(prior)

            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + self.config[
                    "gamma"
                ] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            if prior is None:
                logits = self.discriminator(
                    torch.split(states, [self.n_states, self.n_skills], dim=-1)[0]
                )
            else:
                logits = self.discriminator(prior)

            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(
                self.value_network, self.value_target_network
            )

            return -discriminator_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(
            target_network.parameters(), local_network.parameters()
        ):
            target_param.data.copy_(
                self.config["tau"] * local_param.data
                + (1 - self.config["tau"]) * target_param.data
            )

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
