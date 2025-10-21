import copy
import logging
from typing import Dict

import numpy as np
import torch
import constants

import matplotlib

matplotlib.use("Agg")

from sklearn.metrics import explained_variance_score
from utils.optimization_helpers import polyak_averaging
from utils.utils import clip_gradients

logger = logging.getLogger(__name__)

# Disable autograd anomaly detection for performance
torch.autograd.set_detect_anomaly(False)


class CriticLearner:
    def __init__(self, params: Dict, writer, critic):
        self.params = params
        self.critic = critic
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.target_critic = copy.deepcopy(critic)
        self.writer = writer
        self.target_critic.to(constants.DEVICE)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.critic.to(constants.DEVICE)
        self.n_actions = 3 * 3 * 3
        self.position_counter = np.zeros((11, 11))
        self.position_q_sum = np.zeros((11, 11))
        self.spacing = params["experiment"]["constraints"]["spacing"]
        self.reward_normalization = params["experiment"]["missions"][
            "reward_normalization"
        ]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.batch_size = params["networks"]["batch_size"]
        self.gamma = params["networks"]["gamma"]
        self.lam = params["networks"]["lambda"]
        self.copy_rate = params["networks"]["copy_rate"]
        self.lr = params["networks"]["critic"]["learning_rate"]
        self.momentum = params["networks"]["critic"]["momentum"]
        self.gradient_norm = params["networks"]["critic"]["gradient_norm"]
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        # AMP setup
        self.use_amp = (constants.DEVICE.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.mse_loss = torch.nn.MSELoss()
        self.huber_loss = torch.nn.HuberLoss(reduction="none", delta=0.1)
        self.smooth_l1loss = torch.nn.SmoothL1Loss(reduction="none")
        self.optimizer.zero_grad()
        self.critic_update_mode = params["networks"]["critic"]["update_mode"]
        self.critic_synchronization = params["networks"]["critic"]["synchronization"]
        self.target_update_mode = params["networks"]["critic"]["target_update_mode"]
        self.tau = params["networks"]["critic"]["tau"]

    def learn(self, num_train_step: int, batches, data_pass):
        self.update_target_network(num_train_step, data_pass)
        all_q_values = []
        td_errors = []
        td_targets = []
        chosen_q_values = []
        discounted_returns = []
        critic_log_prob_chosen = []
        for batch in batches:
            batch_states = []
            batch_actions = []
            batch_td_targets = []
            batch_discounted_returns = []
            for batch_idx in batch:
                batch_states.append(batch_idx.state)
                batch_actions.append(batch_idx.action)
                batch_td_targets.append(batch_idx.td_target)
                batch_discounted_returns.append(batch_idx.discounted_return)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                batch_q_values, _ = self.critic.forward(
                    torch.stack(batch_states).squeeze().to(constants.DEVICE)
                )
            batch_q_chosen = torch.gather(
                batch_q_values, dim=1, index=torch.stack(batch_actions)
            )
            td_error = (
                torch.square(
                    batch_q_chosen
                    - torch.stack(batch_td_targets).to(constants.DEVICE).detach()
                )
                .squeeze()
                .mean()
            )
            td_errors.append(td_error)

            td_targets.append(torch.stack(batch_td_targets))
            discounted_returns.append(torch.stack(batch_discounted_returns))
            chosen_q_values.append(batch_q_chosen)

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(td_error).backward()
                # clip_gradients(self.critic, self.gradient_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                td_error.backward()
                # clip_gradients(self.critic, self.gradient_norm)
                self.optimizer.step()

            with torch.no_grad():
                batch_new_q_values, batch_log_probs = self.critic.forward(
                    torch.stack(batch_states).squeeze().to(constants.DEVICE)
                )
            all_q_values.append(batch_new_q_values)
            batch_log_prob_chosen = torch.gather(
                batch_log_probs.squeeze(), dim=1, index=torch.stack(batch_actions)
            )
            critic_log_prob_chosen.append(batch_log_prob_chosen)

        conv1_grad_norm = 0
        conv2_grad_norm = 0
        conv3_grad_norm = 0
        fc1_grad_norm = 0
        fc2_grad_norm = 0
        fc3_grad_norm = 0
        conv1_count = 0
        conv2_count = 0
        conv3_count = 0
        fc1_count = 0
        fc2_count = 0
        fc3_count = 0
        for params in self.critic.conv1.parameters():
            if params.grad is not None:
                conv1_grad_norm += params.grad.data.norm(1).item()
                conv1_count += 1
        for params in self.critic.conv2.parameters():
            if params.grad is not None:
                conv2_grad_norm += params.grad.data.norm(1).item()
                conv2_count += 1
        for params in self.critic.conv3.parameters():
            if params.grad is not None:
                conv3_grad_norm += params.grad.data.norm(1).item()
                conv3_count += 1
        for params in self.critic.fc1.parameters():
            if params.grad is not None:
                fc1_grad_norm += params.grad.data.norm(1).item()
                fc1_count += 1
        for params in self.critic.fc2.parameters():
            if params.grad is not None:
                fc2_grad_norm += params.grad.data.norm(1).item()
                fc2_count += 1
        for params in self.critic.fc3.parameters():
            if params.grad is not None:
                fc3_grad_norm += params.grad.data.norm(1).item()
                fc3_count += 1

        # Pre-compute stacked tensors with consistent shapes
        dr = torch.stack(discounted_returns)                 # [B, N]
        tt = torch.stack(td_targets)                         # [B, N]
        cq = torch.stack(chosen_q_values)                    # [B, N, 1]
        all_q_values_stacked = torch.stack(all_q_values)     # [B, N, A]

        # Numpy arrays for sklearn metrics (flatten to 1D)
        dr_np = dr.detach().cpu().reshape(-1).numpy()
        tt_np = tt.detach().cpu().reshape(-1).numpy()

        # Align shapes for difference: broadcast [B,N] -> [B,N,1]
        # Keep on same device; only move to CPU when converting to numpy
        diff = torch.abs(dr.unsqueeze(-1) - cq)              # [B, N, 1]

        return (
            all_q_values,
            [
                torch.mean(torch.stack(td_errors)).detach().cpu().item(),
                torch.mean(tt).detach().cpu().item(),
                torch.std(tt).detach().cpu().item(),
                torch.mean(cq).detach().cpu().item(),
                torch.mean(all_q_values_stacked).detach().cpu().item(),
                torch.min(all_q_values_stacked).detach().cpu().item(),
                torch.std(all_q_values_stacked).detach().cpu().item(),
                float(explained_variance_score(dr_np, tt_np)),
                torch.mean(dr).detach().cpu().item(),
                torch.std(dr).detach().cpu().item(),
                float(torch.mean(diff).detach().cpu().numpy()),
                float(torch.std(diff).detach().cpu().numpy()),
                torch.mean(torch.stack(critic_log_prob_chosen)).detach().cpu().item(),
                [
                    conv1_grad_norm / conv1_count,
                    conv2_grad_norm / conv2_count,
                    conv3_grad_norm / conv3_count,
                    fc1_grad_norm / fc1_count,
                    0,
                    fc3_grad_norm / fc3_count,
                ],
            ],
        )

    def update_target_network(self, num_train_step: int, data_pass: int):
        if self.target_update_mode == "hard":
            if (num_train_step % self.copy_rate == 0) and (data_pass == 0):
                self.target_critic.load_state_dict(self.critic.state_dict())
                self.target_critic.eval()
        elif self.target_update_mode == "soft":
            polyak_averaging(self.target_critic, self.critic, self.tau)
