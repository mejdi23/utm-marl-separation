"""
train_masac_ctde.py — MASAC avec Centralized Training Decentralized Execution
================================================================================
Implémentation PyTorch native — pendant off-policy de train_mappo_ctde.py.

Architecture CTDE (identique principe que MAPPO-CTDE) :
    Actor       (par agent, partagé) : obs_locale  → action       [exécution]
    Twin Critics(centralisés, partagés): (état_global, actions_tous) → Q(s,a)
                                                                      [entraînement]

    état_global = concat(obs_uav0, obs_uav1, obs_uav2, obs_uav3)

Pourquoi MASAC remplace ISAC dans ce papier :
    ISAC (SAC indépendant) souffre de "reward hacking" — chaque agent
    maximise reward + entropie en ne voyant que son obs locale, ce qui
    peut conduire à des politiques qui évitent les conflits sans jamais
    atteindre l'objectif (cf. ISAC SR=4.8% alors que reward=371).
    En centralisant le critic (qui voit TOUS les agents et TOUTES les
    actions), MASAC corrige la non-stationnarité multi-agent qui est
    la cause racine de ce problème — exactement comme MAPPO corrige IPPO.

Référence :
    Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
    Environments" NeurIPS 2017 (MADDPG) — principe du critic centralisé
    adapté ici à SAC (entropy-regularized off-policy).

Usage :
    python train_masac_ctde.py --density 0.1 --timesteps 2000000 --seed 42
    python train_masac_ctde.py --eval models/masac_ctde_d0.1_s42/best_model.pt
    python train_masac_ctde.py --paper   # toutes les configs
"""

import os
import csv
import time
import argparse
import random
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from utm_marl_v2 import UTMMultiAgent, N_AGENTS, MAX_AGENT_OBS, MAX_INTRUDERS
import utm_marl_v2 as env_mod


class RunningMeanStd:
    """
    Moyenne/variance glissante (Welford), pour normaliser observations et
    rewards. ← FIX préventif : même cause racine que celle diagnostiquée et
    corrigée pour ITD3 (VecNormalize) et MAPPO-CTDE — un critic/Q-function
    qui régresse des cibles à l'échelle brute de la récompense (bonus
    terminaux jusqu'à ~450, cf. Table des coefficients de reward) peut
    diverger et saturer l'acteur (tanh figé à ±1). Appliqué ici par
    anticipation, avant même de lancer un premier run long.
    """
    def __init__(self, shape=(), eps: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float32)
        self.var   = np.ones(shape, dtype=np.float32)
        self.count = eps

    def update(self, x: np.ndarray):
        # ← FIX : float32 au lieu de float64 (moitié moins de mémoire par
        # allocation temporaire — pertinent ici car cette fonction est
        # appelée à CHAQUE step de gradient, des centaines de milliers de
        # fois sur un run complet).
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == self.mean.ndim:
            x = x[None, ...]
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var   = M2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-6

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -clip, clip).astype(np.float32)


# ===========================================================================
# HYPERPARAMÈTRES MASAC
# ===========================================================================
MASAC_HPARAMS = dict(
    lr_actor        = 1e-4,       # réduit 3e-4→1e-4 : plus stable
    lr_critic       = 1e-4,       # réduit 3e-4→1e-4
    lr_alpha        = 1e-4,       # réduit 3e-4→1e-4
    gamma           = 0.99,
    tau             = 0.005,
    hidden_dim      = 256,
    buffer_size     = 100_000,   # ← réduit de 200k (évite l'OOM sur le buffer
                                   # JOINT, ~4x plus lourd qu'un buffer per-agent
                                   # classique car il stocke obs_local+global_state
                                   # pour les 4 agents à chaque transition)
    batch_size      = 256,
    learning_starts = 2_000,
    train_freq      = 4,
    gradient_steps  = 1,
    target_entropy  = None,       # auto = -action_dim
    init_alpha      = 0.2,
)


# ===========================================================================
# DIMENSIONS — identiques à train_mappo_ctde.py pour cohérence
# ===========================================================================

def get_obs_dim() -> int:
    own_dim      = 6   # altitude, vs, hdg, dist, bearing, mission_leg
    agent_dim    = 5 * MAX_AGENT_OBS
    intruder_dim = 5 * MAX_INTRUDERS
    return own_dim + agent_dim + intruder_dim  # = 166


def get_global_state_dim(n_agents: int) -> int:
    return get_obs_dim() * n_agents


def get_joint_action_dim(n_agents: int, action_dim: int = 2) -> int:
    return n_agents * action_dim


def flatten_obs(obs_dict: dict) -> np.ndarray:
    keys = sorted(obs_dict.keys())
    return np.concatenate([obs_dict[k].flatten() for k in keys]).astype(np.float32)


def build_global_state(obs: dict, agents: list) -> np.ndarray:
    return np.concatenate([flatten_obs(obs[a]) for a in agents]).astype(np.float32)


def build_joint_action(actions: dict, agents: list) -> np.ndarray:
    return np.concatenate([np.asarray(actions[a], dtype=np.float32)
                           for a in agents]).astype(np.float32)


# ===========================================================================
# RÉSEAUX DE NEURONES
# ===========================================================================

class GaussianActor(nn.Module):
    """
    Acteur décentralisé SAC : obs_locale → distribution action (tanh-squashed).
    Partagé entre tous les agents (parameter sharing).
    Déployable sans communication → exécution décentralisée.
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, x: torch.Tensor):
        feat    = self.net(x)
        mean    = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor, deterministic: bool = False):
        """Reparameterized sampling + tanh squashing (SAC standard)."""
        mean, log_std = self.forward(x)

        # Guard against NaN (numerical instability)
        mean    = torch.nan_to_num(mean,    nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(log_std, nan=-1.0, posinf=self.LOG_STD_MAX,
                                   neginf=self.LOG_STD_MIN)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp().clamp(1e-6, 10.0)

        if deterministic:
            pre_tanh = mean
        else:
            normal   = Normal(mean, std)
            pre_tanh = normal.rsample()

        action = torch.tanh(pre_tanh)

        if deterministic:
            log_prob = torch.zeros(x.shape[0], device=x.device)
        else:
            log_prob  = Normal(mean, std).log_prob(pre_tanh)
            log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob  = log_prob.sum(-1)

        return action, log_prob


class CentralCriticQ(nn.Module):
    """
    Critic centralisé (Q-function) : (état_global, actions_jointes) → Q(s,a).
    Voit les observations ET les actions de TOUS les agents.
    Utilisé seulement pendant l'entraînement → CTDE.
    Twin critics (Q1, Q2) comme dans SAC standard pour réduire l'overestimation.
    """
    def __init__(self, global_state_dim: int, joint_action_dim: int,
                 hidden_dim: int):
        super().__init__()
        input_dim = global_state_dim + joint_action_dim

        def make_q():
            net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            return net

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, global_state: torch.Tensor, joint_action: torch.Tensor):
        x = torch.cat([global_state, joint_action], dim=-1)
        return self.q1(x), self.q2(x)


# ===========================================================================
# REPLAY BUFFER — partagé, indexé par transition jointe
# ===========================================================================

class JointReplayBuffer:
    """
    Stocke des transitions jointes pour permettre au critic centralisé
    de calculer Q(global_state, joint_action) pendant l'entraînement.
    Chaque entrée contient l'info nécessaire pour TOUS les agents à ce step.
    """

    def __init__(self, capacity: int, obs_dim: int, global_state_dim: int,
                 joint_action_dim: int, n_agents: int, action_dim: int):
        self.capacity   = capacity
        self.n_agents   = n_agents
        self.action_dim = action_dim
        self.ptr        = 0
        self.size       = 0

        # obs_local par agent : (capacity, n_agents, obs_dim)
        self.obs_local      = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs_local  = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.global_state    = np.zeros((capacity, global_state_dim), dtype=np.float32)
        self.next_global_state = np.zeros((capacity, global_state_dim), dtype=np.float32)
        self.joint_action     = np.zeros((capacity, joint_action_dim), dtype=np.float32)
        self.rewards          = np.zeros((capacity, n_agents), dtype=np.float32)
        self.dones            = np.zeros((capacity, n_agents), dtype=np.float32)

    def add(self, obs_local, next_obs_local, global_state, next_global_state,
            joint_action, rewards, dones):
        idx = self.ptr
        self.obs_local[idx]          = obs_local
        self.next_obs_local[idx]     = next_obs_local
        self.global_state[idx]       = global_state
        self.next_global_state[idx]  = next_global_state
        self.joint_action[idx]       = joint_action
        self.rewards[idx]            = rewards
        self.dones[idx]               = dones

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs_local[idx]).to(device),
            torch.from_numpy(self.next_obs_local[idx]).to(device),
            torch.from_numpy(self.global_state[idx]).to(device),
            torch.from_numpy(self.next_global_state[idx]).to(device),
            torch.from_numpy(self.joint_action[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ===========================================================================
# AGENT MASAC CTDE
# ===========================================================================

class MASACAgent:
    """
    Agent MASAC avec Centralized Training Decentralized Execution.

    Entraînement :
        - Actor   reçoit obs locale  → action (reparameterized + tanh)
        - Critics reçoivent état global + actions jointes → Q(s, a_joint)
        - Alpha (entropie) auto-tuné comme dans SAC standard

    Exécution (watch_agent / déploiement) :
        - Seul l'Actor est nécessaire
        - Pas de communication entre agents requise
    """

    def __init__(self, obs_dim: int, global_state_dim: int,
                 joint_action_dim: int, action_dim: int, n_agents: int,
                 device: str, hparams: dict):
        self.n_agents   = n_agents
        self.action_dim = action_dim
        self.device     = device
        self.hp         = hparams

        # Actor partagé (décentralisé à l'exécution)
        self.actor = GaussianActor(obs_dim, action_dim,
                                   hparams["hidden_dim"]).to(device)

        # Critics centralisés (online + target)
        self.critic        = CentralCriticQ(global_state_dim, joint_action_dim,
                                            hparams["hidden_dim"]).to(device)
        self.critic_target = CentralCriticQ(global_state_dim, joint_action_dim,
                                            hparams["hidden_dim"]).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Entropie auto-tunée (alpha)
        target_entropy = hparams["target_entropy"]
        self.target_entropy = (-action_dim if target_entropy is None
                               else target_entropy)
        self.log_alpha = torch.tensor(
            np.log(hparams["init_alpha"]), dtype=torch.float32,
            requires_grad=True, device=device
        )

        # Optimiseurs
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=hparams["lr_actor"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=hparams["lr_critic"])
        self.alpha_opt  = optim.Adam([self.log_alpha],         lr=hparams["lr_alpha"])

        # ← FIX : normalisation des observations (locale + globale) et des
        # rewards, pour éviter la saturation de l'acteur (même cause que
        # ITD3/MAPPO-CTDE, corrigée par anticipation ici).
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.gs_rms  = RunningMeanStd(shape=(global_state_dim,))
        self.rew_rms = RunningMeanStd(shape=())

        print(f"\n  MASAC CTDE initialisé")
        print(f"  Actor params  : {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  Critic params : {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  Obs dim       : {obs_dim}")
        print(f"  Global state  : {global_state_dim}")
        print(f"  Joint action  : {joint_action_dim}")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def get_action(self, obs_flat: np.ndarray, deterministic: bool = False,
                   update_stats: bool = True):
        """Action depuis l'obs locale (exécution décentralisée).
        update_stats=True pendant l'entraînement, False en évaluation
        (stats figées, comme VecNormalize training=False)."""
        if update_stats:
            self.obs_rms.update(obs_flat)
        obs_norm = self.obs_rms.normalize(obs_flat)
        obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_t, deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, buffer: "JointReplayBuffer"):
        """
        Mise à jour SAC avec critic centralisé.
        Chaque agent est mis à jour avec son propre obs locale pour l'actor,
        mais le critic voit toujours l'état global + actions jointes.
        """
        if len(buffer) < self.hp["batch_size"]:
            return 0.0, 0.0, 0.0, self.alpha.item()

        (obs_local, next_obs_local, gs, next_gs,
         joint_act, rewards, dones) = buffer.sample(
            self.hp["batch_size"], self.device
        )
        # obs_local : (B, n_agents, obs_dim)

        B = obs_local.shape[0]
        gamma = self.hp["gamma"]

        # ← FIX : normalisation obs/reward avant tout passage réseau.
        # Les stats (mean/var) ne sont mises à jour qu'1 fois sur 10 —
        # pas besoin de le faire à CHAQUE step de gradient (les stats
        # convergent de toute façon très vite), et ça réduit d'un facteur 10
        # le nombre d'allocations temporaires numpy sur tout l'entraînement,
        # limitant la fragmentation mémoire qui causait le crash Windows.
        self._update_calls = getattr(self, "_update_calls", 0) + 1
        obs_np      = obs_local.cpu().numpy()
        next_obs_np = next_obs_local.cpu().numpy()
        gs_np       = gs.cpu().numpy()
        next_gs_np  = next_gs.cpu().numpy()

        if self._update_calls % 10 == 0:
            rew_np = rewards.cpu().numpy()
            self.obs_rms.update(obs_np.reshape(-1, obs_np.shape[-1]))
            self.gs_rms.update(gs_np)
            self.rew_rms.update(rew_np.reshape(-1))
            del rew_np

        obs_local      = torch.FloatTensor(self.obs_rms.normalize(obs_np)).to(self.device)
        next_obs_local = torch.FloatTensor(self.obs_rms.normalize(next_obs_np)).to(self.device)
        gs             = torch.FloatTensor(self.gs_rms.normalize(gs_np)).to(self.device)
        next_gs        = torch.FloatTensor(self.gs_rms.normalize(next_gs_np)).to(self.device)
        # Rewards : échelle seulement (pas de centrage, pour préserver le
        # signe/la structure du signal), cohérent avec VecNormalize.
        rewards = rewards / float(self.rew_rms.std)
        del obs_np, next_obs_np, gs_np, next_gs_np

        # ── Critic update ───────────────────────────────────────────────
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            for i in range(self.n_agents):
                a_i, lp_i = self.actor.sample(next_obs_local[:, i, :])
                next_actions.append(a_i)
                next_log_probs.append(lp_i)
            next_joint_action  = torch.cat(next_actions, dim=-1)
            next_log_prob_sum  = torch.stack(next_log_probs, dim=-1).sum(-1)

            q1_next, q2_next = self.critic_target(next_gs, next_joint_action)
            q_next           = torch.min(q1_next, q2_next).squeeze(-1)
            q_target_scalar  = q_next - self.alpha.detach() * next_log_prob_sum

            reward_mean = rewards.mean(dim=-1)
            done_mean   = dones.max(dim=-1)[0]
            q_target    = reward_mean + gamma * (1 - done_mean) * q_target_scalar
            # Guard against NaN in targets
            q_target    = torch.nan_to_num(q_target, nan=0.0,
                                           posinf=500.0, neginf=-500.0)

        q1, q2 = self.critic(gs, joint_act)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)
        critic_loss = (nn.functional.mse_loss(q1, q_target)
                       + nn.functional.mse_loss(q2, q_target))

        # Guard: skip update if critic loss is NaN
        if torch.isnan(critic_loss):
            return 0.0, 0.0, 0.0, self.alpha.item()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # tighter clip
        self.critic_opt.step()

        # ── Actor update ────────────────────────────────────────────────
        actor_losses = []
        for i in range(self.n_agents):
            a_i, lp_i = self.actor.sample(obs_local[:, i, :])

            joint_act_resampled = joint_act.clone()
            start = i * self.action_dim
            end   = start + self.action_dim
            joint_act_resampled[:, start:end] = a_i

            q1_pi, q2_pi = self.critic(gs, joint_act_resampled)
            q_pi = torch.min(q1_pi, q2_pi).squeeze(-1)

            actor_loss_i = (self.alpha.detach() * lp_i - q_pi).mean()
            if not torch.isnan(actor_loss_i):
                actor_losses.append(actor_loss_i)

        if not actor_losses:
            return (critic_loss.item(), 0.0, 0.0, self.alpha.item())

        actor_loss = torch.stack(actor_losses).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # tighter clip
        self.actor_opt.step()

        # ── Alpha (entropie) update ────────────────────────────────────
        with torch.no_grad():
            _, lp_for_alpha = self.actor.sample(obs_local[:, 0, :])
        lp_for_alpha = torch.nan_to_num(lp_for_alpha, nan=0.0)
        alpha_loss = -(self.log_alpha
                       * (lp_for_alpha + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft update target critic ──────────────────────────────────
        tau = self.hp["tau"]
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        return (critic_loss.item(), actor_loss.item(),
                alpha_loss.item(), self.alpha.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                   exist_ok=True)
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.detach().cpu(),
            "hparams":       self.hp,
            "obs_rms_mean":  self.obs_rms.mean,
            "obs_rms_var":   self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "gs_rms_mean":   self.gs_rms.mean,
            "gs_rms_var":    self.gs_rms.var,
            "gs_rms_count":  self.gs_rms.count,
            "rew_rms_mean":  self.rew_rms.mean,
            "rew_rms_var":   self.rew_rms.var,
            "rew_rms_count": self.rew_rms.count,
        }, path)
        print(f"  💾 Modèle sauvegardé → {path}")

    def load(self, path: str):
        # ← FIX (préventif, même cause que MAPPO) : PyTorch 2.6+ bloque par
        # défaut le chargement de checkpoints contenant des objets numpy/etc.
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if "critic_target" in ck:
            self.critic_target.load_state_dict(ck["critic_target"])
        if "log_alpha" in ck:
            self.log_alpha = ck["log_alpha"].to(self.device).requires_grad_(True)
        if "obs_rms_mean" in ck:
            self.obs_rms.mean  = ck["obs_rms_mean"]
            self.obs_rms.var   = ck["obs_rms_var"]
            self.obs_rms.count = ck["obs_rms_count"]
            self.gs_rms.mean   = ck["gs_rms_mean"]
            self.gs_rms.var    = ck["gs_rms_var"]
            self.gs_rms.count  = ck["gs_rms_count"]
            self.rew_rms.mean  = ck["rew_rms_mean"]
            self.rew_rms.var   = ck["rew_rms_var"]
            self.rew_rms.count = ck["rew_rms_count"]
        else:
            print("  ⚠️  Checkpoint sans stats de normalisation "
                  "(ancien format) — éval potentiellement incohérente.")
        print(f"  ✅ Modèle chargé ← {path}")


# ===========================================================================
# BOUCLE D'ENTRAÎNEMENT
# ===========================================================================

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_mod.TRAFFIC_DENSITY = args.density

    run_name = f"masac_ctde_d{args.density}_s{args.seed}"
    log_dir  = os.path.join("logs",   f"{run_name}_{int(time.time())}")
    save_dir = os.path.join("models", run_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MASAC CTDE — Centralized Training Decentralized Execution")
    print(f"  (off-policy counterpart to MAPPO-CTDE)")
    print(f"  Density    : {args.density} UAV/km²")
    print(f"  Agents     : {args.n_agents}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  Device     : {device}")
    print(f"  Seed       : {args.seed}")
    print(f"{'='*60}")

    env = UTMMultiAgent(n_agents=args.n_agents, render_mode=None)
    obs, _ = env.reset(seed=args.seed)

    obs_dim          = get_obs_dim()
    global_state_dim = get_global_state_dim(args.n_agents)
    action_dim        = 2
    joint_action_dim  = get_joint_action_dim(args.n_agents, action_dim)

    agent = MASACAgent(
        obs_dim          = obs_dim,
        global_state_dim = global_state_dim,
        joint_action_dim = joint_action_dim,
        action_dim       = action_dim,
        n_agents         = args.n_agents,
        device           = device,
        hparams          = MASAC_HPARAMS,
    )

    buffer = JointReplayBuffer(
        capacity         = MASAC_HPARAMS["buffer_size"],
        obs_dim          = obs_dim,
        global_state_dim = global_state_dim,
        joint_action_dim = joint_action_dim,
        n_agents         = args.n_agents,
        action_dim       = action_dim,
    )

    total_steps   = 0
    episode       = 0
    best_sr       = 0.0
    ep_successes  = deque(maxlen=200)
    ep_crashes    = deque(maxlen=200)
    ep_intrusions = deque(maxlen=200)

    print(f"\n  Entraînement démarré...\n")

    while total_steps < args.timesteps:

        global_state = build_global_state(obs, env.possible_agents)
        obs_local_arr = np.stack(
            [flatten_obs(obs[a]) for a in env.possible_agents], axis=0
        )

        # ── Sélection des actions (decentralized exec) ────────────────────
        actions = {}
        if total_steps < MASAC_HPARAMS["learning_starts"]:
            for ag in env.possible_agents:
                actions[ag] = env.action_space(ag).sample() \
                    if ag not in env._done_agents else np.zeros(2, dtype=np.float32)
        else:
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    actions[ag] = np.zeros(2, dtype=np.float32)
                    continue
                obs_flat      = flatten_obs(obs[ag])
                actions[ag]   = agent.get_action(obs_flat, deterministic=False)

        joint_action = build_joint_action(actions, env.possible_agents)

        # ── Step ────────────────────────────────────────────────────────
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        total_steps += 1

        next_global_state = build_global_state(next_obs, env.possible_agents)
        next_obs_local_arr = np.stack(
            [flatten_obs(next_obs[a]) for a in env.possible_agents], axis=0
        )

        rewards_arr = np.array(
            [rewards.get(a, 0.0) for a in env.possible_agents], dtype=np.float32
        )
        dones_arr = np.array(
            [float(terms.get(a, False) or truncs.get(a, False))
             for a in env.possible_agents], dtype=np.float32
        )

        buffer.add(
            obs_local_arr, next_obs_local_arr,
            global_state, next_global_state,
            joint_action, rewards_arr, dones_arr,
        )

        obs = next_obs

        # ── Update MASAC ────────────────────────────────────────────────
        if (total_steps >= MASAC_HPARAMS["learning_starts"]
                and total_steps % MASAC_HPARAMS["train_freq"] == 0):
            for _ in range(MASAC_HPARAMS["gradient_steps"]):
                cl, al, alpha_l, alpha_val = agent.update(buffer)

        # ── Fin d'épisode ──────────────────────────────────────────────
        ep_done = (
            len(env._done_agents) >= len(env.possible_agents)
            or all(truncs.get(a, False) for a in env.possible_agents)
        )

        if ep_done:
            episode += 1
            for ag in env.possible_agents:
                info = infos.get(ag, {})
                ep_successes.append(info.get("mission_success", 0))
                ep_crashes.append(info.get("crash", 0))
                ep_intrusions.append(info.get("intrusion_rate", 0.0))
            obs, _ = env.reset()

        # ── Logging périodique ────────────────────────────────────────
        if total_steps % 4096 == 0 and len(ep_successes) > 0:
            sr  = np.mean(ep_successes)
            cr  = np.mean(ep_crashes)
            ir  = np.mean(ep_intrusions)
            pct = 100 * total_steps / args.timesteps

            print(f"  [{pct:5.1f}%] steps={total_steps:>8,}  "
                  f"ep={episode:>5}  "
                  f"SR={sr*100:5.1f}%  "
                  f"CR={cr*100:4.1f}%  "
                  f"intr={ir:.3f}  "
                  f"buffer={len(buffer):>7,}")

            if sr > best_sr and total_steps > MASAC_HPARAMS["learning_starts"] * 5:
                best_sr = sr
                agent.save(os.path.join(save_dir, "best_model.pt"))

    # ── Sauvegarde finale ───────────────────────────────────────────────
    agent.save(os.path.join(save_dir, "final_model.pt"))

    print(f"\n  📊 Évaluation finale...")
    results = evaluate_model(
        agent, args.n_agents, args.density,
        n_episodes=20, seed=args.seed + 9999
    )
    _save_csv(results, args, save_dir)

    env.close()
    print(f"\n✅ MASAC CTDE terminé → {save_dir}")
    return save_dir


# ===========================================================================
# ÉVALUATION — per-agent reward, comparable à train_marl.py / train_mappo_ctde.py
# ===========================================================================

def evaluate_model(agent: MASACAgent, n_agents: int, density: float,
                   n_episodes: int = 20, seed: int = 0,
                   render: bool = False) -> dict:

    env_mod.TRAFFIC_DENSITY = density
    render_mode = "human" if render else None
    env = UTMMultiAgent(n_agents=n_agents, render_mode=render_mode)

    successes, crashes, intrusions, rewards, efforts = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done    = False
        ep_rews = {a: 0.0 for a in env.possible_agents}

        while not done:
            actions = {}
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    actions[ag] = np.zeros(2, dtype=np.float32)
                    continue
                obs_flat    = flatten_obs(obs[ag])
                actions[ag] = agent.get_action(obs_flat, deterministic=True, update_stats=False)

            obs, rews, terms, truncs, infos = env.step(actions)
            for ag, r in rews.items():
                ep_rews[ag] += r

            done = (
                len(env._done_agents) >= len(env.possible_agents)
                or all(truncs.get(a, False) for a in env.possible_agents)
            )

        for ag in env.possible_agents:
            info = infos.get(ag, {})
            successes.append(info.get("mission_success", 0))
            crashes.append(info.get("crash", 0))
            intrusions.append(info.get("intrusion_rate", 0.0))
            efforts.append(info.get("control_effort", 0.0))
            rewards.append(ep_rews[ag])   # reward PAR agent, pas total

        n_succ  = sum(infos.get(a,{}).get("mission_success",0) for a in env.possible_agents)
        n_crash = sum(infos.get(a,{}).get("crash",0)           for a in env.possible_agents)
        mean_r  = np.mean(list(ep_rews.values()))
        print(f"  Ep {ep+1:2d} | success={n_succ}/{n_agents}  "
              f"crash={n_crash}  reward={mean_r:.1f}")

    env.close()

    results = {
        "success_rate":   np.mean(successes) * 100,
        "crash_rate":     np.mean(crashes) * 100,
        "intrusion_rate": np.mean(intrusions),
        "mean_reward":    np.mean(rewards),
        "std_reward":     np.std(rewards),
        "control_effort": np.mean(efforts),
    }

    print(f"\n  Success : {results['success_rate']:.1f}%")
    print(f"  Crash   : {results['crash_rate']:.1f}%")
    print(f"  Intr    : {results['intrusion_rate']:.4f}")
    print(f"  Reward  : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  CE      : {results['control_effort']:.3f}")

    return results


def _save_csv(results: dict, args, save_dir: str):
    csv_path = os.path.join("results", "results.csv")
    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algo":           "MASAC-CTDE",
        "n_agents":       args.n_agents,
        "density":        args.density,
        "seed":           args.seed,
        "timesteps":      args.timesteps,
        "success_rate":   round(results["success_rate"], 1),
        "crash_rate":     round(results["crash_rate"], 1),
        "intrusion_rate": round(results["intrusion_rate"], 4),
        "mean_reward":    round(results["mean_reward"], 2),
        "std_reward":     round(results["std_reward"], 2),
        "control_effort": round(results.get("control_effort", 0.0), 3),
        "n_eval_episodes":20,
        "model_path":     save_dir,
    }
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  ✅ CSV → {csv_path}")


# ===========================================================================
# EXPÉRIENCES PAPIER
# ===========================================================================

def run_paper_experiments(args):
    densities = [0.1, 0.3, 0.5]
    seeds     = [42, 123, 456]
    for density in densities:
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"  MASAC-CTDE | density={density} | seed={seed}")
            print(f"{'#'*60}")
            args.density   = density
            args.seed      = seed
            args.timesteps = 2_000_000
            train(args)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MASAC CTDE — UTM Multi-Agent Conflict Resolution (off-policy)"
    )
    parser.add_argument("--n_agents",  type=int,   default=N_AGENTS)
    parser.add_argument("--density",   type=float, default=0.5)
    parser.add_argument("--timesteps", type=int,   default=2_000_000)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--eval",      type=str,   default=None,
                        help="Chemin modèle .pt à évaluer")
    parser.add_argument("--render",    action="store_true")
    parser.add_argument("--paper",     action="store_true",
                        help="Toutes les configs papier")

    args = parser.parse_args()

    if args.eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        obs_dim  = get_obs_dim()
        gs_dim   = get_global_state_dim(args.n_agents)
        ja_dim   = get_joint_action_dim(args.n_agents)
        ag = MASACAgent(obs_dim, gs_dim, ja_dim, 2, args.n_agents,
                        device, MASAC_HPARAMS)
        ag.load(args.eval)
        results = evaluate_model(ag, args.n_agents, args.density,
                       n_episodes=20, render=args.render)
        # ← FIX : --eval n'écrivait jamais dans results.csv (même défaut
        # déjà corrigé dans train_marl.py et train_mappo_ctde.py)
        args.timesteps = "eval_only"
        _save_csv(results, args, args.eval)
    elif args.paper:
        run_paper_experiments(args)
    else:
        train(args)
