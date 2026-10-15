"""
train_mappo_ctde.py — MAPPO avec Centralized Training Decentralized Execution
==============================================================================
Implémentation PyTorch native — pas de dépendance SB3/EPyMARL.

Architecture CTDE :
    Actor  (par agent, partagé) : obs_locale  → action      [exécution]
    Critic (centralisé, partagé): état_global → V(s)        [entraînement]

    état_global = concat(obs_uav0, obs_uav1, obs_uav2, obs_uav3)

Référence :
    Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
    NeurIPS 2022 — https://arxiv.org/abs/2103.01955

Usage :
    python train_mappo_ctde.py --density 0.5 --timesteps 2000000 --seed 42
    python train_mappo_ctde.py --eval models/mappo_ctde_d0.5_s42/best_model.pt
    python train_mappo_ctde.py --paper   # toutes les configs
"""

import os
import csv
import time
import argparse
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from utm_marl_v2 import UTMMultiAgent, N_AGENTS, MAX_AGENT_OBS, MAX_INTRUDERS
import utm_marl_v2 as env_mod


# ===========================================================================
# HYPERPARAMÈTRES MAPPO (Yu et al. 2022)
# ===========================================================================
MAPPO_HPARAMS = dict(
    lr_actor        = 3e-4,
    lr_critic       = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,       # PPO clip range
    entropy_coef    = 0.01,
    value_loss_coef = 0.5,
    max_grad_norm   = 0.5,
    n_epochs        = 5,         # réduit 10→5 : plus rapide, stable
    batch_size      = 64,
    rollout_steps   = 128,       # réduit 512→128 : évite OOM buffer
    hidden_dim      = 256,
)

# ===========================================================================
# CALCUL DIMENSION OBS
# ===========================================================================

def get_obs_dim() -> int:
    """
    Dimension de l'obs locale d'un agent (flatten du Dict obs).
    Fixée à MAX_AGENT_OBS=7 slots pour compatibilité scalabilité
    (zero-shot transfer 4→6→8 agents sans changer l'architecture).
    """
    own_dim      = 6              # altitude, vs, hdg, dist, bearing, mission_leg
    agent_dim    = 5 * MAX_AGENT_OBS   # = 5*7 = 35
    intruder_dim = 5 * MAX_INTRUDERS   # = 5*25 = 125
    return own_dim + agent_dim + intruder_dim  # = 166


def get_global_state_dim(n_agents: int) -> int:
    """Dimension de l'état global = concat des obs de tous les agents."""
    return get_obs_dim() * n_agents


def flatten_obs(obs_dict: dict) -> np.ndarray:
    """Flatten le Dict obs en vecteur numpy."""
    keys = sorted(obs_dict.keys())
    return np.concatenate([obs_dict[k].flatten() for k in keys]).astype(np.float32)


def build_global_state(obs: dict, agents: list) -> np.ndarray:
    """Construit l'état global = concat de toutes les obs locales."""
    return np.concatenate([flatten_obs(obs[a]) for a in agents]).astype(np.float32)


# ===========================================================================
# RÉSEAUX DE NEURONES
# ===========================================================================

class Actor(nn.Module):
    """
    Acteur décentralisé : obs_locale → distribution action.
    Partagé entre tous les agents (parameter sharing).
    Déployable sans communication → CTDE.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head  = nn.Linear(hidden_dim, action_dim)
        self.log_std    = nn.Parameter(torch.zeros(action_dim))

        # Initialisation orthogonale (Yu et al. 2022)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, x: torch.Tensor):
        feat = self.net(x)
        mean = torch.tanh(self.mean_head(feat))   # actions dans [-1, 1]
        # ← Garde-fou anti-NaN (filet de sécurité, en plus du fix de
        # normalisation des returns qui traite la cause racine)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std  = self.log_std.exp().expand_as(mean)
        std  = torch.nan_to_num(std, nan=1.0, posinf=10.0, neginf=1e-6).clamp(1e-6, 10.0)
        return Normal(mean, std)

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        dist   = self.forward(x)
        action = dist.mean if deterministic else dist.sample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, dist.entropy().sum(-1)


class CentralCritic(nn.Module):
    """
    Critic centralisé : état_global → valeur V(s).
    Voit les observations de TOUS les agents simultanément.
    Utilisé seulement pendant l'entraînement → CTDE.
    """
    def __init__(self, global_state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


# ===========================================================================
# ROLLOUT BUFFER
# ===========================================================================

class RolloutBuffer:
    """Stocke les transitions pour le calcul GAE et les updates PPO."""

    def __init__(self):
        self.obs_local    = []   # obs locale par agent
        self.global_states= []   # état global
        self.actions      = []
        self.log_probs    = []
        self.rewards      = []
        self.dones        = []
        self.values       = []   # V(global_state)

    def clear(self):
        self.__init__()

    def add(self, obs_local, global_state, action, log_prob,
            reward, done, value):
        self.obs_local.append(np.array(obs_local,     dtype=np.float32))
        self.global_states.append(np.array(global_state, dtype=np.float32))
        self.actions.append(np.array(action,          dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def compute_gae(self, last_value: float, gamma: float,
                    gae_lambda: float):
        """
        Calcul des avantages GAE avec le critic centralisé.
        C'est ici que CTDE fait la différence vs IPPO :
        les avantages sont calculés sur l'état global.
        """
        advantages = []
        gae        = 0.0
        values     = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t]
                     + gamma * values[t+1] * (1 - self.dones[t])
                     - values[t])
            gae   = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def to_tensors(self, device):
        # from_numpy évite une copie mémoire supplémentaire → moins d'OOM
        def _t(lst):
            return torch.from_numpy(
                np.array(lst, dtype=np.float32)
            ).to(device)

        return (
            _t(self.obs_local),
            _t(self.global_states),
            _t(self.actions),
            _t(self.log_probs),
        )


# ===========================================================================
# AGENT MAPPO CTDE
# ===========================================================================

class RunningMeanStd:
    """
    Moyenne/variance glissante (Welford), généralisée pour supporter les
    scalaires (normalisation des returns) ET les vecteurs (normalisation
    des observations). ← FIX : sans normalisation des observations,
    l'acteur reçoit des valeurs brutes à échelles très différentes
    (distance ~10000, altitude ~3000, cos/sin ~[-1,1]...) qui saturent sa
    sortie tanh (observé : control_effort=790.958 ≈ 800 max, acteur figé
    en sortie extrême — même symptôme que le bug TD3 initial, cause
    différente : ici pas de normalisation d'entrée du tout).
    """
    def __init__(self, shape=(), eps: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == self.mean.ndim:
            x = x[None, ...]   # batch dim=1 si un seul échantillon
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


class MAPPOAgent:
    """
    Agent MAPPO avec Centralized Training Decentralized Execution.

    Entraînement :
        - Actor  reçoit obs locale → calcule policy gradient
        - Critic reçoit état global → calcule valeur pour GAE

    Exécution (watch_agent / déploiement) :
        - Seul l'Actor est nécessaire
        - Pas de communication entre agents requise
    """

    def __init__(self, obs_dim: int, global_state_dim: int,
                 action_dim: int, n_agents: int, device: str,
                 hparams: dict):
        self.n_agents = n_agents
        self.device   = device
        self.hp       = hparams

        # Réseaux (partagés entre agents)
        self.actor  = Actor(obs_dim, action_dim, hparams["hidden_dim"]).to(device)
        self.critic = CentralCritic(global_state_dim, hparams["hidden_dim"]).to(device)

        # Optimiseurs séparés actor/critic (recommandé MAPPO)
        self.actor_opt  = optim.Adam(self.actor.parameters(),
                                     lr=hparams["lr_actor"], eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(),
                                     lr=hparams["lr_critic"], eps=1e-5)

        # Buffer par agent
        self.buffers = {f"uav_{i}": RolloutBuffer()
                        for i in range(n_agents)}

        # ← FIX : normalisation des returns (critique) ET des observations
        # (acteur + critique) — sans cette dernière, l'acteur recevait des
        # obs brutes à échelles très différentes et saturait sa sortie tanh.
        self.ret_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.gs_rms  = RunningMeanStd(shape=(global_state_dim,))

        print(f"\n  MAPPO CTDE initialisé")
        print(f"  Actor params  : {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  Critic params : {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  Obs dim       : {obs_dim}")
        print(f"  Global state  : {global_state_dim}")

    def get_action(self, obs_flat: np.ndarray,
                   deterministic: bool = False, update_stats: bool = True):
        """Sélectionne une action depuis l'obs locale (exécution décentralisée).
        update_stats=True pendant l'entraînement (met à jour les stats de
        normalisation) ; False en évaluation (stats figées, comme
        VecNormalize training=False)."""
        if update_stats:
            self.obs_rms.update(obs_flat)
        obs_norm = self.obs_rms.normalize(obs_flat)
        obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(obs_t, deterministic)
        return action.squeeze().cpu().numpy(), log_prob.item()

    def get_value(self, global_state: np.ndarray, update_stats: bool = True) -> float:
        """Estime la valeur depuis l'état global (critic centralisé).
        ← FIX : le critique prédit des valeurs NORMALISÉES (voir update()),
        on dé-normalise ici pour rester en échelle brute (cohérent avec les
        rewards bruts utilisés dans le calcul GAE). Obs globale normalisée
        en entrée pour la même raison que l'obs locale de l'acteur."""
        if update_stats:
            self.gs_rms.update(global_state)
        gs_norm = self.gs_rms.normalize(global_state)
        gs_t = torch.FloatTensor(gs_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value_norm = self.critic(gs_t)
        return float(value_norm.item()) * float(self.ret_rms.std) + float(self.ret_rms.mean)

    def update(self):
        """
        Mise à jour PPO avec GAE calculé sur l'état global.
        C'est le cœur de CTDE.
        """
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for agent_id, buf in self.buffers.items():
            if len(buf.rewards) == 0:
                continue

            # GAE avec critic centralisé
            last_gs    = buf.global_states[-1] if buf.global_states else np.zeros(
                get_global_state_dim(self.n_agents))
            last_value = self.get_value(last_gs)
            advantages, returns = buf.compute_gae(
                last_value,
                self.hp["gamma"],
                self.hp["gae_lambda"],
            )

            # Normalisation des avantages (stabilise MAPPO)
            adv_t = torch.FloatTensor(advantages).to(self.device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            ret_t = torch.FloatTensor(returns).to(self.device)
            # ← FIX : le critique régresse des returns NORMALISÉS (échelle
            # ~unitaire), au lieu des valeurs brutes (jusqu'à ~1700) qui le
            # faisaient diverger (c_loss=41338 observé, puis NaN).
            self.ret_rms.update(np.array(returns, dtype=np.float32))
            ret_norm_t = (ret_t - float(self.ret_rms.mean)) / float(self.ret_rms.std)

            obs_t, gs_t, act_t, old_lp_t = buf.to_tensors(self.device)
            # ← FIX : normaliser avec les stats courantes (léger décalage
            # acceptable vs. les stats au moment de la collecte, pratique
            # standard — cf. VecNormalize)
            obs_t = (obs_t - torch.FloatTensor(self.obs_rms.mean).to(self.device)) / \
                    torch.FloatTensor(self.obs_rms.std).to(self.device)
            obs_t = obs_t.clamp(-10.0, 10.0)
            gs_t  = (gs_t - torch.FloatTensor(self.gs_rms.mean).to(self.device)) / \
                    torch.FloatTensor(self.gs_rms.std).to(self.device)
            gs_t  = gs_t.clamp(-10.0, 10.0)

            # Mini-batch PPO updates
            n     = len(buf.rewards)
            idx   = np.arange(n)

            for _ in range(self.hp["n_epochs"]):
                np.random.shuffle(idx)
                for start in range(0, n, self.hp["batch_size"]):
                    end  = min(start + self.hp["batch_size"], n)
                    mb   = idx[start:end]

                    # ── Actor loss ────────────────────────────────────────
                    dist        = self.actor.forward(obs_t[mb])
                    new_lp      = dist.log_prob(act_t[mb]).sum(-1)
                    entropy     = dist.entropy().sum(-1).mean()
                    ratio       = (new_lp - old_lp_t[mb]).exp()

                    surr1       = ratio * adv_t[mb]
                    surr2       = ratio.clamp(
                        1 - self.hp["clip_eps"],
                        1 + self.hp["clip_eps"]
                    ) * adv_t[mb]
                    actor_loss  = (-torch.min(surr1, surr2).mean()
                                   - self.hp["entropy_coef"] * entropy)

                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.hp["max_grad_norm"]
                    )
                    self.actor_opt.step()

                    # ── Critic loss (état global, returns normalisés) ─────
                    values      = self.critic(gs_t[mb]).squeeze()
                    critic_loss = (self.hp["value_loss_coef"]
                                   * nn.functional.mse_loss(values, ret_norm_t[mb]))

                    self.critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.hp["max_grad_norm"]
                    )
                    self.critic_opt.step()

                    total_actor_loss  += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy     += entropy.item()
                    n_updates         += 1

            buf.clear()

        if n_updates > 0:
            return (total_actor_loss  / n_updates,
                    total_critic_loss / n_updates,
                    total_entropy     / n_updates)
        return 0.0, 0.0, 0.0

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "hparams":    self.hp,
            "ret_rms_mean":  self.ret_rms.mean,
            "ret_rms_var":   self.ret_rms.var,
            "ret_rms_count": self.ret_rms.count,
            "obs_rms_mean":  self.obs_rms.mean,
            "obs_rms_var":   self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "gs_rms_mean":   self.gs_rms.mean,
            "gs_rms_var":    self.gs_rms.var,
            "gs_rms_count":  self.gs_rms.count,
        }, path)
        print(f"  💾 Modèle sauvegardé → {path}")

    def load(self, path: str):
        # ← FIX : PyTorch 2.6+ a changé le défaut de weights_only à True,
        # ce qui bloque le chargement de nos stats numpy (obs_rms/gs_rms/
        # ret_rms) stockées dans le checkpoint. On le désactive explicitement
        # car c'est notre propre checkpoint (source de confiance).
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if "ret_rms_mean" in ck:
            self.ret_rms.mean  = ck["ret_rms_mean"]
            self.ret_rms.var   = ck["ret_rms_var"]
            self.ret_rms.count = ck["ret_rms_count"]
        if "obs_rms_mean" in ck:
            self.obs_rms.mean  = ck["obs_rms_mean"]
            self.obs_rms.var   = ck["obs_rms_var"]
            self.obs_rms.count = ck["obs_rms_count"]
            self.gs_rms.mean   = ck["gs_rms_mean"]
            self.gs_rms.var    = ck["gs_rms_var"]
            self.gs_rms.count  = ck["gs_rms_count"]
        else:
            print("  ⚠️  Checkpoint sans stats de normalisation d'obs "
                  "(ancien format) — éval potentiellement incohérente.")
        print(f"  ✅ Modèle chargé ← {path}")

# ===========================================================================
# BOUCLE D'ENTRAÎNEMENT
# ===========================================================================

def train(args):
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Patch densité
    env_mod.TRAFFIC_DENSITY = args.density

    # Dossiers
    run_name = f"mappo_ctde_d{args.density}_s{args.seed}"
    log_dir  = os.path.join("logs",   f"{run_name}_{int(time.time())}")
    save_dir = os.path.join("models", run_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MAPPO CTDE — Centralized Training Decentralized Execution")
    print(f"  Density    : {args.density} UAV/km²")
    print(f"  Agents     : {args.n_agents}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  Device     : {device}")
    print(f"  Seed       : {args.seed}")
    print(f"{'='*60}")

    # Environnement
    env = UTMMultiAgent(n_agents=args.n_agents, render_mode=None)
    obs, _ = env.reset(seed=args.seed)

    # Dimensions
    obs_dim          = get_obs_dim()
    global_state_dim = get_global_state_dim(args.n_agents)
    action_dim       = 2   # heading + altitude

    # Agent MAPPO CTDE
    agent = MAPPOAgent(
        obs_dim          = obs_dim,
        global_state_dim = global_state_dim,
        action_dim       = action_dim,
        n_agents         = args.n_agents,
        device           = device,
        hparams          = MAPPO_HPARAMS,
    )

    # Métriques
    total_steps   = 0
    episode       = 0
    best_sr       = 0.0
    ep_rewards    = defaultdict(float)
    ep_successes  = []
    ep_crashes    = []
    ep_intrusions = []
    log_rows      = []

    print(f"\n  Entraînement démarré...\n")

    while total_steps < args.timesteps:

        # ── Collecte d'un rollout ────────────────────────────────────────
        step_in_rollout = 0

        while step_in_rollout < MAPPO_HPARAMS["rollout_steps"]:

            global_state = build_global_state(obs, env.possible_agents)

            actions   = {}
            log_probs = {}

            # Chaque agent agit depuis son obs LOCALE (decentralized exec)
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    actions[ag]   = np.zeros(action_dim, dtype=np.float32)
                    log_probs[ag] = 0.0
                    continue
                obs_flat      = flatten_obs(obs[ag])
                act, lp       = agent.get_action(obs_flat)
                actions[ag]   = act
                log_probs[ag] = lp

            # Valeur depuis état GLOBAL (centralized train)
            value = agent.get_value(global_state)

            # Step
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            total_steps      += 1
            step_in_rollout  += 1

            # Stocker dans buffer par agent
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    continue
                obs_flat = flatten_obs(obs[ag])
                done     = terms.get(ag, False) or truncs.get(ag, False)
                agent.buffers[ag].add(
                    obs_local    = obs_flat,
                    global_state = global_state,
                    action       = actions[ag],
                    log_prob     = log_probs[ag],
                    reward       = rewards.get(ag, 0.0),
                    done         = float(done),
                    value        = value,
                )
                ep_rewards[ag] += rewards.get(ag, 0.0)

            obs = next_obs

            # Reset si épisode terminé
            ep_done = (
                len(env._done_agents) >= len(env.possible_agents)
                or all(truncs.get(a, False) for a in env.possible_agents)
            )

            if ep_done:
                episode += 1

                # Métriques épisode
                for ag in env.possible_agents:
                    info = infos.get(ag, {})
                    ep_successes.append(info.get("mission_success", 0))
                    ep_crashes.append(info.get("crash", 0))
                    ep_intrusions.append(info.get("intrusion_rate", 0.0))

                ep_rewards = defaultdict(float)
                obs, _ = env.reset()

        # ── Update MAPPO ─────────────────────────────────────────────────
        al, cl, ent = agent.update()

        # ── Logging ──────────────────────────────────────────────────────
        if len(ep_successes) > 0:
            sr   = np.mean(ep_successes[-200:])
            cr   = np.mean(ep_crashes[-200:])
            ir   = np.mean(ep_intrusions[-200:])
            pct  = 100 * total_steps / args.timesteps

            print(f"  [{pct:5.1f}%] steps={total_steps:>8,}  "
                  f"ep={episode:>5}  "
                  f"SR={sr*100:5.1f}%  "
                  f"CR={cr*100:4.1f}%  "
                  f"intr={ir:.3f}  "
                  f"a_loss={al:.3f}  c_loss={cl:.3f}")

            # Sauvegarde si meilleur SR
            if sr > best_sr and total_steps > 50_000:
                best_sr = sr
                agent.save(os.path.join(save_dir, "best_model.pt"))

    # ── Sauvegarde finale ────────────────────────────────────────────────
    agent.save(os.path.join(save_dir, "final_model.pt"))

    # ── Évaluation et CSV ────────────────────────────────────────────────
    print(f"\n  📊 Évaluation finale...")
    results = evaluate_model(
        agent, args.n_agents, args.density,
        n_episodes=20, seed=args.seed + 9999
    )
    _save_csv(results, args, save_dir)

    env.close()
    print(f"\n✅ MAPPO CTDE terminé → {save_dir}")
    return save_dir


# ===========================================================================
# ÉVALUATION
# ===========================================================================

def evaluate_model(agent: MAPPOAgent, n_agents: int, density: float,
                   n_episodes: int = 20, seed: int = 0,
                   render: bool = False) -> dict:

    env_mod.TRAFFIC_DENSITY = density
    render_mode = "human" if render else None
    env = UTMMultiAgent(n_agents=n_agents, render_mode=render_mode)

    successes, crashes, intrusions, rewards, efforts = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        # Per-agent reward tracking
        ep_rews = {a: 0.0 for a in env.possible_agents}

        while not done:
            actions = {}
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    actions[ag] = np.zeros(2, dtype=np.float32)
                    continue
                obs_flat    = flatten_obs(obs[ag])
                act, _      = agent.get_action(obs_flat, deterministic=True, update_stats=False)
                actions[ag] = act

            obs, rews, terms, truncs, infos = env.step(actions)

            for ag, r in rews.items():
                ep_rews[ag] += r

            done = (
                len(env._done_agents) >= len(env.possible_agents)
                or all(truncs.get(a, False) for a in env.possible_agents)
            )

        # Per-agent metrics (comparable to train_marl.py)
        for ag in env.possible_agents:
            info = infos.get(ag, {})
            successes.append(info.get("mission_success", 0))
            crashes.append(info.get("crash", 0))
            intrusions.append(info.get("intrusion_rate", 0.0))
            efforts.append(info.get("control_effort", 0.0))
            rewards.append(ep_rews[ag])   # per-agent reward, not total

        n_succ  = sum(infos.get(a,{}).get("mission_success",0) for a in env.possible_agents)
        n_crash = sum(infos.get(a,{}).get("crash",0)           for a in env.possible_agents)
        mean_r  = np.mean(list(ep_rews.values()))
        print(f"  Ep {ep+1:2d} | "
              f"success={n_succ}/{n_agents}  "
              f"crash={n_crash}  "
              f"reward={mean_r:.1f}")

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
        "algo":           "MAPPO-CTDE",
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
            print(f"  MAPPO-CTDE | density={density} | seed={seed}")
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
        description="MAPPO CTDE — UTM Multi-Agent Conflict Resolution"
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
        obs_dim = get_obs_dim()
        gs_dim  = get_global_state_dim(args.n_agents)
        ag = MAPPOAgent(obs_dim, gs_dim, 2, args.n_agents, device, MAPPO_HPARAMS)
        ag.load(args.eval)
        results = evaluate_model(ag, args.n_agents, args.density,
                       n_episodes=20, render=args.render)
        # ← FIX : --eval n'écrivait jamais dans results.csv (même défaut
        # que celui déjà corrigé dans train_marl.py)
        args.timesteps = "eval_only"   # évite d'écrire le default (2M) trompeur
        _save_csv(results, args, args.eval)
    elif args.paper:
        run_paper_experiments(args)
    else:
        train(args)
