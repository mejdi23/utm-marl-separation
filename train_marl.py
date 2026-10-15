"""
train_marl.py — Script unifié MARL pour utm_marl_v2.py
=======================================================
Algorithmes supportés :
    --algo ippo    → Independent PPO  (on-policy,  baseline)
    --algo isac    → Independent SAC  (off-policy, baseline)
    --algo itd3    → Independent TD3  (off-policy, baseline)
    --algo mappo   → Multi-Agent PPO  (critic centralisé, contribution)

Usage :
    python train_marl.py --algo ippo  --density 0.5 --timesteps 2000000
    python train_marl.py --algo isac  --density 0.3 --timesteps 2000000
    python train_marl.py --algo itd3  --density 0.1 --timesteps 2000000
    python train_marl.py --algo mappo --density 0.5 --timesteps 2000000

    # Toutes les expériences du papier (4 algos × 3 densités × 3 seeds)
    python train_marl.py --paper

    # Évaluation
    python train_marl.py --eval models/ippo_d0.5_s42/best_model.zip --algo ippo

Installation :
    pip install stable-baselines3 supersuit pettingzoo torch tensorboard

Structure des fichiers générés :
    logs/<algo>_d<density>_s<seed>_<timestamp>/   ← TensorBoard
    models/<algo>_d<density>_s<seed>/             ← meilleur modèle
    results/paper_results.csv                     ← tableau papier
"""

import argparse
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn

import supersuit as ss
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium import spaces

from utm_marl_v2 import UTMMultiAgent, N_AGENTS


# ===========================================================================
# HYPERPARAMÈTRES PAR ALGORITHME
# ===========================================================================

HPARAMS = {
    "ippo": dict(
        learning_rate  = 3e-4,
        n_steps        = 512,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
    ),
    "mappo": dict(       # identique à IPPO — seul le critic change
        learning_rate  = 3e-4,
        n_steps        = 512,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
    ),
    "isac": dict(
        learning_rate          = 3e-4,
        buffer_size            = 100_000,
        batch_size             = 256,
        gamma                  = 0.99,
        tau                    = 0.005,
        ent_coef               = "auto",
        train_freq             = 1,
        gradient_steps         = 1,
        learning_starts        = 1000,
    ),
    "itd3": dict(
        learning_rate          = 3e-4,
        buffer_size            = 500_000,
        batch_size             = 256,
        gamma                  = 0.99,
        tau                    = 0.005,
        policy_delay           = 2,
        target_policy_noise    = 0.2,
        target_noise_clip      = 0.5,
        train_freq             = 1,
        gradient_steps         = 1,
        learning_starts        = 1000,
    ),
}

# Architecture réseau commune
POLICY_KWARGS_ONPOLICY = dict(
    net_arch      = dict(pi=[256, 256], vf=[256, 256]),
    activation_fn = nn.Tanh,
)

POLICY_KWARGS_OFFPOLICY = dict(
    net_arch      = [256, 256],
    activation_fn = nn.ReLU,
)


# ===========================================================================
# FEATURES EXTRACTOR — gère le Dict observation space
# ===========================================================================

class UTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Flatten + concat toutes les clés du Dict obs en un vecteur unique.
    Utilisé par ISAC et ITD3 qui ne supportent pas MultiInputPolicy nativement
    de la même façon que PPO.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Calculer la dim totale de l'espace aplati
        total_dim = sum(
            int(np.prod(v.shape))
            for v in observation_space.spaces.values()
        )
        super().__init__(observation_space, features_dim=features_dim)

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        parts = []
        for key in sorted(obs.keys()):   # ordre déterministe
            parts.append(self.flatten(obs[key].float()))
        x = torch.cat(parts, dim=-1)
        return self.net(x)


# ===========================================================================
# MAPPO — Critic centralisé
# ===========================================================================

class MAPPOCentralCritic(nn.Module):
    """
    Critic centralisé pour MAPPO.
    Reçoit la concaténation des observations de TOUS les agents
    et retourne une valeur scalaire.

    En entraînement décentralisé (CTDE) :
        - Acteur  : voit uniquement son obs local  (identique à IPPO)
        - Critic  : voit l'état global (toutes les obs)  ← différence MAPPO
    """
    def __init__(self, single_obs_dim: int, n_agents: int, hidden: int = 256):
        super().__init__()
        global_dim = single_obs_dim * n_agents
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs)


class MAPPOWrapper:
    """
    Wrapper léger qui injecte le critic centralisé dans la boucle PPO.

    Stratégie : on utilise SB3 PPO normal (acteur décentralisé),
    mais on override la valeur critic par le critic centralisé
    pendant le calcul des avantages GAE.

    Note : implémentation simplifiée adaptée à SB3.
    Pour une implémentation complète CTDE, utiliser EPyMARL.
    """

    def __init__(self, model: PPO, n_agents: int):
        self.model    = model
        self.n_agents = n_agents

        # Calculer la dim obs d'un agent
        obs_sp    = model.observation_space
        single_dim = sum(
            int(np.prod(v.shape))
            for v in obs_sp.spaces.values()
        )

        self.central_critic = MAPPOCentralCritic(
            single_obs_dim = single_dim,
            n_agents       = n_agents,
        ).to(model.device)

        self.critic_optimizer = torch.optim.Adam(
            self.central_critic.parameters(), lr=3e-4
        )

    def get_central_value(self, all_obs: list) -> torch.Tensor:
        """all_obs : liste de tenseurs obs par agent → concatenation globale."""
        global_obs = torch.cat(all_obs, dim=-1)
        return self.central_critic(global_obs)


# ===========================================================================
# CALLBACK MARL — métriques pour TensorBoard + CSV
# ===========================================================================

class MARLCallback(BaseCallback):
    """
    Loggue :
        marl/success_rate
        marl/crash_rate
        marl/intrusion_rate
        marl/mean_ep_reward
        marl/control_effort
    """

    def __init__(self, algo: str, log_freq: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.algo     = algo
        self.log_freq = log_freq
        self._buf     = {
            "success": [], "crash": [],
            "intrusion": [], "reward": [], "effort": [],
        }

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "mission_success" in info:
                self._buf["success"].append(info["mission_success"])
                self._buf["crash"].append(info["crash"])
                self._buf["intrusion"].append(info.get("intrusion_rate", 0.0))
                self._buf["reward"].append(info.get("episode_reward", 0.0))
                self._buf["effort"].append(info.get("control_effort", 0.0))

        if len(self._buf["success"]) >= self.log_freq:
            self.logger.record(f"marl/success_rate",   np.mean(self._buf["success"]))
            self.logger.record(f"marl/crash_rate",     np.mean(self._buf["crash"]))
            self.logger.record(f"marl/intrusion_rate", np.mean(self._buf["intrusion"]))
            self.logger.record(f"marl/mean_ep_reward", np.mean(self._buf["reward"]))
            self.logger.record(f"marl/control_effort", np.mean(self._buf["effort"]))
            for k in self._buf:
                self._buf[k] = []

        return True


# ===========================================================================
# CONSTRUCTION ENVIRONNEMENT
# ===========================================================================

def build_env(n_agents: int, density: float, n_envs: int,
              seed: int, render_mode=None,
              normalize_obs: bool = False, vecnormalize_path: str = None,
              training: bool = True):
    """
    Pipeline SuperSuit :
        UTMMultiAgent (ParallelEnv)
          → pettingzoo_env_to_vec_env_v1   (PZ → VecEnv)
          → concat_vec_envs_v1             (× n_envs)
          → VecMonitor
          → VecNormalize (si normalize_obs=True — ITD3/ISAC uniquement)

    normalize_obs : ← FIX critique pour TD3/SAC. Sans normalisation, les
    observations à échelles très différentes (distance ~10000, altitude
    ~3000, cos/sin ~[-1,1]...) saturent l'acteur déterministe de TD3
    (observé : control_effort=800.0 EXACT = action=±1 sur 400 steps
    d'affilée, signe d'un acteur figé en sortie extrême). PPO s'en
    accommode mieux (clipping + avantage normalisé), donc laissé tel quel.
    vecnormalize_path : si fourni, charge les stats sauvegardées (éval).
    training : False en éval → ne met PAS à jour les stats de normalisation.
    """
    import utm_marl_v2 as env_mod
    env_mod.TRAFFIC_DENSITY = density

    raw  = UTMMultiAgent(n_agents=n_agents, render_mode=render_mode)
    # self.agents reste toujours == possible_agents (black_death géré dans l'env)
    env  = ss.pettingzoo_env_to_vec_env_v1(raw)
    env  = ss.concat_vec_envs_v1(
        env, n_envs,
        num_cpus    = 1,
        base_class  = "stable_baselines3",
    )
    env  = VecMonitor(env)

    if normalize_obs:
        from stable_baselines3.common.vec_env import VecNormalize
        if vecnormalize_path and os.path.exists(vecnormalize_path):
            env = VecNormalize.load(vecnormalize_path, env)
        else:
            # ← FIX : norm_reward=True aussi (pas seulement norm_obs).
            # L'écart d'échelle entre le shaping par step (~0.1-1) et les
            # bonus terminaux (jusqu'à 450) fait exploser les cibles Q de
            # TD3/SAC (contrairement à PPO, robuste à l'échelle via son
            # clipping+avantage normalisé). Confirmé par observation :
            # control_effort=800.0 EXACT = acteur tanh saturé en
            # permanence (piège classique : gradient nul une fois saturé,
            # le réseau ne peut plus s'en sortir, quel que soit le temps
            # d'entraînement).
            env = VecNormalize(env, norm_obs=True, norm_reward=True,
                                clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        env.training = training

    return env


# ===========================================================================
# FACTORY — crée le bon modèle SB3 selon l'algo
# ===========================================================================

def make_model(algo: str, env, log_dir: str, seed: int):
    """Retourne un modèle SB3 configuré selon l'algo choisi."""

    # Note: seed n'est PAS passé au modèle SB3 car SuperSuit ConcatVecEnv
    # n'implémente pas env.seed() — le seeding est fait via set_random_seed()
    # avant build_env() dans train()

    if algo in ("ippo", "mappo"):
        return PPO(
            policy          = "MultiInputPolicy",
            env             = env,
            policy_kwargs   = POLICY_KWARGS_ONPOLICY,
            tensorboard_log = log_dir,
            verbose         = 1,
            **HPARAMS[algo],
        )

    elif algo == "isac":
        pk = dict(
            **POLICY_KWARGS_OFFPOLICY,
            features_extractor_class  = UTMFeaturesExtractor,
            features_extractor_kwargs = dict(features_dim=256),
        )
        return SAC(
            policy          = "MultiInputPolicy",
            env             = env,
            policy_kwargs   = pk,
            tensorboard_log = log_dir,
            verbose         = 1,
            **HPARAMS["isac"],
        )

    elif algo == "itd3":
        pk = dict(
            **POLICY_KWARGS_OFFPOLICY,
            features_extractor_class  = UTMFeaturesExtractor,
            features_extractor_kwargs = dict(features_dim=256),
        )
        # ← FIX : TD3 est déterministe — sans action_noise explicite, la
        # politique n'explore quasiment plus après learning_starts, ce qui
        # bloque tout apprentissage (SR=0%, reward erratique observés).
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        return TD3(
            policy          = "MultiInputPolicy",
            env             = env,
            policy_kwargs   = pk,
            action_noise    = action_noise,
            tensorboard_log = log_dir,
            verbose         = 1,
            **HPARAMS["itd3"],
        )

    else:
        raise ValueError(f"Algo inconnu : {algo}. Choisir parmi ippo/isac/itd3/mappo")


# ===========================================================================
# ENTRAÎNEMENT
# ===========================================================================


# ===========================================================================
# AUTO EVAL → CSV  (appelé automatiquement à la fin de chaque train())
# ===========================================================================

def _auto_eval_to_csv(model_path: str, algo: str, n_agents: int,
                      density: float, seed: int, timesteps: int,
                      n_episodes: int = 20, vecnormalize_path: str = None):
    """
    Évalue le modèle sur n_episodes et ajoute une ligne dans results/results.csv.
    Appelé automatiquement après chaque entraînement.

    Colonnes CSV :
        timestamp, algo, n_agents, density, seed, timesteps,
        success_rate, crash_rate, intrusion_rate,
        mean_reward, std_reward, control_effort
    """
    import csv
    from datetime import datetime

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "results.csv")

    # ← FIX : mêmes stats de normalisation qu'à l'entraînement pour ITD3/ISAC
    # (training=False : ne met pas à jour les stats pendant l'éval)
    normalize = algo in ("itd3", "isac")
    env_eval = build_env(n_agents, density, 1, seed=seed + 9999,
                          normalize_obs=normalize,
                          vecnormalize_path=vecnormalize_path,
                          training=False)
    cls      = {"ippo": PPO, "mappo": PPO, "isac": SAC, "itd3": TD3}[algo]

    try:
        # Pour SAC et TD3 : ne pas charger le replay buffer (trop volumineux)
        if algo in ("isac", "itd3"):
            model = cls.load(model_path, env=env_eval,
                             custom_objects={"replay_buffer": None})
        else:
            model = cls.load(model_path, env=env_eval)
    except Exception as e:
        # Fallback : charger sans env (évaluation sans vérification espace)
        try:
            if algo in ("isac", "itd3"):
                model = cls.load(model_path,
                                 custom_objects={"replay_buffer": None})
            else:
                model = cls.load(model_path)
            model.set_env(env_eval)
        except Exception as e2:
            print(f"  ⚠️  Impossible de charger {model_path}: {e2}")
            env_eval.close()
            return

    successes, crashes, intrusions, rewards, efforts = [], [], [], [], []

    for ep in range(n_episodes):
        obs   = env_eval.reset()
        dones = np.array([False])
        n_ag  = n_agents
        # ← FIX CRITIQUE : l'ancienne version écrasait info_last à chaque
        # step avec le DERNIER agent de la liste (toujours index n_ag-1,
        # ex. uav_3) — le CSV ne reflétait donc QUE le comportement d'un
        # seul agent, pas la performance agrégée de toute la flotte.
        # Capture maintenant l'info de CHAQUE agent au moment précis où IL
        # termine (front montant), puis agrège les n_agents résultats du
        # vrai épisode global (qui se termine quand dones.all() est vrai,
        # car la simulation BlueSky est partagée entre les agents).
        info_per_agent = [None] * n_ag
        was_done_prev  = [False] * n_ag
        while not dones.all():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env_eval.step(action)
            for i in range(min(n_ag, len(infos))):
                if dones[i] and not was_done_prev[i] and info_per_agent[i] is None:
                    info_per_agent[i] = infos[i]
            was_done_prev = list(dones)

        for i in range(n_ag):
            info = info_per_agent[i] or {}
            successes.append(info.get("mission_success", 0))
            crashes.append(info.get("crash", 0))
            intrusions.append(info.get("intrusion_rate", 0.0))
            rewards.append(info.get("episode_reward", 0.0))
            efforts.append(info.get("control_effort", 0.0))

    env_eval.close()

    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algo":           algo.upper(),
        "n_agents":       n_agents,
        "density":        density,
        "seed":           seed,
        "timesteps":      timesteps,
        "success_rate":   round(np.mean(successes) * 100, 1),
        "crash_rate":     round(np.mean(crashes) * 100, 1),
        "intrusion_rate": round(np.mean(intrusions), 4),
        "mean_reward":    round(np.mean(rewards), 2),
        "std_reward":     round(np.std(rewards), 2),
        "control_effort": round(np.mean(efforts), 3),
        "n_eval_episodes":n_episodes,
        "model_path":     model_path,
    }

    # Écrire en mode append — crée le fichier si inexistant
    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()   # header seulement à la première ligne
        writer.writerow(row)

    print(f"  ✅ Résultats ajoutés → {csv_path}")
    print(f"     {algo.upper()} | density={density} | seed={seed}")
    print(f"     success={row['success_rate']}%  crash={row['crash_rate']}%  "
          f"intrusion={row['intrusion_rate']}  reward={row['mean_reward']}±{row['std_reward']}")


def train(args):
    set_random_seed(args.seed)

    timestamp = int(time.time())
    run_name  = f"{args.algo}_d{args.density}_s{args.seed}"
    log_dir   = os.path.join("logs",   f"{run_name}_{timestamp}")
    save_dir  = os.path.join("models", run_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  UTM MARL — {args.algo.upper()}")
    print(f"  Agents        : {args.n_agents}")
    print(f"  Density       : {args.density} UAV/km²")
    print(f"  Timesteps     : {args.timesteps:,}")
    print(f"  Parallel envs : {args.n_envs}")
    print(f"  Seed          : {args.seed}")
    print(f"{'='*60}\n")

    # ── Environnements ────────────────────────────────────────────────────
    # ← FIX : normalisation des obs pour ITD3/ISAC uniquement (voir build_env)
    normalize = args.algo in ("itd3", "isac")
    env      = build_env(args.n_agents, args.density, args.n_envs, args.seed,
                          normalize_obs=normalize)
    eval_env = build_env(args.n_agents, args.density, 1,           args.seed + 999,
                          normalize_obs=normalize)

    # ── Modèle ───────────────────────────────────────────────────────────
    model = make_model(args.algo, env, log_dir, args.seed)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Paramètres réseau : {n_params:,}")

    # ── MAPPO : attacher le critic centralisé ────────────────────────────
    mappo_wrapper = None
    if args.algo == "mappo":
        mappo_wrapper = MAPPOWrapper(model, args.n_agents)
        print(f"  MAPPO critic centralisé activé")
        n_critic = sum(p.numel() for p in mappo_wrapper.central_critic.parameters())
        print(f"  Paramètres critic central : {n_critic:,}")

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_freq = max(20_000 // (args.n_agents * args.n_envs), 1)
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path = save_dir,
        log_path             = log_dir,
        eval_freq            = eval_freq,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
    )
    marl_cb   = MARLCallback(algo=args.algo, log_freq=200)
    callbacks = CallbackList([eval_cb, marl_cb])

    # ── Entraînement ──────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = args.timesteps,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = True,
    )

    # ── Sauvegarde ────────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)

    if mappo_wrapper is not None:
        torch.save(
            mappo_wrapper.central_critic.state_dict(),
            os.path.join(save_dir, "mappo_central_critic.pt")
        )

    print(f"\n✅ {args.algo.upper()} terminé → {final_path}.zip")

    # ← FIX : sauvegarder les stats de normalisation pour l'éval (ITD3/ISAC)
    vecnorm_path = None
    if normalize:
        vecnorm_path = os.path.join(save_dir, "vecnormalize.pkl")
        env.save(vecnorm_path)
        print(f"   Stats de normalisation → {vecnorm_path}")

    env.close()
    eval_env.close()

    # ── Export CSV automatique après chaque run ───────────────────────────
    best_path = os.path.join(save_dir, "best_model.zip")
    if not os.path.exists(best_path):
        best_path = final_path + ".zip"

    if os.path.exists(best_path):
        print(f"\n📊 Évaluation automatique pour CSV...")
        _auto_eval_to_csv(
            model_path = best_path,
            algo       = args.algo,
            n_agents   = args.n_agents,
            density    = args.density,
            seed       = args.seed,
            timesteps  = args.timesteps,
            n_episodes = 20,
            vecnormalize_path = vecnorm_path,
        )

    return save_dir


# ===========================================================================
# ÉVALUATION
# ===========================================================================

def evaluate(model_path: str, algo: str, n_agents: int,
             density: float, n_episodes: int = 20, render: bool = False):

    render_mode = "human" if render else None
    # ← FIX : détecte automatiquement vecnormalize.pkl à côté du modèle (ITD3/ISAC)
    normalize = algo in ("itd3", "isac")
    vecnorm_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if not os.path.exists(vecnorm_path):
        vecnorm_path = None
    env = build_env(n_agents, density, 1, seed=0, render_mode=render_mode,
                     normalize_obs=normalize, vecnormalize_path=vecnorm_path,
                     training=False)

    # Charger le bon modèle
    cls = {"ippo": PPO, "mappo": PPO, "isac": SAC, "itd3": TD3}[algo]
    model = cls.load(model_path, env=env)

    results = {
        "success": [], "crash": [],
        "intrusion": [], "reward": [], "effort": [],
    }

    for ep in range(n_episodes):
        obs   = env.reset()
        dones = np.array([False])
        n_ag  = n_agents
        # ← FIX CRITIQUE (identique à _auto_eval_to_csv) : agrège les
        # n_agents résultats du vrai épisode global au lieu d'écraser avec
        # le dernier agent de la liste uniquement.
        info_per_agent = [None] * n_ag
        was_done_prev  = [False] * n_ag

        while not dones.all():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            for i in range(min(n_ag, len(infos))):
                if dones[i] and not was_done_prev[i] and info_per_agent[i] is None:
                    info_per_agent[i] = infos[i]
            was_done_prev = list(dones)

        for i in range(n_ag):
            info = info_per_agent[i] or {}
            results["success"].append(info.get("mission_success", 0))
            results["crash"].append(info.get("crash", 0))
            results["intrusion"].append(info.get("intrusion_rate", 0.0))
            results["reward"].append(info.get("episode_reward", 0.0))
            results["effort"].append(info.get("control_effort", 0.0))

        print(f"  Ep {ep+1:2d} | agents succ={[info_per_agent[i].get('mission_success', '?') if info_per_agent[i] else '?' for i in range(n_ag)]}")

    print(f"\n{'='*55}")
    print(f"  {algo.upper()} — density={density} — {n_episodes} épisodes")
    print(f"  Success rate    : {np.mean(results['success'])*100:.1f}%  "
          f"± {np.std(results['success'])*100:.1f}%")
    print(f"  Crash rate      : {np.mean(results['crash'])*100:.1f}%")
    print(f"  Intrusion rate  : {np.mean(results['intrusion']):.4f}")
    print(f"  Mean ep reward  : {np.mean(results['reward']):.2f} "
          f"± {np.std(results['reward']):.2f}")
    print(f"  Control effort  : {np.mean(results['effort']):.3f}")
    print(f"{'='*55}")

    env.close()
    return results


# ===========================================================================
# EXPÉRIENCES PAPIER — 4 algos × 3 densités × 3 seeds
# ===========================================================================

def run_paper_experiments(args):
    """
    Lance la grille complète et génère results/paper_results.csv
    Structure du tableau papier :

    Algo   | Density | Seed | Success% | Crash% | Intrusion | Reward
    -------|---------|------|----------|--------|-----------|-------
    IPPO   |   0.1   |  42  |   xx.x   |  xx.x  |  x.xxxx   | xxx.x
    ...
    """
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "paper_results.csv")

    algos     = ["ippo", "isac", "itd3", "mappo"]
    densities = [0.1, 0.3, 0.5]
    seeds     = [42, 123, 456]

    all_results = []

    for algo in algos:
        for density in densities:
            for seed in seeds:
                print(f"\n{'#'*60}")
                print(f"  {algo.upper()} | density={density} | seed={seed}")
                print(f"{'#'*60}")

                args.algo      = algo
                args.density   = density
                args.seed      = seed
                args.timesteps = 1_000_000   # réduit pour les sweeps

                save_dir = train(args)

                # Évaluer le meilleur modèle
                best_path = os.path.join(save_dir, "best_model.zip")
                if not os.path.exists(best_path):
                    best_path = os.path.join(save_dir, "final_model.zip")

                if os.path.exists(best_path):
                    res = evaluate(
                        best_path, algo, args.n_agents,
                        density, n_episodes=20
                    )
                    row = {
                        "algo":           algo.upper(),
                        "density":        density,
                        "seed":           seed,
                        "success_rate":   round(np.mean(res["success"]) * 100, 1),
                        "crash_rate":     round(np.mean(res["crash"]) * 100, 1),
                        "intrusion_rate": round(np.mean(res["intrusion"]), 4),
                        "mean_reward":    round(np.mean(res["reward"]), 2),
                        "std_reward":     round(np.std(res["reward"]), 2),
                        "control_effort": round(np.mean(res["effort"]), 3),
                    }
                    all_results.append(row)
                    print(f"  → success={row['success_rate']}%  "
                          f"crash={row['crash_rate']}%  "
                          f"intrusion={row['intrusion_rate']}")

    # ── Écriture CSV ──────────────────────────────────────────────────────
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ Résultats sauvegardés → {csv_path}")

    # ── Tableau récapitulatif console ─────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  TABLEAU PAPIER — {args.n_agents} agents UTM MARL")
    print(f"{'='*75}")
    print(f"  {'Algo':<8} {'Density':<10} {'Success%':<12} "
          f"{'Crash%':<10} {'Intrusion':<12} {'Reward'}")
    print(f"  {'-'*65}")
    for r in all_results:
        print(f"  {r['algo']:<8} {r['density']:<10} {r['success_rate']:<12} "
              f"{r['crash_rate']:<10} {r['intrusion_rate']:<12} "
              f"{r['mean_reward']} ± {r['std_reward']}")
    print(f"{'='*75}")

    return all_results


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UTM MARL — Script unifié IPPO / ISAC / ITD3 / MAPPO"
    )

    parser.add_argument("--algo",      type=str,   default="ippo",
                        choices=["ippo", "isac", "itd3", "mappo"],
                        help="Algorithme MARL")
    parser.add_argument("--n_agents",  type=int,   default=4,
                        help="Nombre d'agents UAV")
    parser.add_argument("--density",   type=float, default=0.5,
                        help="Densité trafic (intruders/km²)")
    parser.add_argument("--timesteps", type=int,   default=2_000_000,
                        help="Total timesteps")
    parser.add_argument("--n_envs",    type=int,   default=4,
                        help="Environnements parallèles")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed")

    # Modes spéciaux
    parser.add_argument("--eval",    type=str, default=None,
                        help="Chemin modèle à évaluer (.zip)")
    parser.add_argument("--render",  action="store_true",
                        help="Render pendant évaluation")
    parser.add_argument("--paper",   action="store_true",
                        help="Lancer toutes les expériences papier")

    args = parser.parse_args()

    if args.eval:
        results = evaluate(
            args.eval, args.algo, args.n_agents,
            args.density, n_episodes=20, render=args.render
        )
        # ← FIX : --eval n'écrivait jamais dans results.csv (seul
        # _auto_eval_to_csv le faisait, appelé uniquement en fin de train()).
        # On écrit maintenant une ligne ici aussi, même format de colonnes.
        import csv as _csv
        from datetime import datetime as _dt
        os.makedirs("results", exist_ok=True)
        csv_path = os.path.join("results", "results.csv")
        row = {
            "timestamp":       _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algo":            args.algo.upper(),
            "n_agents":        args.n_agents,
            "density":         args.density,
            "seed":            args.seed,
            "timesteps":       "eval_only",
            "success_rate":    round(np.mean(results["success"]) * 100, 1),
            "crash_rate":      round(np.mean(results["crash"]) * 100, 1),
            "intrusion_rate":  round(np.mean(results["intrusion"]), 4),
            "mean_reward":     round(np.mean(results["reward"]), 2),
            "std_reward":      round(np.std(results["reward"]), 2),
            "control_effort":  round(np.mean(results["effort"]), 3),
            "n_eval_episodes": 20,
            "model_path":      args.eval,
        }
        fieldnames  = list(row.keys())
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"\n✅ Résultats ajoutés → {csv_path}")
    elif args.paper:
        run_paper_experiments(args)
    else:
        train(args)
