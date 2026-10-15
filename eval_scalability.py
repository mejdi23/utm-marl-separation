"""
eval_scalability.py — Analyse de scalabilité zero-shot
========================================================
Évalue les modèles entraînés sur 4 agents directement sur 6 et 8 agents
SANS ré-entraînement (zero-shot transfer).

Possible grâce à :
    - Parameter sharing : tous les agents partagent le même réseau
    - obs_dim fixe (166, depuis l\'ajout de mission_leg) : MAX_AGENT_OBS=7 slots, zero-padded
    - Waypoints dynamiques : _compute_waypoint() s\'adapte à n_agents

Résultat → figures et CSV pour la section Scalability du papier.

Usage :
    python eval_scalability.py --algo ippo --density 0.5 --seed 42
    python eval_scalability.py --algo itd3 --density 0.5 --seed 42
    python eval_scalability.py --algo mappo-ctde --density 0.5 --seed 42
    python eval_scalability.py --algo masac-ctde --density 0.5 --seed 42
    python eval_scalability.py --all --density 0.5  # tous les algos
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

import utm_marl_v2 as env_mod
from utm_marl_v2 import UTMMultiAgent

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

AGENT_COUNTS  = [4, 6, 8]
N_EPISODES    = 50  # augmenté de 20 → 50 pour confirmer le creux à 6 agents
                     # (intervalles de confiance trop larges à 20 épisodes)
os.makedirs("results/scalability", exist_ok=True)
os.makedirs("figures/scalability", exist_ok=True)


# ===========================================================================
# ÉVALUATION ZERO-SHOT
# ===========================================================================

def evaluate_zeroshot(model_path: str, algo: str, n_agents: int,
                      density: float, seed: int = 0,
                      render: bool = False) -> dict:
    """
    Évalue un modèle entraîné sur 4 agents directement sur n_agents
    sans ré-entraînement (zero-shot).
    """
    env_mod.TRAFFIC_DENSITY = density
    env = UTMMultiAgent(n_agents=n_agents, render_mode="human" if render else None)

    # Charger le modèle selon le type
    if algo in ("ippo", "itd3", "mappo", "isac"):
        from stable_baselines3 import PPO, TD3, SAC
        cls_map = {"ippo": PPO, "mappo": PPO, "itd3": TD3, "isac": SAC}
        model = cls_map[algo].load(model_path)
        mode  = "sb3"
    elif algo == "mappo-ctde":
        from train_mappo_ctde import (
            MAPPOAgent, MAPPO_HPARAMS,
            get_obs_dim, get_global_state_dim, flatten_obs
        )
        import torch
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        obs_dim = get_obs_dim()
        # Pour MAPPO-CTDE : critic ne peut pas être réutilisé (dim différente)
        # On charge uniquement l\'actor
        gs_dim  = get_global_state_dim(4)   # entraîné sur 4 agents
        model   = MAPPOAgent(obs_dim, gs_dim, 2, 4, device, MAPPO_HPARAMS)
        model.load(model_path)
        mode    = "mappo_ctde"
    elif algo == "masac-ctde":
        from train_masac_ctde import (
            MASACAgent, MASAC_HPARAMS,
            get_obs_dim, get_global_state_dim, get_joint_action_dim, flatten_obs
        )
        import torch
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        obs_dim = get_obs_dim()
        gs_dim  = get_global_state_dim(4)
        ja_dim  = get_joint_action_dim(4)
        model   = MASACAgent(obs_dim, gs_dim, ja_dim, 2, 4, device, MASAC_HPARAMS)
        model.load(model_path)
        mode    = "masac_ctde"
    else:
        raise ValueError(f"Unknown algo: {algo}")

    def get_action(agent, obs_dict):
        if mode == "sb3":
            obs_batch = {k: v.reshape(1, -1).astype(np.float32)
                         for k, v in obs_dict.items()}
            action, _ = model.predict(obs_batch, deterministic=True)
            return np.array(action).flatten()
        elif mode in ("mappo_ctde", "masac_ctde"):
            from train_mappo_ctde import flatten_obs as fo
            obs_flat = fo(obs_dict)
            result   = model.get_action(obs_flat, deterministic=True)
            action   = result[0] if isinstance(result, tuple) else result
            return np.array(action).flatten()

    successes, crashes, intrusions, efforts = [], [], [], []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed + ep)
        done   = False

        while not done:
            actions = {}
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    actions[ag] = np.zeros(2, dtype=np.float32)
                else:
                    actions[ag] = get_action(ag, obs[ag])

            obs, rews, terms, truncs, infos = env.step(actions)
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

    env.close()

    return {
        "n_agents":       n_agents,
        "success_rate":   np.mean(successes) * 100,
        "crash_rate":     np.mean(crashes) * 100,
        "intrusion_rate": np.mean(intrusions),
        "control_effort": np.mean(efforts),
        "sr_std":         np.std(successes) * 100,
    }


# ===========================================================================
# FIGURE SCALABILITÉ
# ===========================================================================

STYLES = {
    "ippo":       {"color": "#1a6faf", "marker": "o", "ls": "-"},
    "itd3":       {"color": "#c0392b", "marker": "s", "ls": "--"},
    "mappo-ctde": {"color": "#27ae60", "marker": "^", "ls": "-."},
    "masac-ctde": {"color": "#e67e22", "marker": "D", "ls": ":"},
}

def plot_scalability(all_results: dict, density: float):
    """
    all_results : {algo: [{n_agents, success_rate, crash_rate, ...}]}
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    metrics = [
        ("success_rate",   "Success Rate (%)",   True),
        ("crash_rate",     "Crash Rate (%)",      False),
        ("intrusion_rate", "Intrusion Rate",      False),
    ]

    for ax, (metric, ylabel, hib) in zip(axes, metrics):
        for algo, results in all_results.items():
            if not results:
                continue
            ns  = [r["n_agents"] for r in results]
            vs  = [r[metric]     for r in results]
            ax.plot(ns, vs,
                    color  = STYLES[algo]["color"],
                    marker = STYLES[algo]["marker"],
                    ls     = STYLES[algo]["ls"],
                    linewidth=1.5, markersize=5,
                    label=algo.upper())
            # Error bar pour SR
            if metric == "success_rate":
                stds = [r.get("sr_std", 0) for r in results]
                ax.fill_between(ns,
                                [v-s for v,s in zip(vs,stds)],
                                [v+s for v,s in zip(vs,stds)],
                                color=STYLES[algo]["color"], alpha=0.08)

        ax.set_xticks(AGENT_COUNTS)
        ax.set_xticklabels([f"{n}" for n in AGENT_COUNTS])
        ax.set_xlabel("Number of Agents", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        arrow = "(↑ better)" if hib else "(↓ better)"
        ax.set_title(f"{ylabel}\n{arrow}", fontsize=7.5, fontweight="bold")
        ax.legend(fontsize=6.5, ncol=2)

    fig.suptitle(
        f"Zero-Shot Scalability: 4→6→8 Agents\n"
        f"(Models trained on 4 agents, density={density})",
        fontsize=8, fontweight="bold"
    )
    fig.tight_layout()
    out = f"figures/scalability/scalability_d{density}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ {out}")


def plot_scalability_heatmap(all_results: dict, density: float):
    """Heatmap algo × n_agents pour success rate."""
    algos = [a for a in all_results if all_results[a]]
    if not algos:
        return

    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    matrix = np.zeros((len(algos), len(AGENT_COUNTS)))

    for i, algo in enumerate(algos):
        for j, n in enumerate(AGENT_COUNTS):
            res = next((r for r in all_results[algo] if r["n_agents"] == n), None)
            if res:
                matrix[i, j] = res["success_rate"]

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(AGENT_COUNTS)))
    ax.set_yticks(range(len(algos)))
    ax.set_xticklabels([str(n) for n in AGENT_COUNTS])
    ax.set_yticklabels([a.upper() for a in algos])
    ax.set_xlabel("Number of Agents", fontsize=8)
    ax.set_title(f"Success Rate (%) — Scalability\n(density={density})",
                 fontsize=8, fontweight="bold")

    for i in range(len(algos)):
        for j in range(len(AGENT_COUNTS)):
            ax.text(j, i, f"{matrix[i,j]:.0f}%",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if matrix[i,j] < 40 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="SR (%)")
    fig.tight_layout()
    out = f"figures/scalability/heatmap_d{density}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ {out}")


# ===========================================================================
# CSV SCALABILITÉ
# ===========================================================================

def save_scalability_csv(all_results: dict, density: float, seed: int):
    csv_path = "results/scalability/scalability_results.csv"
    fieldnames = ["timestamp", "algo", "density", "seed",
                  "n_agents", "zero_shot", "success_rate", "crash_rate",
                  "intrusion_rate", "control_effort"]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for algo, results in all_results.items():
            for r in results:
                writer.writerow({
                    "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "algo":           algo.upper(),
                    "density":        density,
                    "seed":           seed,
                    "n_agents":       r["n_agents"],
                    "zero_shot":      int(r["n_agents"] != 4),  # 0=trained, 1=zero-shot
                    "success_rate":   round(r["success_rate"], 1),
                    "crash_rate":     round(r["crash_rate"], 1),
                    "intrusion_rate": round(r["intrusion_rate"], 4),
                    "control_effort": round(r["control_effort"], 3),
                })
    print(f"✅ CSV → {csv_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def run_algo(algo: str, density: float, seed: int) -> list:
    """Lance l\'évaluation zero-shot pour un algo sur 4, 6, 8 agents."""

    # Trouver le modèle entraîné sur 4 agents
    model_paths = {
        "ippo":       f"models/ippo_d{density}_s{seed}/best_model.zip",
        "itd3":       f"models/itd3_d{density}_s{seed}/best_model.zip",
        "mappo-ctde": f"models/mappo_ctde_d{density}_s{seed}/best_model.pt",
        "masac-ctde": f"models/masac_ctde_d{density}_s{seed}/best_model.pt",
    }

    model_path = model_paths.get(algo)
    if not model_path or not os.path.exists(model_path):
        # Essayer final_model
        model_path = model_path.replace("best_model", "final_model")
        if not os.path.exists(model_path):
            print(f"  ⚠️  {algo} model not found: {model_path}")
            return []

    results = []
    print(f"\n  {algo.upper()} — density={density} — zero-shot scalability")
    print(f"  Model: {model_path}")

    for n in AGENT_COUNTS:
        print(f"  Testing {n} agents...", end=" ", flush=True)
        try:
            r = evaluate_zeroshot(model_path, algo, n, density, seed)
            results.append(r)
            print(f"SR={r['success_rate']:.1f}%  CR={r['crash_rate']:.1f}%  "
                  f"intr={r['intrusion_rate']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UTM MARL — Zero-shot scalability evaluation"
    )
    parser.add_argument("--algo",    type=str,   default="ippo",
                        choices=["ippo","itd3","mappo-ctde","masac-ctde","all"])
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--render",  action="store_true")
    args = parser.parse_args()

    algos = (["ippo", "itd3", "mappo-ctde", "masac-ctde"]
             if args.algo == "all" else [args.algo])

    print(f"\n{'='*60}")
    print(f"  ZERO-SHOT SCALABILITY ANALYSIS")
    print(f"  Density : {args.density}")
    print(f"  Algos   : {algos}")
    print(f"  Testing : {AGENT_COUNTS} agents")
    print(f"{'='*60}")

    all_results = {}
    for algo in algos:
        all_results[algo] = run_algo(algo, args.density, args.seed)

    # Figures
    plot_scalability(all_results, args.density)
    plot_scalability_heatmap(all_results, args.density)

    # CSV
    save_scalability_csv(all_results, args.density, args.seed)

    # Console summary
    print(f"\n{'='*60}")
    print(f"  SCALABILITY SUMMARY (density={args.density})")
    print(f"{'='*60}")
    print(f"  {'Algo':<12} {'4 agents':>10} {'6 agents':>10} {'8 agents':>10}")
    print(f"  {'-'*45}")
    for algo, results in all_results.items():
        if not results:
            continue
        srs = {r["n_agents"]: r["success_rate"] for r in results}
        print(f"  {algo.upper():<12} "
              f"{srs.get(4, 'N/A'):>10.1f} "
              f"{srs.get(6, 'N/A'):>10.1f} "
              f"{srs.get(8, 'N/A'):>10.1f}")
    print(f"{'='*60}")
