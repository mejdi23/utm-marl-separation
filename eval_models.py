"""
eval_models.py — evaluation COMMUNE des 5 algorithmes appris (N=4), comparable aux baselines
==========================================================================================
Meme protocole que eval_baselines.py : 20 episodes x 4 agents, graine d'evaluation
seed + 9999 + ep, actions deterministes, statistiques de normalisation gelees.
Ajoute la mesure los_free_rate (mission reussie ET aucune perte de separation).

Checkpoints : IPPO/ITD3/ISAC -> best_model ; MAPPO/MASAC-CTDE -> final_model
(si absent, repli sur best_model). Option --ctde_ckpt best pour forcer best_model.

Usage (dossier utm_v3, env marl_env) :
  python eval_models.py --algo masac-ctde --densities 0.8 --seeds 42 --smoke
  python eval_models.py --algo masac-ctde --densities 0.8 --seeds 42
  python eval_models.py --algo all --densities 0.3 0.8 --seeds 1 7 42
Sortie : results/eval/eval_models.csv (reprise automatique)
"""

import os
import csv
import pickle
import argparse
import inspect
import numpy as np
from datetime import datetime

import importlib
env_mod = importlib.import_module(os.environ.get("UTM_ENV", "utm_marl_v2"))
UTMMultiAgent = env_mod.UTMMultiAgent

OUT_DIR = "results/eval"
CSV_PATH = os.path.join(OUT_DIR, "eval_models.csv")
os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = ["timestamp", "algo", "density", "seed", "n_agents", "n_episodes",
          "success_rate", "sr_ci_low", "sr_ci_high", "crash_rate",
          "intrusion_rate", "control_effort", "los_free_rate", "normalized_obs"]

ALL_ALGOS = ["ippo", "itd3", "isac", "mappo-ctde", "masac-ctde"]
SB3_ALGOS = ("ippo", "itd3", "mappo", "isac")
NORMALIZED_ALGOS = ("itd3", "isac")          # entraînés avec VecNormalize


# ===========================================================================
# CHEMINS DES MODÈLES
# ===========================================================================

CTDE_CKPT = "final"   # 'final' = meme checkpoint que le tableau principal (train() evalue l'agent final)


def model_path_for(algo: str, density: float, seed: int):
    base = {
        "ippo":       f"models/ippo_d{density}_s{seed}/best_model.zip",
        "itd3":       f"models/itd3_d{density}_s{seed}/best_model.zip",
        "isac":       f"models/isac_d{density}_s{seed}/best_model.zip",
        "mappo-ctde": f"models/mappo_ctde_d{density}_s{seed}/best_model.pt",
        "masac-ctde": f"models/masac_ctde_d{density}_s{seed}/best_model.pt",
    }[algo]
    if algo.endswith("-ctde") and CTDE_CKPT == "final":
        base = base.replace("best_model", "final_model")
    if os.path.exists(base):
        return base
    # repli dans l'autre sens (ex. entrainement interrompu : seul best_model existe)
    alt = (base.replace("final_model", "best_model") if "final_model" in base
           else base.replace("best_model", "final_model"))
    if os.path.exists(alt):
        print(f"    [info] {os.path.basename(base)} absent -> utilise {os.path.basename(alt)}")
        return alt
    return None


def load_vecnormalize(model_path: str):
    """Recharge les statistiques VecNormalize sauvegardées à l'entraînement
    (même méthode que stable_baselines3.VecNormalize.load : pickle)."""
    vn_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        raise FileNotFoundError(
            f"vecnormalize.pkl introuvable à côté du modèle : {vn_path}. "
            "Sans ces statistiques, les observations ne peuvent pas être "
            "normalisées comme à l'entraînement.")
    import stable_baselines3  # noqa: F401  (nécessaire pour dé-pickler la classe)
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    if not getattr(vn, "norm_obs", False):
        raise RuntimeError("vecnormalize.pkl chargé mais norm_obs=False : inattendu.")
    return vn


# ===========================================================================
# ÉVALUATION ZERO-SHOT D'UN MODÈLE
# ===========================================================================

def bootstrap_ci(per_episode: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """IC 95 % de la moyenne, bootstrap sur les épisodes."""
    rng = np.random.default_rng(seed)
    n = len(per_episode)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = per_episode[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_zeroshot(model_path: str, algo: str, n_agents: int, density: float,
                      seed: int, n_episodes: int, verbose_first_obs: bool = False):
    env_mod.TRAFFIC_DENSITY = density
    env = UTMMultiAgent(n_agents=n_agents, render_mode=None)

    vn = None
    if algo in SB3_ALGOS:
        from stable_baselines3 import PPO, TD3, SAC
        cls_map = {"ippo": PPO, "mappo": PPO, "itd3": TD3, "isac": SAC}
        model = cls_map[algo].load(model_path)
        mode = "sb3"
        if algo in NORMALIZED_ALGOS:
            vn = load_vecnormalize(model_path)
    elif algo == "mappo-ctde":
        from train_mappo_ctde import (MAPPOAgent, MAPPO_HPARAMS,
                                      get_obs_dim, get_global_state_dim)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MAPPOAgent(get_obs_dim(), get_global_state_dim(4), 2, 4, device, MAPPO_HPARAMS)
        model.load(model_path)
        mode = "ctde"
    elif algo == "masac-ctde":
        from train_masac_ctde import (MASACAgent, MASAC_HPARAMS, get_obs_dim,
                                      get_global_state_dim, get_joint_action_dim)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MASACAgent(get_obs_dim(), get_global_state_dim(4),
                           get_joint_action_dim(4), 2, 4, device, MASAC_HPARAMS)
        model.load(model_path)
        mode = "ctde"
    else:
        raise ValueError(f"Algo inconnu : {algo}")

    printed = {"done": not verbose_first_obs}

    def get_action(obs_dict):
        if mode == "sb3":
            batch = {k: np.asarray(v, dtype=np.float32).reshape(1, -1)
                     for k, v in obs_dict.items()}
            raw_max = max(float(np.abs(v).max()) for v in batch.values())
            if vn is not None:
                batch = vn.normalize_obs(batch)       # <-- correctif #47
            if not printed["done"]:
                nrm_max = max(float(np.abs(v).max()) for v in batch.values())
                print(f"    [check obs] |obs| max brut = {raw_max:.1f} ; "
                      f"après traitement = {nrm_max:.2f} "
                      f"({'normalisé' if vn is not None else 'non normalisé (IPPO)'})")
                printed["done"] = True
            action, _ = model.predict(batch, deterministic=True)
            return np.array(action).flatten()
        from train_mappo_ctde import flatten_obs as fo
        # Les stats de normalisation doivent rester GELEES a l'evaluation
        # (comme evaluate_model() de train_masac_ctde.py : update_stats=False).
        kw = {"deterministic": True}
        if "update_stats" in inspect.signature(model.get_action).parameters:
            kw["update_stats"] = False
        result = model.get_action(fo(obs_dict), **kw)
        action = result[0] if isinstance(result, tuple) else result
        return np.array(action).flatten()

    ep_succ, ep_crash, ep_intr, ep_eff, ep_los = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 9999 + ep)   # meme graines que eval_baselines
        done = False
        while not done:
            actions = {}
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    actions[ag] = np.zeros(2, dtype=np.float32)
                else:
                    actions[ag] = get_action(obs[ag])
            obs, rews, terms, truncs, infos = env.step(actions)
            done = (len(env._done_agents) >= len(env.possible_agents)
                    or all(truncs.get(a, False) for a in env.possible_agents))

        s, c, i, e, lf = [], [], [], [], []
        for ag in env.possible_agents:
            info = infos.get(ag, {})
            s.append(info.get("mission_success", 0))
            c.append(info.get("crash", 0))
            i.append(info.get("intrusion_rate", 0.0))
            e.append(info.get("control_effort", 0.0))
            lf.append(int(info.get("mission_success", 0) and
                          info.get("intrusion_rate", 0.0) == 0.0))
        ep_succ.append(np.mean(s)); ep_crash.append(np.mean(c))
        ep_intr.append(np.mean(i));  ep_eff.append(np.mean(e)); ep_los.append(np.mean(lf))

    env.close()
    ep_succ = np.array(ep_succ)
    lo, hi = bootstrap_ci(ep_succ)
    return {
        "success_rate":   100 * float(ep_succ.mean()),
        "sr_ci_low":      100 * lo,
        "sr_ci_high":     100 * hi,
        "crash_rate":     100 * float(np.mean(ep_crash)),
        "intrusion_rate": float(np.mean(ep_intr)),
        "control_effort": float(np.mean(ep_eff)),
        "los_free_rate":  100 * float(np.mean(ep_los)),
    }


# ===========================================================================
# CSV (append + reprise)
# ===========================================================================

def existing_keys():
    keys = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                keys.add((r["algo"], float(r["density"]), int(r["seed"]),
                          int(r["n_agents"]), int(r["n_episodes"])))
    return keys


def append_row(algo, density, seed, n, n_ep, r):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algo": algo.upper(), "density": density, "seed": seed,
            "n_agents": n, "n_episodes": n_ep,
            "success_rate": round(r["success_rate"], 1),
            "sr_ci_low": round(r["sr_ci_low"], 1),
            "sr_ci_high": round(r["sr_ci_high"], 1),
            "crash_rate": round(r["crash_rate"], 1),
            "intrusion_rate": round(r["intrusion_rate"], 4),
            "control_effort": round(r["control_effort"], 3),
            "los_free_rate": round(r["los_free_rate"], 1),
            "normalized_obs": int(algo in NORMALIZED_ALGOS),
        })


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zero-shot scalability (multi-seed)")
    p.add_argument("--algo", default="masac-ctde", choices=ALL_ALGOS + ["all"])
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--densities", type=float, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--agent_counts", type=int, nargs="+", default=[4])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--ctde_ckpt", choices=["final", "best"], default="final",
                   help="checkpoint des modeles CTDE (defaut: final, comme le tableau principal)")
    p.add_argument("--smoke", action="store_true",
                   help="1 épisode, N=4, sans écrire dans le CSV : vérifie le correctif")
    a = p.parse_args()
    globals()["CTDE_CKPT"] = a.ctde_ckpt

    algos = ALL_ALGOS if a.algo == "all" else [a.algo]
    densities = a.densities or [a.density]
    seeds = a.seeds or [a.seed]
    counts = [4] if a.smoke else a.agent_counts
    n_ep = 1 if a.smoke else a.episodes
    done = existing_keys() if not a.smoke else set()

    print("=" * 62)
    print(f"  EVAL MODELES (protocole commun)  algos={algos}")
    print(f"  densities={densities} seeds={seeds} N={counts} episodes={n_ep}")
    print("=" * 62)

    for algo in algos:
        for dens in densities:
            for seed in seeds:
                mp = model_path_for(algo, dens, seed)
                if mp is None:
                    print(f"  [saute] {algo} d={dens} s={seed} : modèle introuvable")
                    continue
                for n in counts:
                    if (algo.upper(), dens, seed, n, n_ep) in done:
                        print(f"  [déjà fait] {algo} d={dens} s={seed} N={n}")
                        continue
                    print(f"  {algo.upper():<10} d={dens} s={seed} N={n} ...", flush=True)
                    try:
                        r = evaluate_zeroshot(mp, algo, n, dens, seed, n_ep,
                                              verbose_first_obs=a.smoke or n == counts[0])
                    except Exception as e:
                        print(f"    ERREUR : {e}")
                        continue
                    print(f"    SR={r['success_rate']:.1f}% "
                          f"[{r['sr_ci_low']:.1f}-{r['sr_ci_high']:.1f}]  "
                          f"crash={r['crash_rate']:.1f}%  "
                          f"intr={r['intrusion_rate']:.3f}  LOS-free={r['los_free_rate']:.1f}%  "
                          f"effort={r['control_effort']:.1f}")
                    if n == 4 and algo in NORMALIZED_ALGOS and r["control_effort"] > 300:
                        print("    ⚠️  control_effort > 300 à N=4 : acteur saturé, "
                              "le problème de normalisation n'est PAS résolu. Arrêtez et "
                              "envoyez-moi cette sortie.")
                    if not a.smoke:
                        append_row(algo, dens, seed, n, n_ep, r)

    print(f"\n  CSV → {CSV_PATH}" if not a.smoke else "\n  Test rapide terminé (rien écrit).")
