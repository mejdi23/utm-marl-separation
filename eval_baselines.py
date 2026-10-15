"""
eval_baselines.py — baselines NON APPRISES pour la note #34 du superviseur
==========================================================================
Evalue, dans EXACTEMENT les memes conditions que les politiques apprises
(20 episodes x 4 agents, graines d'evaluation seed+9999+ep, actions deterministes),
trois politiques de reference :

  noop    : action (0, 0) a chaque pas = "voler droit vers la cible a l'altitude
            cible, sans aucune manoeuvre d'evitement". C'est la reference
            "pas de resolution de conflit".
  noop_off: comme noop, mais avec "RESO OFF" force apres chaque reset (au cas ou
            une resolution BlueSky serait active par defaut dans l'environnement).
  random  : actions uniformes dans [-1, 1]^2 (graine fixee). Reference de bruit.
  mvp     : action (0, 0) + resolution de conflit integree de BlueSky (MVP)
            activee pour les agents. Les intrus (acid commencant par
            --intruder_prefix, defaut "INT") sont NON cooperatifs (RESOOFF).

Chaque graine {1, 7, 42, 123} reutilise les memes scenarios que l'evaluation
des politiques apprises (seed+9999), ce qui permet des comparaisons appariees.

ATTENTION — script NON TESTE dans l'environnement du papier (je n'ai pas
acces a BlueSky ni a utm_marl_v2.py). Lancez d'abord :
    python eval_baselines.py --smoke
et envoyez-moi la sortie complete avant la campagne.

Usage :
    python eval_baselines.py --smoke
    python eval_baselines.py --policy noop --densities 0.1 0.3 0.5 0.8 --seeds 1 7 42 123
    python eval_baselines.py --policy mvp  --densities 0.1 0.3 0.5 0.8 --seeds 1 7 42 123
    python eval_baselines.py --policy all
Sortie : results/baselines/baselines_results.csv (reprise automatique).
"""

import os
import csv
import argparse
import random
from datetime import datetime

import numpy as np

import importlib
# Module d'environnement : variable UTM_ENV (defaut utm_marl_v2). Ex. (Windows) :
#   set UTM_ENV=utm_marl_v3
env_mod = importlib.import_module(os.environ.get("UTM_ENV", "utm_marl_v2"))
UTMMultiAgent, N_AGENTS = env_mod.UTMMultiAgent, env_mod.N_AGENTS

OUT_DIR = os.path.join("results", "baselines" + ("" if os.environ.get("UTM_ENV", "utm_marl_v2") == "utm_marl_v2" else "_" + os.environ["UTM_ENV"]))
OUT_CSV = os.path.join(OUT_DIR, "baselines_results.csv")
FIELDS = ["timestamp", "algo", "n_agents", "density", "seed", "n_eval_episodes",
          "success_rate", "crash_rate", "intrusion_rate", "control_effort",
          "cr_active_frac", "cr_active_int", "los_free_rate"]

POLICIES = ["noop", "noop_off", "random", "mvp"]
ALL_POLICIES = ["noop", "random", "mvp"]   # noop_off identique a noop (verifie au smoke test)


def existing_keys():
    keys = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                keys.add((r["algo"], float(r["density"]), int(r["seed"]),
                          int(r["n_agents"])))
    return keys


def append_row(row):
    os.makedirs(OUT_DIR, exist_ok=True)
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Resolution de conflit BlueSky (MVP)
# ---------------------------------------------------------------------------
def _bs():
    import bluesky as bs
    return bs


def enable_mvp():
    """Active la detection/resolution BlueSky. A rappeler apres chaque reset()."""
    bs = _bs()
    for cmd in ("ASAS ON", "RESO MVP"):
        bs.stack.stack(cmd)


def mark_intruders_noncoop(prefix, marked):
    """Les intrus ne manoeuvrent pas (RESOOFF) : seuls les agents resolvent.
    RESOOFF est envoye UNE SEULE FOIS par intrus (la commande peut basculer
    l'etat si elle est repetee). `marked` = ensemble des acid deja traites."""
    bs = _bs()
    cur = [str(a) for a in list(bs.traf.id) if str(a).startswith(prefix)]
    marked.intersection_update(cur)
    new = [a for a in cur if a not in marked]
    if new:
        bs.stack.stack("RESOOFF " + " ".join(new))
        marked.update(new)


def cr_active_fraction_probe(intruder_prefix="INT"):
    """Fraction des avions dont la resolution est active, separee en
    (agents, intrus). Diagnostic : si les intrus manoeuvrent aussi, la
    baseline MVP serait trop favorable."""
    bs = _bs()
    for path in (("traf", "cr", "active"), ("traf", "asas", "active")):
        obj = bs
        ok = True
        for p in path:
            obj = getattr(obj, p, None)
            if obj is None:
                ok = False
                break
        if ok:
            try:
                arr = np.asarray(obj, dtype=bool)
                ids = [str(x) for x in list(bs.traf.id)]
                if arr.size != len(ids):
                    return float("nan"), float("nan")
                ag = [arr[i] for i, a in enumerate(ids) if not a.startswith(intruder_prefix)]
                it = [arr[i] for i, a in enumerate(ids) if a.startswith(intruder_prefix)]
                return (float(np.mean(ag)) if ag else 0.0,
                        float(np.mean(it)) if it else 0.0)
            except Exception:
                pass
    return float("nan"), float("nan")


def cr_diag(intruder_prefix="INT"):
    """Diagnostic : etat interne du module de resolution BlueSky (smoke test)."""
    bs = _bs()
    cr = getattr(bs.traf, "cr", None)
    print("    [cr-diag] classe =", type(cr).__name__)
    names = [n for n in dir(cr) if not n.startswith("_")
             and any(k in n.lower() for k in ("reso", "active", "noreso"))]
    print("    [cr-diag] attributs =", names)
    for n in names:
        try:
            v = getattr(cr, n)
            if not callable(v):
                print("    [cr-diag]", n, "=", str(v)[:160].replace("\n", " "))
        except Exception as e:
            print("    [cr-diag]", n, "illisible :", e)
    ids = [str(x) for x in list(bs.traf.id)]
    print("    [cr-diag] indices intrus (5 premiers) =",
          [i for i, a in enumerate(ids) if a.startswith(intruder_prefix)][:5])


# ---------------------------------------------------------------------------
# Evaluation (meme boucle que evaluate_model de train_mappo_ctde.py)
# ---------------------------------------------------------------------------
def evaluate_policy(policy, n_agents, density, seed, n_episodes, intruder_prefix,
                    verbose=False):
    env_mod.TRAFFIC_DENSITY = density
    env = UTMMultiAgent(n_agents=n_agents, render_mode=None)
    rng = np.random.default_rng(seed)

    successes, crashes, intrusions, efforts = [], [], [], []
    los_free = []
    cr_fracs = []
    cr_fracs_int = []
    eval_seed = seed + 9999

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed + ep)
        marked = set()
        if policy == "mvp":
            enable_mvp()
            mark_intruders_noncoop(intruder_prefix, marked)
            try:
                _bs().stack.process()   # execute les commandes avant le 1er pas
            except Exception:
                pass
        elif policy == "noop_off":
            _bs().stack.stack("RESO OFF")
        done = False
        step = 0
        while not done:
            actions = {}
            for ag in env.possible_agents:
                if ag in env._done_agents:
                    actions[ag] = np.zeros(2, dtype=np.float32)
                elif policy == "random":
                    actions[ag] = rng.uniform(-1, 1, size=2).astype(np.float32)
                else:
                    actions[ag] = np.zeros(2, dtype=np.float32)

            if policy == "mvp":
                mark_intruders_noncoop(intruder_prefix, marked)

            obs, rews, terms, truncs, infos = env.step(actions)
            step += 1
            _fa, _fi = cr_active_fraction_probe(intruder_prefix)
            cr_fracs.append(_fa)
            cr_fracs_int.append(_fi)
            if verbose and ep == 0 and step == 30:
                try:
                    _b = _bs()
                    _ids = [str(x) for x in list(_b.traf.id)]
                    _ix = sorted([i for i, a in enumerate(_ids) if a.startswith(intruder_prefix)],
                                 key=lambda i: _ids[i])[:4]
                    print("    [int-pos30]", policy,
                          [(_ids[i], round(float(_b.traf.lat[i]), 5), round(float(_b.traf.lon[i]), 5),
                            round(float(_b.traf.trk[i]), 1)) for i in _ix])
                except Exception as e:
                    print("    [int-pos30] erreur :", e)
            if verbose and ep == 0 and step == 1:
                try:
                    print("    [diag] acid presents :", list(_bs().traf.id)[:10])
                except Exception as e:
                    print("    [diag] impossible de lister bs.traf.id :", e)

            done = (len(env._done_agents) >= len(env.possible_agents)
                    or all(truncs.get(a, False) for a in env.possible_agents))

        for ag in env.possible_agents:
            info = infos.get(ag, {})
            successes.append(info.get("mission_success", 0))
            crashes.append(info.get("crash", 0))
            intrusions.append(info.get("intrusion_rate", 0.0))
            # mission reussie ET aucune perte de separation (intrusion_rate == 0)
            los_free.append(int(info.get("mission_success", 0) and
                                info.get("intrusion_rate", 0.0) == 0.0))
            efforts.append(info.get("control_effort", 0.0))

    env.close()
    cr_frac = float(np.nanmean(cr_fracs)) if cr_fracs else float("nan")
    cr_frac_int = float(np.nanmean(cr_fracs_int)) if cr_fracs_int else float("nan")
    return {
        "success_rate": float(np.mean(successes) * 100),
        "crash_rate": float(np.mean(crashes) * 100),
        "intrusion_rate": float(np.mean(intrusions)),
        "control_effort": float(np.mean(efforts)),
        "cr_active_frac": cr_frac,
        "cr_active_int": cr_frac_int,
        "los_free_rate": float(np.mean(los_free) * 100) if los_free else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description="Baselines non apprises (note #34)")
    p.add_argument("--policy", default="noop", choices=POLICIES + ["all"])
    p.add_argument("--densities", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.8])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42, 123])
    p.add_argument("--n_agents", type=int, default=N_AGENTS)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--intruder_prefix", default="INT")
    p.add_argument("--smoke", action="store_true",
                   help="1 episode, densite 0.5, seed 42, noop, noop_off puis mvp ; n'ecrit pas le CSV")
    a = p.parse_args()

    random.seed(0)
    np.random.seed(0)

    policies = ALL_POLICIES if a.policy == "all" else [a.policy]
    if a.smoke:
        for pol in ("noop", "mvp"):
            print(f"\n=== SMOKE {pol} (densite 0.5, seed 42, 1 episode) ===")
            r = evaluate_policy(pol, a.n_agents, 0.5, 42, 1, a.intruder_prefix,
                                verbose=True)
            print("   ", {k: (round(v, 3) if v == v else v) for k, v in r.items()})
        print("\nVerifiez : (1) les acid affiches (intrus = prefixe '%s' ?), "
              "(2) cr_active_frac > 0 pour mvp (sinon la resolution BlueSky "
              "n'est pas active) ET cr_active_frac = 0 pour noop (sinon une "
              "resolution BlueSky est deja active dans l'environnement), "
              "(3) noop : control_effort = 0." % a.intruder_prefix)
        return

    done = existing_keys()
    for pol in policies:
        for dens in a.densities:
            for seed in a.seeds:
                key = (pol.upper(), dens, seed, a.n_agents)
                if key in done:
                    print(f"[skip] {key}")
                    continue
                print(f"\n>>> {pol.upper()} | densite={dens} | seed={seed}")
                try:
                    r = evaluate_policy(pol, a.n_agents, dens, seed, a.episodes,
                                        a.intruder_prefix)
                except Exception as e:
                    print("  ERREUR :", repr(e))
                    continue
                row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "algo": pol.upper(), "n_agents": a.n_agents,
                       "density": dens, "seed": seed, "n_eval_episodes": a.episodes,
                       "success_rate": round(r["success_rate"], 1),
                       "crash_rate": round(r["crash_rate"], 1),
                       "intrusion_rate": round(r["intrusion_rate"], 4),
                       "control_effort": round(r["control_effort"], 3),
                       "cr_active_frac": ("" if r["cr_active_frac"] != r["cr_active_frac"]
                                          else round(r["cr_active_frac"], 4)),
                       "cr_active_int": ("" if r["cr_active_int"] != r["cr_active_int"]
                                         else round(r["cr_active_int"], 4)),
                       "los_free_rate": round(r["los_free_rate"], 1)}
                append_row(row)
                print("  ", row)


if __name__ == "__main__":
    main()
