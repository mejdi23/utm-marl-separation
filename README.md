# Multi-Agent RL for Tactical Separation Management of UAS in UTM

Code, results and evaluation scripts for the paper

> **Multi-Agent Reinforcement Learning for Tactical Separation Management and Dynamic Rerouting of UAS in UTM**
> Mohamed Mejdi Said, Lamia Chaari Fourati, Aicha Idriss Hentati (CRNS / SM@RTS, Sfax)

The study benchmarks five MARL algorithms in a BlueSky-based urban air mobility environment:
three decentralized baselines (IPPO, ITD3, ISAC) and two CTDE variants (MAPPO-CTDE, MASAC-CTDE),
at four traffic densities (0.1, 0.3, 0.5, 0.8 intruders/km²) and four seeds (1, 7, 42, 123),
i.e. 80 training runs, plus non-learned reference policies and a zero-shot evaluation on harder encounters.

## Contents

| File | Purpose |
|---|---|
| `utm_marl_v2.py` | Training environment (PettingZoo `ParallelEnv` on BlueSky): two-leg mission, 4 agents, random intruders. |
| `utm_marl_v3.py` | Same environment with targeted (collision-course) encounters. **Evaluation only**, no policy is trained on it. |
| `train_marl.py` | Training of IPPO, ITD3, ISAC (Stable-Baselines3). |
| `train_mappo_ctde.py` | MAPPO-CTDE (custom PyTorch). |
| `train_masac_ctde.py` | MASAC-CTDE (custom PyTorch). |
| `run_phase_b_campaign.py` | Runs the multi-seed campaign sequentially. |
| `eval_models.py` | Common evaluation of the five trained algorithms (20 episodes x 4 agents per configuration). |
| `eval_baselines.py` | Non-learned references: NOOP (no control), RANDOM, MVP (BlueSky conflict resolution). |
| `eval_scalability.py` | Zero-shot evaluation with 6 and 8 agents. |
| `results.csv` | Training-environment results of the 80 runs. |
| `results/` | Evaluation CSVs (baselines on both environment versions, `eval_models.csv`). |

## Requirements

- Python 3.11
- `bluesky-simulator`, `bluesky-gym`, `pettingzoo`, `supersuit`, `gymnasium`
- `stable-baselines3`, `torch`, `numpy`, `pygame`

All experiments were run on a single Windows laptop (AMD Ryzen 5 3500U, 4 cores / 8 threads, 5.9 GB RAM, no CUDA GPU), CPU-only.

## Usage

Training (one run per call; run them one after another, see the note below):

```
python train_marl.py --algo ippo --n_agents 4 --density 0.8 --seed 42 --timesteps 300000
python train_marl.py --algo itd3 --n_agents 4 --density 0.8 --seed 42 --timesteps 300000 --n_envs 1
python train_marl.py --algo isac --n_agents 4 --density 0.8 --seed 42 --timesteps 300000 --n_envs 1
python train_mappo_ctde.py --help
python train_masac_ctde.py --help
```

Trained policies are written to `models/<algo>_d<density>_s<seed>/` (`best_model` and, for the CTDE methods, `final_model`; ITD3 and ISAC also save `vecnormalize.pkl`).

Evaluation of the trained policies (training environment, then targeted-encounter variant):

```
python eval_models.py --algo all --densities 0.1 0.3 0.5 0.8 --seeds 1 7 42 123

set UTM_ENV=utm_marl_v3
python eval_models.py --algo all --densities 0.3 0.8 --seeds 1 7 42 123
```

Non-learned reference policies:

```
python eval_baselines.py --policy all --densities 0.1 0.3 0.5 0.8 --seeds 1 7 42 123

set UTM_ENV=utm_marl_v3
python eval_baselines.py --policy noop --densities 0.3 0.8 --seeds 1 7 42 123
python eval_baselines.py --policy mvp  --densities 0.3 0.8 --seeds 1 7 42 123
```

`UTM_ENV` selects the environment module (default `utm_marl_v2`). On Linux or macOS use `export UTM_ENV=utm_marl_v3`. Output goes to `results/baselines/` (or `results/baselines_<UTM_ENV>/`) and `results/eval/eval_models.csv`. Both scripts resume automatically; delete the output folder before a fresh run.

## Evaluation protocol

20 episodes x 4 agents per (algorithm, density, seed), evaluation seed `seed + 9999 + episode`, deterministic actions, frozen observation-normalization statistics. Metrics: mission success rate, collision rate, intrusion rate, control effort.

## Notes

- **Run training and evaluation sequentially.** BlueSky keeps module-level simulation state; concurrent processes on the same machine were not tested.
- **Separation minima are design parameters** (300 m horizontal, 65 ft vertical), not regulatory values; no numerical tactical minimum exists for small UAS in U-space/UTM (see the paper, Sec. IV).
- **Known training failures, reported in the paper:** ITD3 seed 1 diverged at densities 0.3, 0.5 and 0.8; the MAPPO-CTDE policy of seed 7 at density 0.8 applies no control.
- Trained models are large; if they are not in this repository they are available from the authors on request.

## Citation

```
@article{said2026marl,
  title  = {Multi-Agent Reinforcement Learning for Tactical Separation Management and Dynamic Rerouting of UAS in UTM},
  author = {Said, Mohamed Mejdi and Chaari Fourati, Lamia and Idriss Hentati, Aicha},
  year   = {2026}
}
```

## License

To be chosen by the authors (for example MIT or Apache-2.0).
