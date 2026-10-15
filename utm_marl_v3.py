"""
UTM Multi-Agent Conflict Resolution — PettingZoo ParallelEnv
=============================================================
Migré depuis UTM_B.py (single-agent gymnasium)
Chaque ligne modifiée est commentée avec  # ← MARL

Configuration :
    - N agents UAV (M600) avec waypoints DIFFÉRENTS (croisements forcés)
    - Spawn côté opposé au waypoint → traverse le volume → conflits réels
    - + intruders aléatoires en arrière-plan
    - Obs dim fixe (MAX_AGENT_OBS=7 slots) → scalabilité zero-shot 4→6→8
    - TARGET_RADIUS=30m → évite SR=100% trop facile
    - Terminaison individuelle par agent

API PettingZoo ParallelEnv :
    env = UTMMultiAgent(n_agents=4)
    obs, infos = env.reset(seed=42)
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rews, terms, truncs, infos = env.step(actions)
"""

import math
import random

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv          # ← MARL : import PettingZoo

import bluesky as bs
from bluesky.tools import geo
from bluesky_gym.envs.common.screen_dummy import ScreenDummy


# ===========================================================================
# CONSTANTES — inchangées depuis UTM_B.py
# ===========================================================================

CONTROL_RADIUS_KM         = 3.0
TRAFFIC_DENSITY           = 0.5
MEAN_INTRUDER_SPEED       = 12.0
MAX_INTRUDERS             = 25

# --- v3 : environnement plus exigeant (reponse a la baseline NOOP ~99%) -----
# (a) altitudes des intrus reparties sur TOUTE la bande occupee par les agents
#     (v2 : 40-120 seulement -> les agents 1-3, a >= 198, ne rencontraient
#     quasiment aucun intrus).
# (b) une fraction des intrus est creee sur une trajectoire de collision avec
#     un agent actif (rencontre ciblee, secteur frontal, meme altitude +-).
INTRUDER_ALT_SPREAD       = True
TARGETED_FRACTION         = 0.8     # part des intrus crees en rencontre ciblee
TARGET_LOOKAHEAD_S        = (15.0, 45.0)   # delai avant la rencontre predite (s)
TARGET_SECTOR_DEG         = 60.0    # ecart max autour de la direction frontale
TARGET_ALT_JITTER         = 4.0     # unites brutes BlueSky (m) < 5 m : dans la zone de collision
INTRUSION_PERSISTENCE_STEPS = 60  # augmente: agents doivent pouvoir traverser les conflits

# Separation : pas de minimum reglementaire chiffre pour le sUAS-sUAS en U-space/UTM.
# 300 m / 65 ft = parametres de conception (identiques a la v2) ; collision = echelle
# sNMAC UAS-UAS de la NASA (15.24 m ; ici 20 m / 5 m). Les 2000 ft de l'ASTM F3442
# concernent la rencontre avec du trafic habite et ne sont pas utilises.
HORIZONTAL_SEP            = 300.0   # m
VERTICAL_SEP              = 65.0    # ft

INTRUSION_PENALTY         = -50.0
CRASH_PENALTY             = -100.0
CONTROL_PENALTY           = -0.02
DIST_REWARD_SCALE         = -0.002
MISSION_SUCCESS_BONUS     = 100.0
# ← MARL 2-legs : bonus intermédiaire au passage du waypoint (avant le centre)
WAYPOINT_BONUS             = 150.0

ACTION_FREQUENCY          = 10
MAX_STEPS                 = 400  # augmente pour donner plus de temps

MAX_HEADING_CHANGE        = np.deg2rad(8)
MAX_VERTICAL_SPEED        = 5.0

UAV_CRUISE_SPEED          = 23      # knots
UAV_MIN_SPEED             = 3.0
UAV_MAX_SPEED             = 18.0

# Centre du volume UTM (point de référence)
CENTER_LAT                = 52.0
CENTER_LON                = 4.0
BASE_ALT                  = 98.0    # ft

# Écart d'altitude entre agents voisins — DOIT être nettement > VERTICAL_SEP (65ft)
# sinon les trajectoires qui se croisent près du centre déclenchent l'ASAS interne
# de BlueSky (CDMETHOD STATEBASED), qui dévie les agents de leur route optimale
# même en l'absence de tout trafic externe. Marge choisie : ~1.5x VERTICAL_SEP.
AGENT_ALT_STEP             = 100.0  # ft (remplace l'ancien pas de 25.0 < VERTICAL_SEP)

# Étagement d'altitude en LEG 2 (approche finale vers le vertiport commun) —
# volontairement plus SERRÉ qu'AGENT_ALT_STEP pour forcer un vrai séquencement
# appris par le RL (pas une séparation garantie), mais pas nul non plus.
# Justifié par la littérature vertiport réelle :
#   - FAA Vertiport Design Standards (2022) : approche structurée en zones
#     FATO/TLOF, séquencée — pas une convergence simultanée sur un point unique.
#   - Couloirs verticaux empilés (400/700/1000/1300/1600ft) utilisés en
#     approche des vertiports AAM avant fusion finale (Wei et al., "Improving
#     Autonomous Separation Assurance through Distributed RL with Attention
#     Networks", arXiv:2308.04958) — modèle réduit ici à l'échelle du sim.
#   - Le vrai goulot d'étranglement vertiport est le séquencement temporel
#     d'occupation du TLOF pad (un seul appareil à la fois), pas une
#     collision géométrique garantie (cf. littérature "vertiport terminal
#     scheduling", arXiv:2408.01152) — d'où un écart réduit mais non-nul.
LEG2_ALT_STEP               = 50.0  # ft (< VERTICAL_SEP=65ft : conflit possible mais pas garanti)

# Rayon de la zone de succès — réduit pour forcer la précision
TARGET_RADIUS             = 50.0    # m  (50m = compromis 30 trop strict / 100 trop facile)

# Waypoints sur un anneau autour du centre (croisements forcés)
# Chaque agent a un waypoint DIFFÉRENT → trajectoires croisées → conflits réels
WAYPOINT_RING_KM          = 1.2     # km du centre (equilibre)

# Rétro-compatibilité (utilisé dans quelques fonctions)
TARGET_LAT                = CENTER_LAT
TARGET_LON                = CENTER_LON
TARGET_ALT                = BASE_ALT

# ← MARL : slots d'observation pour les autres agents (séparés des intruders)
N_AGENTS                  = 4       # nombre d'agents RL par défaut
MAX_AGENT_OBS             = 7       # supporte jusqu'à 8 agents (n_agents-1 max)

# BlueSky stocke bs.traf.alt en MÈTRES (convention SI interne), alors que
# TARGET_ALT / BASE_ALT / walt / VERTICAL_SEP sont tous exprimés en PIEDS
# dans ce fichier. Toute lecture de bs.traf.alt destinée à être comparée à
# ces constantes doit être convertie via M_TO_FT — sinon l'erreur mesurée
# est ~3.28x trop petite (bug confirmé : cause du SR=0% multi-agent, et
# latent-mais-masqué dans UTM_B.py à cause du seuil de succès généreux).
M_TO_FT                    = 3.28084


def compute_spawnrate(density, mean_speed, radius_km):
    radius_m       = radius_km * 1000.0
    avg_flight_time = (math.pi * radius_m / 2.0) / mean_speed
    area_km2       = math.pi * (radius_km ** 2)
    spawnrate      = avg_flight_time / (density * area_km2)
    return max(spawnrate, 1.0)


def _compute_waypoint(agent_idx: int, n_agents: int):
    """
    Waypoint de l'agent i :
    - Position sur un anneau autour du centre (trajectoires croisees)
    - Altitude differente par agent (separation verticale structurelle)
    UAV0: 98ft  UAV1: 123ft  UAV2: 148ft  UAV3: 173ft
    """
    angle = 2 * math.pi * agent_idx / n_agents
    dlat  = (WAYPOINT_RING_KM / 111.0) * math.cos(angle)
    dlon  = (WAYPOINT_RING_KM / (111.0 * math.cos(math.radians(CENTER_LAT)))) * math.sin(angle)
    walt  = BASE_ALT + (agent_idx % max(n_agents, 4)) * AGENT_ALT_STEP
    return CENTER_LAT + dlat, CENTER_LON + dlon, walt


def _compute_spawn(agent_idx: int, n_agents: int):
    """
    Spawn de l'agent i = côté OPPOSÉ au waypoint → doit traverser le volume.
    Crée des croisements garantis entre agents.
    """
    # Angle opposé au waypoint
    angle = math.pi + 2 * math.pi * agent_idx / n_agents
    # Spawn sur le bord du volume
    radius = CONTROL_RADIUS_KM * 0.75  # equilibre distance/conflits
    dlat   = (radius / 111.0) * math.cos(angle)
    dlon   = (radius / (111.0 * math.cos(math.radians(CENTER_LAT)))) * math.sin(angle)
    return CENTER_LAT + dlat, CENTER_LON + dlon, BASE_ALT + (agent_idx % 4) * AGENT_ALT_STEP


# ===========================================================================
# ENVIRONMENT
# ===========================================================================

class UTMMultiAgent(ParallelEnv):          # ← MARL : ParallelEnv au lieu de gym.Env
    """
    4 UAVs naviguent vers le même vertiport (TARGET) en se séparant.
    Les intruders aléatoires restent présents comme bruit de trafic.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "name": "utm_marl_v3",             # ← MARL : champ requis par PettingZoo
    }

    # -----------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------
    def __init__(self, n_agents: int = N_AGENTS, render_mode=None):
        super().__init__()                  # ← MARL : super().__init__() sans args

        self.n_agents = n_agents

        # ← MARL : attributs requis par PettingZoo
        self.possible_agents = [f"uav_{i}" for i in range(n_agents)]
        self.agents          = []           # sera rempli au reset()

        # Render (identique à UTM_B)
        self.window_width  = 800
        self.window_height = 600
        self.window_size   = (self.window_width, self.window_height)
        self.window        = None
        self.clock         = None
        self.render_mode   = render_mode

        # Intruders (identique à UTM_B)
        self.spawnrate = compute_spawnrate(
            TRAFFIC_DENSITY, MEAN_INTRUDER_SPEED, CONTROL_RADIUS_KM
        )
        self.intruder_intruding = [False] * MAX_INTRUDERS

        # ← MARL : état par agent (dicts indexés par agent_id)
        self._intrusion_steps  = {}
        self._intrusion_time   = {}
        self._total_intrusions = {}
        self._total_reward     = {}
        self._episode_success  = {}
        self._episode_crash    = {}
        self._prev_dist        = {}
        self._control_effort   = {}
        # ← MARL 2-legs : 1 = en route vers le waypoint, 2 = waypoint atteint,
        #   en route vers le centre (vertiport). Bascule dans _get_reward.
        self._leg               = {}
        self._waypoint_reached  = {}

        self.step_count   = 0
        self.aircraftnumber = 0
        self._seed        = None
        self._done_agents = set()   # agents terminés (gardés pour SuperSuit)
        self._waypoints   = {}      # agent_id → (lat, lon, alt_ft)
        self._leg2_targets = {}     # agent_id → (CENTER_LAT, CENTER_LON, alt_ft étagée)

        # BlueSky — identique à UTM_B
        bs.init(mode="sim", detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack("DT 1;FF")
        bs.settings.asas_pzr = 0.16
        bs.settings.asas_pzh = 65
        bs.settings.asas_dtlookahead = 60
        bs.stack.stack("CDMETHOD STATEBASED")
        print("ASAS PZR NOW:", bs.settings.asas_pzr)

    # -----------------------------------------------------------------------
    # ← MARL : observation_space et action_space sont des MÉTHODES (par agent)
    #          Dans gym.Env c'était des attributs fixes.
    # -----------------------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Dict:
        # n_obs_agents fixé à MAX_AGENT_OBS pour compatibilité scalabilité
        # (même dim obs pour n_agents=4, 6, 8 → zero-shot transfer possible)
        n_obs = MAX_AGENT_OBS
        return spaces.Dict({
            "altitude":       spaces.Box(0.0, 3000.0,  shape=(1,), dtype=np.float32),
            "vertical_speed": spaces.Box(-MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED,
                                          shape=(1,), dtype=np.float32),
            "heading":        spaces.Box(0.0, 360.0,   shape=(1,), dtype=np.float32),
            "target_distance":spaces.Box(0.0, 10000.0, shape=(1,), dtype=np.float32),
            "target_bearing": spaces.Box(-180.0, 180.0,shape=(1,), dtype=np.float32),
            # ← MARL 2-legs : 0.0 = en route vers le waypoint, 1.0 = waypoint
            # atteint, en route vers le centre (vertiport)
            "mission_leg":    spaces.Box(0.0, 1.0,     shape=(1,), dtype=np.float32),
            # Agents voisins — toujours MAX_AGENT_OBS slots (zero-padded)
            # → obs_dim FIXE quelle que soit n_agents → scalabilité zero-shot
            "agent_distance":     spaces.Box(0.0, 5000.0,   shape=(n_obs,), dtype=np.float32),
            "agent_altitude_diff":spaces.Box(-1000.0,1000.0,shape=(n_obs,), dtype=np.float32),
            "agent_bearing_cos":  spaces.Box(-1.0, 1.0,     shape=(n_obs,), dtype=np.float32),
            "agent_bearing_sin":  spaces.Box(-1.0, 1.0,     shape=(n_obs,), dtype=np.float32),
            "agent_relative_vz":  spaces.Box(-20.0, 20.0,   shape=(n_obs,), dtype=np.float32),
            "intruder_distance":    spaces.Box(0.0, 5000.0,   shape=(MAX_INTRUDERS,), dtype=np.float32),
            "intruder_altitude_diff":spaces.Box(-1000.0,1000.0,shape=(MAX_INTRUDERS,),dtype=np.float32),
            "intruder_bearing_cos": spaces.Box(-1.0, 1.0,     shape=(MAX_INTRUDERS,), dtype=np.float32),
            "intruder_bearing_sin": spaces.Box(-1.0, 1.0,     shape=(MAX_INTRUDERS,), dtype=np.float32),
            "intruder_relative_vz": spaces.Box(-20.0, 20.0,   shape=(MAX_INTRUDERS,), dtype=np.float32),
        })

    def action_space(self, agent: str) -> spaces.Box:
        # ← MARL : méthode au lieu d'attribut — mais contenu identique à UTM_B
        return spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
        )

    # -----------------------------------------------------------------------
    # ← MARL : helper — acid BlueSky pour un agent PZ
    #   "uav_0" → "UAV0", "uav_3" → "UAV3"
    # -----------------------------------------------------------------------
    def _acid(self, agent: str) -> str:
        return agent.upper().replace("_", "")

    # -----------------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # ← MARL : pas de super().reset(seed) — PettingZoo ne l'exige pas
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        bs.traf.reset()

        # ← MARL : réinitialiser la liste active des agents
        self.agents      = self.possible_agents[:]
        self._done_agents = set()   # agents terminés mais gardés dans self.agents
        self.step_count  = 0
        self.intruder_intruding = [False] * MAX_INTRUDERS
        self.aircraftnumber = 0

        # ← MARL : réinitialiser les dicts d'état par agent
        for agent in self.agents:
            self._intrusion_steps[agent]  = 0
            self._intrusion_time[agent]   = 0
            self._total_intrusions[agent] = 0
            self._total_reward[agent]     = 0.0
            self._episode_success[agent]  = False
            self._episode_crash[agent]    = False
            self._control_effort[agent]   = 0.0
            # ← MARL 2-legs : tous les agents démarrent leg=1 (vers leur waypoint)
            self._leg[agent]              = 1
            self._waypoint_reached[agent] = False

        # Calculer waypoints et spawns dynamiquement
        self._waypoints = {}   # reset explicite avant de remplir
        self._leg2_targets = {}  # ← MARL 2-legs : cible centre, altitude étagée
        for i, agent in enumerate(self.agents):
            wlat, wlon, walt = _compute_waypoint(i, self.n_agents)
            self._waypoints[agent] = (wlat, wlon, walt)
            leg2_alt = TARGET_ALT + (i % max(self.n_agents, 4)) * LEG2_ALT_STEP
            self._leg2_targets[agent] = (CENTER_LAT, CENTER_LON, leg2_alt)

        # Créer les avions dans BlueSky
        for i, agent in enumerate(self.agents):
            slat, slon, salt = _compute_spawn(i, self.n_agents)
            wlat, wlon, _    = self._waypoints[agent]
            qdr, _           = geo.kwikqdrdist(slat, slon, wlat, wlon)

            bs.traf.cre(
                acid   = self._acid(agent),
                actype = "M600",
                aclat  = float(slat),
                aclon  = float(slon),
                achdg  = float(qdr),
                acalt  = float(salt),
                acspd  = UAV_CRUISE_SPEED,
            )

        # Vérification création
        for agent in self.agents:
            if bs.traf.id2idx(self._acid(agent)) == -1:
                raise RuntimeError(f"Failed to create {self._acid(agent)}")

        # Vérification waypoints
        for agent in self.agents:
            if agent not in self._waypoints:
                raise RuntimeError(f"Missing waypoint for {agent}")

        # Intruders aléatoires — identique à UTM_B
        self._initialize_intruders()

        # prev_dist par agent
        for agent in self.agents:
            self._prev_dist[agent] = self._dist_to_target(agent)

        # ← MARL : reset retourne {agent: obs} et {agent: info}
        obs   = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {}               for a in self.agents}
        return obs, infos

    # -----------------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------------
    def step(self, actions: dict):
        # 1. Appliquer les actions de chaque agent actif
        for agent, action in actions.items():
            if agent in self.agents:
                self._get_action(agent, action)   # ← MARL : agent en paramètre

        # 2. Avancer BlueSky (TOUS les avions avancent ensemble)
        for _ in range(ACTION_FREQUENCY):
            self._spawn_intruders()
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()

        self.step_count += 1
        truncated_global = self.step_count >= MAX_STEPS

        # 3. Collecter obs/reward/done par agent
        observations = {}
        rewards      = {}
        terminations = {}
        truncations  = {}
        infos        = {}

        for agent in self.agents:
            obs              = self._get_obs(agent)
            reward, term     = self._get_reward(agent)   # ← MARL : agent en param
            observations[agent] = obs
            rewards[agent]      = reward
            terminations[agent] = term
            truncations[agent]  = truncated_global
            infos[agent]        = self._get_info(agent)
            self._total_reward[agent] += reward

        # 4. ← MARL : marquer les agents terminés mais les garder dans self.agents
        # SuperSuit MarkovVectorWrapper exige que self.agents == self.possible_agents
        # On ne retire PAS les agents — on leur donne obs nulle et reward=0 après terminaison
        # (comportement black_death interne)
        for a in list(self._done_agents):
            if a in observations:
                observations[a]  = self._zero_obs()
                rewards[a]       = 0.0
                terminations[a]  = True
                truncations[a]   = truncated_global

        # Marquer les nouveaux agents terminés
        for a in self.agents:
            if terminations[a] or truncations[a]:
                self._done_agents.add(a)

        return observations, rewards, terminations, truncations, infos

    # -----------------------------------------------------------------------
    # INTRUDERS — copié exact depuis UTM_B, sans modification
    # -----------------------------------------------------------------------
    def _initialize_intruders(self):
        area          = math.pi * CONTROL_RADIUS_KM ** 2
        desired_count = int(TRAFFIC_DENSITY * area)
        for i in range(min(desired_count, MAX_INTRUDERS)):
            self._create_specific_intruder(i)
        self.aircraftnumber = min(desired_count, MAX_INTRUDERS)

    def _cleanup_intruders(self):
        for i in range(MAX_INTRUDERS):
            acid    = f"INT{i}"
            idx_int = bs.traf.id2idx(acid)
            if idx_int != -1:
                _, dist_nm = geo.kwikqdrdist(
                    bs.traf.lat[idx_int], bs.traf.lon[idx_int],
                    TARGET_LAT, TARGET_LON,
                )
                if dist_nm * 1.852 > CONTROL_RADIUS_KM:
                    bs.stack.stack(f"DEL {acid}")
                    self.intruder_intruding[i] = False

    def _spawn_intruders(self):
        self._cleanup_intruders()
        active = [i for i in range(MAX_INTRUDERS)
                  if bs.traf.id2idx(f"INT{i}") != -1]
        self.aircraftnumber = len(active)

        area          = math.pi * CONTROL_RADIUS_KM ** 2
        desired_count = int(TRAFFIC_DENSITY * area)

        if len(active) < desired_count:
            for i in range(MAX_INTRUDERS):
                if len(active) >= desired_count:
                    break
                if bs.traf.id2idx(f"INT{i}") == -1:
                    self._create_specific_intruder(i)
                    active.append(i)

    def _targeted_spawn(self):
        """Position/cap/altitude d'un intrus sur trajectoire de collision avec
        un agent actif, ou None si impossible."""
        live = []
        for ag in self.agents:
            if ag in self._done_agents:
                continue
            idx = bs.traf.id2idx(self._acid(ag))
            if idx != -1:
                live.append(idx)
        if not live:
            return None
        k    = live[int(np.random.randint(len(live)))]
        t    = float(np.random.uniform(*TARGET_LOOKAHEAD_S))
        trk  = float(bs.traf.trk[k])
        v_ag = float(bs.traf.tas[k])                      # m/s
        v_in = UAV_CRUISE_SPEED * 0.514444                # kt -> m/s
        # point predit de l'agent a t secondes (trajectoire rectiligne)
        plat, plon = geo.kwikpos(float(bs.traf.lat[k]), float(bs.traf.lon[k]),
                                 trk, v_ag * t / 1852.0)
        # l'intrus arrive de l'avant de l'agent (secteur frontal +-TARGET_SECTOR_DEG)
        phi  = (trk + float(np.random.uniform(-TARGET_SECTOR_DEG, TARGET_SECTOR_DEG))) % 360.0
        ilat, ilon = geo.kwikpos(plat, plon, phi, v_in * t / 1852.0)
        hdg, _ = geo.kwikqdrdist(ilat, ilon, plat, plon)
        _, dist_nm = geo.kwikqdrdist(ilat, ilon, TARGET_LAT, TARGET_LON)
        if dist_nm * 1.852 > CONTROL_RADIUS_KM:
            return None                                   # hors volume -> repli aleatoire
        alt = float(bs.traf.alt[k]) + float(np.random.uniform(-TARGET_ALT_JITTER,
                                                               TARGET_ALT_JITTER))
        return float(ilat), float(ilon), float(hdg) % 360.0, max(alt, 10.0)

    def _create_specific_intruder(self, i):
        """
        Spawn intruder dans le volume UTM.
        CRECONFS supprime car bug BlueSky windfield (TypeError numpy scalar).
        v3 : TARGETED_FRACTION des intrus sont crees sur trajectoire de
        collision avec un agent actif ; les autres sont aleatoires, avec une
        altitude repartie sur la bande des agents (INTRUDER_ALT_SPREAD).
        """
        spawn = None
        if TARGETED_FRACTION > 0 and np.random.rand() < TARGETED_FRACTION:
            spawn = self._targeted_spawn()
        if spawn is not None:
            ilat, ilon, hdg, ialt = spawn
        else:
            angle  = float(np.random.uniform(0, 2 * math.pi))
            radius = float(np.random.uniform(0.3, CONTROL_RADIUS_KM))
            dlat   = (radius / 111.0) * math.cos(angle)
            dlon   = (radius / (111.0 * math.cos(math.radians(TARGET_LAT)))) * math.sin(angle)
            ilat   = TARGET_LAT + dlat
            ilon   = TARGET_LON + dlon
            if np.random.rand() < 0.3:
                hdg, _ = geo.kwikqdrdist(ilat, ilon, TARGET_LAT, TARGET_LON)
                hdg    = float(hdg) + float(np.random.uniform(-30, 30))
            else:
                hdg = float(np.random.uniform(0, 360))
            if INTRUDER_ALT_SPREAD:
                ialt = float(np.random.uniform(BASE_ALT - 20.0,
                                               BASE_ALT + 3 * AGENT_ALT_STEP + 20.0))
            else:
                ialt = float(np.random.uniform(40, 120))

        bs.traf.cre(
            acid   = f"INT{i}",
            actype = "M600",
            aclat  = float(ilat),
            aclon  = float(ilon),
            achdg  = float(hdg) % 360.0,
            acalt  = float(ialt),
            acspd  = UAV_CRUISE_SPEED,
        )

    # -----------------------------------------------------------------------
    # OBSERVATION
    # -----------------------------------------------------------------------
    def _alt_ft(self, idx: int) -> float:
        """bs.traf.alt est en mètres (SI BlueSky) → conversion en pieds,
        seule unité utilisée pour TARGET_ALT/BASE_ALT/walt/VERTICAL_SEP."""
        return float(bs.traf.alt[idx]) * M_TO_FT

    def _current_target(self, agent: str):
        """← MARL 2-legs : cible ACTIVE de l'agent selon sa leg.
        leg==1 → son waypoint individuel (lat, lon, walt)
        leg==2 → le centre/vertiport, altitude étagée par agent (approche
                 en couloirs, cf. LEG2_ALT_STEP) — pas un point unique exact
                 pour tous, cohérent avec les procédures d'approche vertiport
                 réelles (voir commentaire sur LEG2_ALT_STEP).
        """
        if self._leg.get(agent, 1) == 1:
            return self._waypoints.get(agent, (CENTER_LAT, CENTER_LON, BASE_ALT))
        return self._leg2_targets.get(agent, (CENTER_LAT, CENTER_LON, TARGET_ALT))

    def _get_obs(self, agent: str) -> dict:
        if agent in self._done_agents:
            return self._zero_obs()
        idx = bs.traf.id2idx(self._acid(agent))
        if idx == -1:
            return self._zero_obs()

        alt = self._alt_ft(idx)
        vz  = float(bs.traf.vs[idx])
        hdg = float(bs.traf.hdg[idx])

        wlat, wlon, _ = self._current_target(agent)
        qdr, dist_nm  = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            wlat, wlon,
        )
        dist_m      = dist_nm * 1852.0
        rel_bearing = ((qdr - hdg + 180) % 360) - 180

        # Slot 1 : autres agents RL — toujours MAX_AGENT_OBS slots (zero-padded)
        # Tri par distance pour scalabilité (voit les MAX_AGENT_OBS plus proches)
        ag_d, ag_a, ag_c, ag_s, ag_vz = [], [], [], [], []
        other_data = []
        for other in self.possible_agents:
            if other == agent:
                continue
            oidx = bs.traf.id2idx(self._acid(other))
            if oidx == -1:
                continue
            q, d = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[oidx], bs.traf.lon[oidx],
            )
            d_m = d * 1852.0
            b   = np.deg2rad(q - hdg)
            other_data.append((d_m,
                               self._alt_ft(oidx) - alt,
                               float(np.cos(b)),
                               float(np.sin(b)),
                               float(bs.traf.vs[oidx]) - vz))
        # Trier par distance, garder les MAX_AGENT_OBS plus proches
        other_data.sort(key=lambda x: x[0])
        other_data = other_data[:MAX_AGENT_OBS]
        # Zero-pad jusqu'à MAX_AGENT_OBS
        while len(other_data) < MAX_AGENT_OBS:
            other_data.append((10000.0, 0.0, 0.0, 0.0, 0.0))
        for d_m, a_diff, cos_b, sin_b, rvz in other_data:
            ag_d.append(d_m); ag_a.append(a_diff)
            ag_c.append(cos_b); ag_s.append(sin_b); ag_vz.append(rvz)

        # Slot 2 : intruders aléatoires
        intr_d, intr_a, intr_c, intr_s, intr_vz = [], [], [], [], []
        for i in range(MAX_INTRUDERS):
            int_idx = bs.traf.id2idx(f"INT{i}")
            if int_idx == -1:
                intr_d.append(10000.0); intr_a.append(0.0)
                intr_c.append(0.0);     intr_s.append(0.0); intr_vz.append(0.0)
                continue
            q, d = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx],
            )
            intr_d.append(d * 1852.0)
            intr_a.append(self._alt_ft(int_idx) - alt)
            b = np.deg2rad(q - hdg)
            intr_c.append(float(np.cos(b)))
            intr_s.append(float(np.sin(b)))
            intr_vz.append(float(bs.traf.vs[int_idx]) - vz)

        return {
            "altitude":              np.array([alt],         dtype=np.float32),
            "vertical_speed":        np.array([vz],          dtype=np.float32),
            "heading":               np.array([hdg],         dtype=np.float32),
            "target_distance":       np.array([dist_m],      dtype=np.float32),
            "target_bearing":        np.array([rel_bearing], dtype=np.float32),
            "mission_leg":           np.array([1.0 if self._leg.get(agent, 1) == 2 else 0.0],
                                                dtype=np.float32),
            "agent_distance":        np.array(ag_d,    dtype=np.float32),
            "agent_altitude_diff":   np.array(ag_a,    dtype=np.float32),
            "agent_bearing_cos":     np.array(ag_c,    dtype=np.float32),
            "agent_bearing_sin":     np.array(ag_s,    dtype=np.float32),
            "agent_relative_vz":     np.array(ag_vz,   dtype=np.float32),
            "intruder_distance":     np.array(intr_d,  dtype=np.float32),
            "intruder_altitude_diff":np.array(intr_a,  dtype=np.float32),
            "intruder_bearing_cos":  np.array(intr_c,  dtype=np.float32),
            "intruder_bearing_sin":  np.array(intr_s,  dtype=np.float32),
            "intruder_relative_vz":  np.array(intr_vz, dtype=np.float32),
        }

    def _zero_obs(self) -> dict:
        return {
            "altitude":              np.zeros(1,               dtype=np.float32),
            "vertical_speed":        np.zeros(1,               dtype=np.float32),
            "heading":               np.zeros(1,               dtype=np.float32),
            "target_distance":       np.zeros(1,               dtype=np.float32),
            "target_bearing":        np.zeros(1,               dtype=np.float32),
            "mission_leg":           np.zeros(1,               dtype=np.float32),
            "agent_distance":        np.full(MAX_AGENT_OBS, 10000., dtype=np.float32),
            "agent_altitude_diff":   np.zeros(MAX_AGENT_OBS,  dtype=np.float32),
            "agent_bearing_cos":     np.zeros(MAX_AGENT_OBS,  dtype=np.float32),
            "agent_bearing_sin":     np.zeros(MAX_AGENT_OBS,  dtype=np.float32),
            "agent_relative_vz":     np.zeros(MAX_AGENT_OBS,  dtype=np.float32),
            "intruder_distance":     np.full(MAX_INTRUDERS, 10000., dtype=np.float32),
            "intruder_altitude_diff":np.zeros(MAX_INTRUDERS,  dtype=np.float32),
            "intruder_bearing_cos":  np.zeros(MAX_INTRUDERS,  dtype=np.float32),
            "intruder_bearing_sin":  np.zeros(MAX_INTRUDERS,  dtype=np.float32),
            "intruder_relative_vz":  np.zeros(MAX_INTRUDERS,  dtype=np.float32),
        }

    # -----------------------------------------------------------------------
    # ACTION
    # -----------------------------------------------------------------------
    def _get_action(self, agent: str, action: np.ndarray):
        # ← MARL 2-legs : cap ET altitude commandés vers la cible ACTIVE
        # (le waypoint individuel tant que leg==1, puis le centre/vertiport
        #  une fois le waypoint atteint — voir _current_target)
        self._control_effort[agent] += float(np.sum(np.abs(action)))
        acid = self._acid(agent)
        idx  = bs.traf.id2idx(acid)
        if idx == -1:
            return
        wlat, wlon, walt = self._current_target(agent)
        qdr_to_waypoint, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            wlat, wlon,
        )
        hdg_cmd = (qdr_to_waypoint + action[0] * 20.0) % 360.0
        bs.stack.stack(f"{acid} HDG {hdg_cmd:.2f}")
        # Altitude cible = waypoint de CET agent
        # Plafond élargi pour supporter jusqu'à 8 agents avec AGENT_ALT_STEP=100ft
        # (BASE_ALT=98 + 7*100 = 798ft pour le dernier agent en scalabilité zero-shot)
        target_alt_cmd = np.clip(walt + action[1] * 30.0, 20.0, BASE_ALT + 8 * AGENT_ALT_STEP)
        bs.stack.stack(f"{acid} ALT {target_alt_cmd:.1f}")

    # -----------------------------------------------------------------------
    # CONFLICT DETECTION
    # -----------------------------------------------------------------------
    def _check_conflict(self, idx: int) -> bool:
        cd = bs.traf.cd
        for pair in cd.confpairs_unique:
            if idx in pair:
                return True
        return False

    def _get_min_tcpa(self, idx: int):
        cd = bs.traf.cd
        min_tcpa = None
        for i, pair in enumerate(cd.confpairs):
            if idx in pair:
                if cd.tcpa is not None and len(cd.tcpa) > i:
                    t = cd.tcpa[i]
                    min_tcpa = t if min_tcpa is None else min(min_tcpa, t)
        return min_tcpa

    # -----------------------------------------------------------------------
    # INTRUSION / CRASH
    # -----------------------------------------------------------------------
    def _check_intrusion(self, agent: str, idx: int) -> bool:
        alt = self._alt_ft(idx)
        intrusion = False
        for other in self.possible_agents:
            if other == agent:
                continue
            oidx = bs.traf.id2idx(self._acid(other))
            if oidx == -1:
                continue
            _, d   = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[oidx], bs.traf.lon[oidx],
            )
            h_dist = d * 1852.0
            v_dist = abs(alt - self._alt_ft(oidx))
            if h_dist < HORIZONTAL_SEP and v_dist < VERTICAL_SEP:
                intrusion = True
        for i in range(MAX_INTRUDERS):
            int_idx = bs.traf.id2idx(f"INT{i}")
            if int_idx == -1:
                continue
            _, d   = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx],
            )
            h_dist = d * 1852.0
            v_dist = abs(alt - self._alt_ft(int_idx))
            if h_dist < HORIZONTAL_SEP and v_dist < VERTICAL_SEP:
                if not self.intruder_intruding[i]:
                    self._total_intrusions[agent] = self._total_intrusions.get(agent, 0) + 1
                    self.intruder_intruding[i] = True
                intrusion = True
            else:
                self.intruder_intruding[i] = False
        return intrusion

    def _check_crash(self, idx: int) -> bool:
        alt = float(bs.traf.alt[idx])
        for other in self.possible_agents:
            oidx = bs.traf.id2idx(self._acid(other))
            if oidx == -1 or oidx == idx:
                continue
            _, d = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[oidx], bs.traf.lon[oidx],
            )
            if d * 1852.0 < 20.0 and abs(alt - float(bs.traf.alt[oidx])) < 5.0:
                return True
        for i in range(MAX_INTRUDERS):
            int_idx = bs.traf.id2idx(f"INT{i}")
            if int_idx == -1:
                continue
            _, d = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx],
            )
            if d * 1852.0 < 20.0 and abs(alt - float(bs.traf.alt[int_idx])) < 5.0:
                return True
        return False

    # -----------------------------------------------------------------------
    # REWARD
    # -----------------------------------------------------------------------
    def _get_reward(self, agent: str):
        if agent in self._done_agents:
            return 0.0, True

        acid = self._acid(agent)
        idx  = bs.traf.id2idx(acid)
        if idx == -1:
            return CRASH_PENALTY, True

        reward     = 0.0
        terminated = False
        alt        = self._alt_ft(idx)

        reward -= 0.1

        # ← MARL 2-legs : progression mesurée vers la cible ACTIVE
        # (waypoint tant que leg==1, centre une fois le waypoint atteint)
        tlat, tlon, talt = self._current_target(agent)
        _, dist_nm = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            tlat, tlon,
        )
        dist_m  = dist_nm * 1852.0
        reward += 0.3 * (self._prev_dist[agent] - dist_m)  # renforcé
        self._prev_dist[agent] = dist_m

        qdr_to_target, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            tlat, tlon,
        )
        angle_diff = np.deg2rad(((qdr_to_target - bs.traf.hdg[idx] + 180) % 360) - 180)
        reward += 0.4 * math.cos(float(angle_diff))  # renforcé

        # Altitude shaping vers la cible ACTIVE (walt en leg 1, TARGET_ALT en leg 2)
        reward -= 0.02 * abs(alt - talt)

        if self._check_conflict(idx):
            reward -= 1.0
            tcpa = self._get_min_tcpa(idx)
            if tcpa is not None and tcpa < 30.0:
                reward -= 3.0 * (1.0 - tcpa / 30.0)

        min_sep_margin = 1.0
        for other in self.possible_agents:
            if other == agent:
                continue
            oidx = bs.traf.id2idx(self._acid(other))
            if oidx == -1:
                continue
            _, d   = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[oidx], bs.traf.lon[oidx],
            )
            h_dist = d * 1852.0
            v_dist = abs(alt - self._alt_ft(oidx))
            sep    = min(h_dist / HORIZONTAL_SEP, v_dist / VERTICAL_SEP)
            min_sep_margin = min(min_sep_margin, sep)

        for i in range(MAX_INTRUDERS):
            int_idx = bs.traf.id2idx(f"INT{i}")
            if int_idx == -1:
                continue
            _, d   = geo.kwikqdrdist(
                bs.traf.lat[idx],  bs.traf.lon[idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx],
            )
            h_dist = d * 1852.0
            v_dist = abs(alt - self._alt_ft(int_idx))
            sep    = min(h_dist / HORIZONTAL_SEP, v_dist / VERTICAL_SEP)
            min_sep_margin = min(min_sep_margin, sep)

        if min_sep_margin < 1.0:
            reward -= 5.0 * (1.0 - min_sep_margin)  # réduit

        if self._check_intrusion(agent, idx):
            reward -= 3.0  # réduit pour ne pas bloquer la navigation
            self._intrusion_steps[agent]  += 1
            self._intrusion_time[agent]   += 1
            # cumulative penalty removed: caused premature termination
            if self._intrusion_steps[agent] >= INTRUSION_PERSISTENCE_STEPS:
                terminated = True
        else:
            self._intrusion_steps[agent] = 0

        if self._check_crash(idx):
            reward -= 250.0
            terminated = True
            self._episode_crash[agent] = True

        # ← FIX : un crash annule tout succès possible ce même step. Sans ce
        # garde-fou, un agent qui crashe PRÈS du vertiport (probable pendant
        # la fusion leg2, exactement là où les crashs sont les plus fréquents)
        # pouvait aussi satisfaire la condition de succès dans le même step,
        # enregistrant crash=True ET mission_success=True simultanément
        # (observé : SR=100% et crash=2.5% dans le même résultat d'éval,
        # mathématiquement impossible si les deux sont mutuellement exclusifs).
        if not self._episode_crash[agent]:
            # ← MARL 2-legs : leg 1 → waypoint atteint (bonus intermédiaire, PAS terminal)
            if self._leg.get(agent, 1) == 1:
                wp = self._waypoints.get(agent)
                if wp is not None:
                    wlat_, wlon_, walt_ = wp
                    if dist_m < TARGET_RADIUS and abs(alt - walt_) < 80.0:
                        reward += WAYPOINT_BONUS
                        self._leg[agent] = 2
                        self._waypoint_reached[agent] = True
                        # Repartir sur une base propre pour le reward de progression
                        # vers le centre (évite un saut artificiel de _prev_dist)
                        _, d_center_nm = geo.kwikqdrdist(
                            bs.traf.lat[idx], bs.traf.lon[idx],
                            CENTER_LAT, CENTER_LON,
                        )
                        self._prev_dist[agent] = d_center_nm * 1852.0

            # ← MARL 2-legs : leg 2 → centre atteint APRÈS le waypoint = succès final
            elif self._leg.get(agent, 1) == 2:
                if dist_m < TARGET_RADIUS and abs(alt - talt) < 80.0:
                    reward += MISSION_SUCCESS_BONUS + 200.0  # bonus fort, cohérent avec l'ancien 500.0
                    terminated = True
                    self._episode_success[agent] = True

        return reward, terminated

    # -----------------------------------------------------------------------
    # INFO
    # -----------------------------------------------------------------------
    def _get_info(self, agent: str) -> dict:
        sc = max(self.step_count, 1)
        return {
            "episode_reward":    round(self._total_reward[agent], 2),
            "episode_length":    self.step_count,
            "intrusions":        self._total_intrusions.get(agent, 0),
            "intrusion_rate":    self._total_intrusions.get(agent, 0) / sc,
            "intrusion_duration":self._intrusion_time[agent],
            "crash":             int(self._episode_crash[agent]),
            "mission_success":   int(self._episode_success[agent]),
            "waypoint_reached":  int(self._waypoint_reached.get(agent, False)),
            "mission_leg":       self._leg.get(agent, 1),
            "control_effort":    round(self._control_effort[agent], 3),
            "in_air":            self.aircraftnumber,
        }

    # -----------------------------------------------------------------------
    # HELPER
    # -----------------------------------------------------------------------
    def _dist_to_target(self, agent: str) -> float:
        idx = bs.traf.id2idx(self._acid(agent))
        if idx == -1:
            return 0.0
        tlat, tlon, _ = self._current_target(agent)
        _, d = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            tlat, tlon,
        )
        return d * 1852.0

    def _acid(self, agent: str) -> str:
        return agent.upper().replace("_", "")

    # -----------------------------------------------------------------------
    # RENDER
    # -----------------------------------------------------------------------
    AGENT_COLORS = [
        (0,   180, 255),
        (255, 120,   0),
        (180,   0, 255),
        (0,   255, 160),
    ]

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("UTM MARL")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((20, 20, 30))
        cx, cy  = self.window_width // 2, self.window_height // 2
        px_r    = min(self.window_width, self.window_height) / 2 - 40
        SCALE   = px_r / (CONTROL_RADIUS_KM * 1000)
        font    = pygame.font.SysFont("monospace", 11)

        def w2s(x, y):
            return int(cx + x * SCALE), int(cy - y * SCALE)

        pygame.draw.circle(canvas, (50, 70, 100), (cx, cy), int(px_r), 2)
        pygame.draw.circle(canvas, (0, 220, 80), (cx, cy), 8)

        for i, agent in enumerate(self.possible_agents):
            idx = bs.traf.id2idx(self._acid(agent))
            if idx == -1:
                continue
            qdr_a, d_nm = geo.kwikqdrdist(
                TARGET_LAT, TARGET_LON,
                bs.traf.lat[idx], bs.traf.lon[idx],
            )
            d_m   = d_nm * 1852.0
            ax    = d_m * math.sin(math.radians(qdr_a))
            ay    = d_m * math.cos(math.radians(qdr_a))
            asc   = w2s(ax, ay)
            col   = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            hdg   = float(bs.traf.hdg[idx])
            theta = math.radians(hdg)
            sz    = 12
            nose  = (asc[0] + sz*math.sin(theta),       asc[1] - sz*math.cos(theta))
            left  = (asc[0] + sz*.6*math.sin(theta+2.5), asc[1] - sz*.6*math.cos(theta+2.5))
            right = (asc[0] + sz*.6*math.sin(theta-2.5), asc[1] - sz*.6*math.cos(theta-2.5))
            pygame.draw.polygon(canvas, col, [nose, left, right])
            lbl = font.render(f"{agent} {int(self._alt_ft(idx))}ft", True, col)
            canvas.blit(lbl, (asc[0]+14, asc[1]-8))
            pygame.draw.circle(canvas, col, asc, int(HORIZONTAL_SEP*SCALE), 1)

        for i in range(MAX_INTRUDERS):
            int_idx = bs.traf.id2idx(f"INT{i}")
            if int_idx == -1:
                continue
            qdr_i, dist = geo.kwikqdrdist(
                TARGET_LAT, TARGET_LON,
                bs.traf.lat[int_idx], bs.traf.lon[int_idx],
            )
            d_m = dist * 1852.0
            x   = d_m * math.sin(math.radians(qdr_i))
            y   = d_m * math.cos(math.radians(qdr_i))
            pygame.draw.circle(canvas, (255, 80, 80), w2s(x, y), 4)

        hud = [f"Step:{self.step_count}/{MAX_STEPS}", f"Active:{len(self.agents)}/{self.n_agents}"]
        for j, line in enumerate(hud):
            canvas.blit(font.render(line, True, (200,200,200)), (10, 10+j*16))

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(30)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


# ===========================================================================
# Sanity check
# ===========================================================================
if __name__ == "__main__":
    print("=== UTM MARL v2 — Sanity Check ===")
    env = UTMMultiAgent(n_agents=4, render_mode=None)
    obs, infos = env.reset(seed=42)
    print(f"Agents : {env.agents}")
    print(f"Obs keys : {list(obs['uav_0'].keys())}")
    steps = 0
    while env.agents and steps < 20:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rews, terms, truncs, infos = env.step(actions)
        steps += 1
    print(f"Done in {steps} steps.")
    env.close()
