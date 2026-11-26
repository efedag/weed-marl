"""Drone + yer robotu ekipleri için yabancı ot ilaçlama ortamı.

PettingZoo ``ParallelEnv`` arayüzünü uygular. Gözlem, state, eylem ve
config alanlarının ayrıntılı düzeni ``docs/env_interface.md`` içindedir.

Hızlı deneme:
    python -m sim.weed_env
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from marl.reward import RewardConfig, calculate_reward
from sim.comms import Message, make_channel
from sim.detection import PerfectDetector

# Hücre kodları (detector/convert_to_grid64.py ile aynı)
SOIL, CROP, WEED = 0, 1, 2

# Eylemler (herkes için aynı Discrete(6) uzayı)
STAY, UP, DOWN, LEFT, RIGHT, SPRAY = range(6)
N_ACTIONS = 6
ACTION_NAMES = ("STAY", "UP", "DOWN", "LEFT", "RIGHT", "SPRAY")
MOVES = np.array(
    [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int64
)

# Gözlem düzeni
N_SCALAR_FEATURES = 6  # konum(2) + batarya(1) + tip one-hot(2) + mesaj yaşı(1)
N_WINDOW_CHANNELS = 4  # bilinen ot, ilaçlanmış, keşfedilmemiş, diğer ajanlar
N_REPORT_FEATURES = 3  # göreli satır, göreli sütun, geçerlilik maskesi
# state'teki tarla kanalları: kalan ot, ilaçlanmış, drone'larca keşfedilmiş
N_STATE_MAP_CHANNELS = 3
N_STATE_AGENT_FEATURES = 3  # konum(2) + batarya(1)


@dataclass
class WeedReport:
    """Drone'un bir adımda robotlara gönderdiği ot raporu."""

    bounds: tuple[int, int, int, int]  # taranan blok: (r0, r1, c0, c1), yarı açık
    cells: np.ndarray  # (n, 2) ot olarak algılanan hücreler (satır, sütun)
    confidence: np.ndarray  # (n,) dedektör güveni


@dataclass
class WeedEnvConfig:
    """Ortamın tüm ayarları. İleride YAML'dan ``from_dict`` ile okunacak."""

    # Tarla
    grid_path: str | None = None  # .npy yolu; None ise rastgele tarla üretilir
    grid_size: int = 16  # rastgele tarla kenarı
    weed_density: float = 0.1  # rastgele tarlada ot hücresi oranı
    crop_density: float = 0.2  # rastgele tarlada ürün hücresi oranı
    crop_size: int | None = None  # .npy tarlayı crop_size x crop_size kırp
    crop_origin: tuple[int, int] | None = None  # None: kırpma köşesi rastgele

    # Ajanlar ve bölüm
    n_drones: int = 2
    n_robots: int = 2
    max_steps: int = 200

    # Batarya
    drone_battery: float = 200.0
    robot_battery: float = 200.0
    drone_move_cost: float = 1.0
    robot_move_cost: float = 1.0
    stay_cost: float = 0.0
    spray_cost: float = 1.0

    # Algılama ve gözlem
    detection_radius: int = 2  # drone'un algıladığı kare: (2r+1) x (2r+1)
    obs_window: int = 5  # yerel pencere k (tek sayı)
    n_reports: int = 5  # robot gözlemindeki en yakın K ot raporu
    state_map_size: int = 16  # state'teki indirgenmiş harita kenarı

    # İletişim
    channel: str = "perfect"
    channel_params: dict = field(default_factory=dict)

    # Ödül (marl/reward.py)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "WeedEnvConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Bilinmeyen config alanları: {sorted(unknown)}")
        data = dict(data)
        if isinstance(data.get("reward"), dict):
            data["reward"] = RewardConfig(**data["reward"])
        if data.get("crop_origin") is not None:
            data["crop_origin"] = tuple(data["crop_origin"])
        return cls(**data)

    def validate(self) -> None:
        def check(condition, message):
            if not condition:
                raise ValueError(message)

        check(1 <= self.n_drones <= 4, "n_drones 1-4 arasında olmalı")
        check(1 <= self.n_robots <= 4, "n_robots 1-4 arasında olmalı")
        check(self.max_steps >= 1, "max_steps en az 1 olmalı")
        check(self.grid_size >= 4, "grid_size en az 4 olmalı")
        check(
            0 <= self.weed_density <= 1
            and 0 <= self.crop_density <= 1
            and self.weed_density + self.crop_density <= 1,
            "weed_density ve crop_density [0, 1] aralığında ve toplamı en fazla 1 olmalı",
        )
        check(self.crop_size is None or self.crop_size >= 4, "crop_size en az 4 olmalı")
        check(self.drone_battery > 0 and self.robot_battery > 0, "batarya pozitif olmalı")
        check(
            min(self.drone_move_cost, self.robot_move_cost, self.stay_cost, self.spray_cost)
            >= 0,
            "maliyetler negatif olamaz",
        )
        check(self.detection_radius >= 0, "detection_radius negatif olamaz")
        check(self.obs_window >= 1 and self.obs_window % 2 == 1, "obs_window pozitif tek sayı olmalı")
        check(self.n_reports >= 1, "n_reports en az 1 olmalı")
        check(self.state_map_size >= 1, "state_map_size en az 1 olmalı")


def observation_size(obs_window: int, n_reports: int) -> int:
    """Gözlem vektörünün uzunluğu: 6 + 4*k*k + 3*K."""
    return (
        N_SCALAR_FEATURES
        + N_WINDOW_CHANNELS * obs_window * obs_window
        + N_REPORT_FEATURES * n_reports
    )


def state_size(n_agents: int, map_size: int) -> int:
    """state vektörünün uzunluğu: 3*(N+M) + 3*S*S + 1."""
    return (
        N_STATE_AGENT_FEATURES * n_agents
        + N_STATE_MAP_CHANNELS * map_size * map_size
        + 1
    )


def _block_mean(values: np.ndarray, size: int) -> np.ndarray:
    """(H, W) diziyi size x size bloklarının ortalamasına indirger."""
    h, w = values.shape
    row_edges = np.linspace(0, h, size + 1).astype(np.int64)
    col_edges = np.linspace(0, w, size + 1).astype(np.int64)
    sums = np.add.reduceat(
        np.add.reduceat(values.astype(np.float32), row_edges[:-1], axis=0),
        col_edges[:-1],
        axis=1,
    )
    areas = np.outer(np.diff(row_edges), np.diff(col_edges))
    return sums / areas


class WeedEnv(ParallelEnv):
    """Drone'lar otları algılayıp robotlara bildirir, robotlar ilaçlar."""

    metadata = {
        "name": "weed_env_v0",
        "render_modes": ["ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, config=None, detector=None, render_mode=None):
        if config is None:
            config = WeedEnvConfig()
        elif isinstance(config, dict):
            config = WeedEnvConfig.from_dict(config)
        config.validate()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Desteklenmeyen render_mode: {render_mode}")

        self.config = config
        self.detector = detector if detector is not None else PerfectDetector()
        self.render_mode = render_mode

        self.drones = [f"drone_{i}" for i in range(config.n_drones)]
        self.robots = [f"robot_{i}" for i in range(config.n_robots)]
        self.possible_agents = self.drones + self.robots
        self.agents = []
        self._index = {name: i for i, name in enumerate(self.possible_agents)}
        self._robot_index = {name: j for j, name in enumerate(self.robots)}

        n = len(self.possible_agents)
        self._is_drone = np.array([True] * config.n_drones + [False] * config.n_robots)
        self._capacity = np.array(
            [config.drone_battery] * config.n_drones + [config.robot_battery] * config.n_robots,
            dtype=np.float64,
        )
        self._move_cost = np.array(
            [config.drone_move_cost] * config.n_drones
            + [config.robot_move_cost] * config.n_robots,
            dtype=np.float64,
        )

        self._base_field = self._load_field(config.grid_path) if config.grid_path else None
        if self._base_field is not None:
            if config.crop_size is not None:
                if config.crop_size > min(self._base_field.shape):
                    raise ValueError("crop_size tarladan büyük olamaz")
                self.height = self.width = config.crop_size
            else:
                self.height, self.width = self._base_field.shape
        else:
            self.height = self.width = config.grid_size
        self._map_size = min(config.state_map_size, self.height, self.width)

        self.obs_size = observation_size(config.obs_window, config.n_reports)
        self.state_size = state_size(n, self._map_size)
        obs_space = spaces.Box(-1.0, 1.0, shape=(self.obs_size,), dtype=np.float32)
        self._observation_spaces = {a: obs_space for a in self.possible_agents}
        self._action_spaces = {a: spaces.Discrete(N_ACTIONS) for a in self.possible_agents}
        self.state_space = spaces.Box(0.0, 1.0, shape=(self.state_size,), dtype=np.float32)

        self.channel = make_channel(config.channel, config.channel_params)
        self._rng = None
        self._last_obs = None

    # ------------------------------------------------------------------ uzaylar
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ------------------------------------------------------------------ tarla
    @staticmethod
    def _load_field(path) -> np.ndarray:
        grid = np.load(Path(path), allow_pickle=False)
        if grid.ndim != 2:
            raise ValueError(f"Tarla 2 boyutlu olmalı, gelen şekil: {grid.shape}")
        if not np.isin(grid, (SOIL, CROP, WEED)).all():
            raise ValueError("Tarla yalnızca 0 (toprak), 1 (ürün), 2 (ot) içermeli")
        return grid.astype(np.uint8)

    def _make_field(self, rng: np.random.Generator) -> np.ndarray:
        c = self.config
        if self._base_field is not None:
            grid = self._base_field
            if c.crop_size is not None:
                size = c.crop_size
                if c.crop_origin is None:
                    r0 = int(rng.integers(0, grid.shape[0] - size + 1))
                    c0 = int(rng.integers(0, grid.shape[1] - size + 1))
                else:
                    r0, c0 = c.crop_origin
                    if not (0 <= r0 <= grid.shape[0] - size and 0 <= c0 <= grid.shape[1] - size):
                        raise ValueError("crop_origin tarla dışına taşıyor")
                grid = grid[r0 : r0 + size, c0 : c0 + size]
            return grid.copy()

        size = c.grid_size
        u = rng.random((size, size))
        grid = np.full((size, size), SOIL, dtype=np.uint8)
        grid[u < c.weed_density + c.crop_density] = CROP
        grid[u < c.weed_density] = WEED
        if not (grid == WEED).any():
            grid[rng.integers(size), rng.integers(size)] = WEED
        return grid

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        field_rng, start_rng, self._detector_rng, channel_rng = self._rng.spawn(4)

        c = self.config
        n, nd, nr = len(self.possible_agents), c.n_drones, c.n_robots
        self.field = self._make_field(field_rng)
        h, w = self.field.shape
        self.weed_mask = self.field == WEED
        self.n_weeds = int(self.weed_mask.sum())
        self.sprayed = np.zeros((h, w), dtype=bool)

        self.positions = np.stack(
            [start_rng.integers(0, h, n), start_rng.integers(0, w, n)], axis=1
        ).astype(np.int64)
        self.battery = self._capacity.copy()
        self.path_length = np.zeros(n, dtype=np.int64)

        # Bilgi durumları
        self.drone_explored = np.zeros((nd, h, w), dtype=bool)
        self.drone_weed = np.zeros((nd, h, w), dtype=bool)
        self.drone_confidence = np.zeros((nd, h, w), dtype=np.float32)
        self.robot_explored = np.zeros((nr, h, w), dtype=bool)
        self.robot_known_weed = np.zeros((nr, h, w), dtype=bool)
        self.seen_sprayed = np.zeros((n, h, w), dtype=bool)
        self.steps_since_delivery = np.zeros(nr, dtype=np.int64)

        # Sayaçlar
        self.step_count = 0
        self.weeds_sprayed = 0
        self.unnecessary_sprays = 0
        self.completion_step = None

        self.channel.reset(channel_rng)
        self.agents = list(self.possible_agents)

        self._sense_and_communicate()
        observations = self._observations()
        masks = self.action_masks()
        infos = {a: {"action_mask": masks[self._index[a]]} for a in self.agents}
        return observations, infos

    # ------------------------------------------------------------------ step
    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        c = self.config
        masks = self.action_masks()
        acts = np.array(
            [int(actions.get(a, STAY)) for a in self.possible_agents], dtype=np.int64
        )
        if ((acts < 0) | (acts >= N_ACTIONS)).any():
            raise ValueError(f"Geçersiz eylem: {actions}")
        # Maskeyle kapalı eylem STAY sayılır
        acts = np.where(masks[np.arange(len(acts)), acts] == 1, acts, STAY)

        # İlaçlama: robot bulunduğu hücreyi ilaçlar. Maske sayesinde yalnızca
        # yeterli bataryası olan robotlar buraya gelir. Daha önce ilaçlanmış
        # hücre (aynı adımda başka bir robotun ilaçladığı dahil) tekrar sayılmaz.
        spraying = acts == SPRAY
        weeds_step, unnecessary_step = 0, 0
        for a in np.flatnonzero(spraying):
            row, col = self.positions[a]
            if self.sprayed[row, col]:
                continue
            self.sprayed[row, col] = True
            if self.weed_mask[row, col]:
                weeds_step += 1
            else:
                unnecessary_step += 1
        self.weeds_sprayed += weeds_step
        self.unnecessary_sprays += unnecessary_step

        # Hareket: ızgara dışına çıkan hareket STAY olur
        new_positions = np.clip(
            self.positions + MOVES[acts], 0, [self.height - 1, self.width - 1]
        )
        moved = (new_positions != self.positions).any(axis=1)
        cost = np.where(moved, self._move_cost, np.where(spraying, c.spray_cost, c.stay_cost))
        self.battery = np.maximum(self.battery - cost, 0.0)
        self.path_length += moved
        self.positions = new_positions
        self.step_count += 1

        self._sense_and_communicate()

        # Takım ödülü adım başına bir kez hesaplanır, her ajana aynı değer verilir
        team_reward = float(calculate_reward(weeds_step, unnecessary_step, c.reward))

        all_sprayed = not (self.weed_mask & ~self.sprayed).any()
        if all_sprayed and self.completion_step is None:
            self.completion_step = self.step_count
        robots_depleted = bool(self._depleted()[~self._is_drone].all())
        terminated = all_sprayed or robots_depleted
        truncated = self.step_count >= c.max_steps

        observations = self._observations()
        new_masks = self.action_masks()
        rewards = {a: team_reward for a in self.agents}
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {
            a: {"action_mask": new_masks[self._index[a]], "team_reward": team_reward}
            for a in self.agents
        }

        if terminated or truncated:
            summary = self._episode_summary(terminated, truncated, all_sprayed, robots_depleted)
            for a in self.agents:
                infos[a]["episode"] = summary
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ kurallar
    def _depleted(self) -> np.ndarray:
        """STAY dışında hiçbir eylemi yapamayacak kadar bataryası azalmış ajanlar."""
        cheapest = np.where(
            self._is_drone, self._move_cost, np.minimum(self._move_cost, self.config.spray_cost)
        )
        return self.battery < cheapest

    def action_masks(self) -> np.ndarray:
        """(N+M, 6) int8 eylem maskesi, possible_agents sırasıyla."""
        masks = np.zeros((len(self.possible_agents), N_ACTIONS), dtype=np.int8)
        masks[:, STAY] = 1
        masks[:, UP : RIGHT + 1] = (self.battery >= self._move_cost)[:, None]
        masks[:, SPRAY] = (~self._is_drone) & (self.battery >= self.config.spray_cost)
        return masks

    # ------------------------------------------------------------------ algılama ve iletişim
    def _sense_and_communicate(self) -> None:
        r = self.config.detection_radius
        h, w = self.height, self.width
        messages = []
        for i, drone in enumerate(self.drones):
            row, col = self.positions[i]
            r0, r1 = max(0, row - r), min(h, row + r + 1)
            c0, c1 = max(0, col - r), min(w, col + r + 1)
            labels, confidence = self.detector.detect(
                self.field[r0:r1, c0:c1], self._detector_rng
            )
            weed = labels == WEED
            self.drone_explored[i, r0:r1, c0:c1] = True
            self.drone_weed[i, r0:r1, c0:c1] = weed
            self.drone_confidence[i, r0:r1, c0:c1] = confidence
            rows, cols = np.nonzero(weed)
            report = WeedReport(
                bounds=(int(r0), int(r1), int(c0), int(c1)),
                cells=np.stack([rows + r0, cols + c0], axis=1),
                confidence=confidence[rows, cols],
            )
            messages.extend(Message(drone, robot, self.step_count, report) for robot in self.robots)

        received = np.zeros(len(self.robots), dtype=bool)
        for message in self.channel.transmit(messages):
            j = self._robot_index[message.receiver]
            report = message.payload
            r0, r1, c0, c1 = report.bounds
            self.robot_explored[j, r0:r1, c0:c1] = True
            self.robot_known_weed[j, r0:r1, c0:c1] = False
            self.robot_known_weed[j, report.cells[:, 0], report.cells[:, 1]] = True
            received[j] = True
        self.steps_since_delivery = np.where(received, 0, self.steps_since_delivery + 1)

        # Her ajan kendi yerel penceresindeki ilaçlanmış hücreleri görür
        half = self.config.obs_window // 2
        for a, (row, col) in enumerate(self.positions):
            r0, r1 = max(0, row - half), min(h, row + half + 1)
            c0, c1 = max(0, col - half), min(w, col + half + 1)
            self.seen_sprayed[a, r0:r1, c0:c1] = self.sprayed[r0:r1, c0:c1]

    # ------------------------------------------------------------------ gözlem
    def _window(self, values: np.ndarray, row: int, col: int) -> np.ndarray:
        """Ajan merkezli k x k pencere; ızgara dışı hücreler 0."""
        k = self.config.obs_window
        half = k // 2
        out = np.zeros((k, k), dtype=np.float32)
        r0, r1 = max(0, row - half), min(self.height, row + half + 1)
        c0, c1 = max(0, col - half), min(self.width, col + half + 1)
        out[r0 - (row - half) : r1 - (row - half), c0 - (col - half) : c1 - (col - half)] = values[
            r0:r1, c0:c1
        ]
        return out

    def _observe(self, a: int, occupancy: np.ndarray) -> np.ndarray:
        c = self.config
        k, n_reports = c.obs_window, c.n_reports
        row, col = (int(v) for v in self.positions[a])
        obs = np.zeros(self.obs_size, dtype=np.float32)

        obs[0] = row / (self.height - 1)
        obs[1] = col / (self.width - 1)
        obs[2] = self.battery[a] / self._capacity[a]
        if self._is_drone[a]:
            obs[3] = 1.0
            known_weed = self.drone_weed[a]
            explored = self.drone_explored[a]
        else:
            obs[4] = 1.0
            j = a - c.n_drones
            obs[5] = min(self.steps_since_delivery[j], c.max_steps) / c.max_steps
            known_weed = self.robot_known_weed[j]
            explored = self.robot_explored[j]

        known_weed = known_weed & ~self.seen_sprayed[a]
        others = self._window(occupancy, row, col)
        others[k // 2, k // 2] -= 1
        window = np.stack(
            [
                self._window(known_weed, row, col),
                self._window(self.seen_sprayed[a], row, col),
                self._window(~explored, row, col),
                others > 0,
            ]
        )
        start = N_SCALAR_FEATURES
        obs[start : start + window.size] = window.ravel()

        if not self._is_drone[a]:
            rows, cols = np.nonzero(known_weed)
            if rows.size:
                distance = np.abs(rows - row) + np.abs(cols - col)
                nearest = np.lexsort((cols, rows, distance))[:n_reports]
                reports = np.zeros((n_reports, N_REPORT_FEATURES), dtype=np.float32)
                reports[: nearest.size, 0] = (rows[nearest] - row) / (self.height - 1)
                reports[: nearest.size, 1] = (cols[nearest] - col) / (self.width - 1)
                reports[: nearest.size, 2] = 1.0
                obs[start + window.size :] = reports.ravel()
        return obs

    def _observations(self) -> dict:
        occupancy = np.zeros((self.height, self.width), dtype=np.float32)
        np.add.at(occupancy, (self.positions[:, 0], self.positions[:, 1]), 1.0)
        self._last_obs = [self._observe(a, occupancy) for a in range(len(self.possible_agents))]
        return {a: self._last_obs[self._index[a]] for a in self.agents}

    # ------------------------------------------------------------------ merkezi state
    def state(self) -> np.ndarray:
        """Merkezi kritik için global durum; ajan gözlemlerine verilmez."""
        if self._last_obs is None:
            raise RuntimeError("state() için önce reset() çağrılmalı")
        agent_part = np.column_stack(
            [
                self.positions[:, 0] / (self.height - 1),
                self.positions[:, 1] / (self.width - 1),
                self.battery / self._capacity,
            ]
        ).ravel()
        maps = [
            _block_mean(self.weed_mask & ~self.sprayed, self._map_size),
            _block_mean(self.sprayed, self._map_size),
            _block_mean(self.drone_explored.any(axis=0), self._map_size),
        ]
        progress = np.array([self.step_count / self.config.max_steps])
        return np.concatenate([agent_part, *[m.ravel() for m in maps], progress]).astype(
            np.float32
        )

    # ------------------------------------------------------------------ bölüm özeti
    def _episode_summary(self, terminated, truncated, all_sprayed, robots_depleted) -> dict:
        reasons = []
        if all_sprayed:
            reasons.append("all_weeds_sprayed")
        if robots_depleted:
            reasons.append("robots_depleted")
        if truncated:
            reasons.append("max_steps")
        return {
            "steps": self.step_count,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_reasons": reasons,
            "messages_sent": self.channel.sent,
            "messages_lost": self.channel.lost,
            "weeds_total": self.n_weeds,
            "weeds_sprayed": self.weeds_sprayed,
            "unnecessary_sprays": self.unnecessary_sprays,
            "weed_sprayed_ratio": self.weeds_sprayed / self.n_weeds if self.n_weeds else 1.0,
            "sprayed_area_ratio": float(self.sprayed.mean()),
            "path_length": {a: int(self.path_length[i]) for a, i in self._index.items()},
            "path_length_total": int(self.path_length.sum()),
            "completion_step": self.completion_step,
        }

    # ------------------------------------------------------------------ EPyMARL yardımcıları
    def get_obs(self) -> list:
        return list(self._last_obs)

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self._last_obs[agent_id]

    def get_obs_size(self) -> int:
        return self.obs_size

    def get_state(self) -> np.ndarray:
        return self.state()

    def get_state_size(self) -> int:
        return self.state_size

    def get_avail_actions(self) -> list:
        return [m.tolist() for m in self.action_masks()]

    def get_avail_agent_actions(self, agent_id: int) -> list:
        return self.action_masks()[agent_id].tolist()

    def get_total_actions(self) -> int:
        return N_ACTIONS

    def get_env_info(self) -> dict:
        return {
            "state_shape": self.state_size,
            "obs_shape": self.obs_size,
            "n_actions": N_ACTIONS,
            "n_agents": len(self.possible_agents),
            "episode_limit": self.config.max_steps,
        }

    # ------------------------------------------------------------------ çizim
    def render(self):
        mode = self.render_mode or "ansi"
        if mode == "rgb_array":
            return self._render_rgb()
        return self._render_ansi()

    def _render_ansi(self) -> str:
        chars = np.array([".", "c", "W"])[self.field]
        chars = chars.astype("<U1")
        chars[self.sprayed & self.weed_mask] = "x"
        chars[self.sprayed & ~self.weed_mask] = "s"
        for a, (row, col) in enumerate(self.positions):
            symbol = "D" if self._is_drone[a] else "R"
            chars[row, col] = "*" if chars[row, col] in ("D", "R", "*") else symbol
        lines = ["".join(line) for line in chars]
        # Windows konsollarında bozulmaması için açıklama satırı ASCII
        lines.append(
            f"adim {self.step_count}/{self.config.max_steps} | "
            f"ilaclanan ot {self.weeds_sprayed}/{self.n_weeds} | "
            "D drone, R robot, W ot, x ilaclanmis ot, s gereksiz ilaclama, c urun, * cakisma"
        )
        return "\n".join(lines)

    def _render_rgb(self, scale: int = 8) -> np.ndarray:
        palette = np.array([[110, 80, 50], [60, 160, 70], [220, 50, 50]], dtype=np.uint8)
        image = palette[self.field].copy()
        image[self.sprayed & self.weed_mask] = (240, 200, 40)
        image[self.sprayed & ~self.weed_mask] = (150, 150, 150)
        image[self.positions[self._is_drone, 0], self.positions[self._is_drone, 1]] = (40, 90, 230)
        image[self.positions[~self._is_drone, 0], self.positions[~self._is_drone, 1]] = (20, 20, 20)
        return np.kron(image, np.ones((scale, scale, 1), dtype=np.uint8))

    def close(self):
        pass


def _demo():
    env = WeedEnv(WeedEnvConfig(grid_size=16, max_steps=50))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    while env.agents:
        masks = env.action_masks()
        actions = {
            a: int(rng.choice(np.flatnonzero(masks[env._index[a]]))) for a in env.agents
        }
        _, _, _, _, infos = env.step(actions)
    print(env.render())
    print(next(iter(infos.values()))["episode"])


if __name__ == "__main__":
    _demo()
