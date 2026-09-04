"""Local FastAPI bridge between Cesium and the v10 InfernoEnv."""

import os
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from pyproj import Transformer

from infernotactics.data.config import (
    PROJECT_ROOT as PYTHON_PROJECT_ROOT,
    LEGACY_GRID_META_PATH,
    LEGACY_GRID_STATIC_PATH,
    REAL_DEPOTS_PATH,
)
from infernotactics.operations.environment import (
    TRAINING_IGNITION_POINT, VALIDATION_IGNITION_POINTS,
    InfernoEnv,
)
from infernotactics.domain.resources import RESOURCE_TYPES  # noqa: E402
from infernotactics.policy.models import RelativeInfernoModel
from infernotactics.policy.targets import decode_action  # noqa: E402
from infernotactics.policy import DEFAULT_MAX_DISPATCH_SLOTS, forward_policy  # noqa: E402
from infernotactics.service import (  # noqa: E402
    SessionCapacityError,
    SessionManager,
    SessionNotFoundError,
)
from infernotactics.policy.heuristic import HeuristicPolicy
from infernotactics.training.checkpoints import (  # noqa: E402
    checkpoint_metadata,
    load_checkpoint,
    model_state_from_checkpoint,
)

PYTHON_PROJECT_DIR = Path(PYTHON_PROJECT_ROOT)
STATIC_DATA_DIR = PYTHON_PROJECT_DIR / "data" / "visualization"
CHECKPOINT = os.environ.get(
    "INFERNO_CHECKPOINT",
    str(PYTHON_PROJECT_DIR / "models" / "containment_policy_v10.pt"),
)
WORLD_GRID = os.environ.get("INFERNO_GRID_STATIC", LEGACY_GRID_STATIC_PATH)
WORLD_META = os.environ.get("INFERNO_GRID_META", LEGACY_GRID_META_PATH)
MAX_DISPATCH_SLOTS = int(os.environ.get("INFERNO_MAX_DISPATCH_SLOTS", DEFAULT_MAX_DISPATCH_SLOTS))
MAX_SESSIONS = int(os.environ.get("INFERNO_MAX_SESSIONS", 8))
_forward = forward_policy

app = FastAPI(title="InfernoTactics Simulation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ResetRequest(BaseModel):
    ignition_point: list[int] | None = None
    scenario: str = "single"
    session_id: str = "default"

    @field_validator("ignition_point")
    @classmethod
    def validate_ignition_point(cls, value):
        if value is not None and len(value) != 2:
            raise ValueError("ignition_point must contain [row, column]")
        return value

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, value):
        allowed = {"single", "multi", "anchor", *VALIDATION_IGNITION_POINTS}
        if value not in allowed:
            raise ValueError(f"scenario must be one of {sorted(allowed)}")
        return value


class StepRequest(BaseModel):
    actions: list[list] | None = None
    mode: Literal["manual", "autopilot", "heuristic"] = "manual"
    session_id: str = "default"


class Session:
    def __init__(self, model=None, routing_context=None):
        self.device = torch.device("cpu")
        self.env = InfernoEnv(
            grid_static_path=WORLD_GRID,
            grid_meta_path=WORLD_META,
            seed=9100,
            routing_context=routing_context,
        )
        obs = self.env.reset(seed=9100)
        if model is None:
            self.model = RelativeInfernoModel(obs["grid"].shape[0], len(obs["scalars"]), len(RESOURCE_TYPES), self.env.n_zones).to(self.device)
            if os.path.exists(CHECKPOINT):
                checkpoint = load_checkpoint(CHECKPOINT, map_location=self.device)
                saved_world = checkpoint_metadata(checkpoint).get("world_fingerprint")
                if saved_world is not None and saved_world != self.env.world.fingerprint:
                    raise RuntimeError(
                        "Checkpoint/world mismatch. Configure matching INFERNO_CHECKPOINT, "
                        "INFERNO_GRID_STATIC, and INFERNO_GRID_META paths."
                    )
                self.model.load_state_dict(model_state_from_checkpoint(checkpoint))
            else:
                raise FileNotFoundError(f"Configured checkpoint does not exist: {CHECKPOINT}")
            self.model.eval()
        else:
            self.model = model
        self.heuristic = HeuristicPolicy(self.env)
        self.transformer = Transformer.from_crs(self.env.meta["crs"], "EPSG:4326", always_xy=True)
        self.obs = obs
        self.total_reward = 0.0
        self.events = []
        self.done = False
        self.contained = False
        self.timeout = False
        self.last_buildings_destroyed = 0
        self.total_buildings_destroyed = 0
        self.lock = threading.RLock()
        self.station_points = {
            station["station_id"]: {"lat": station["lat"], "lon": station["lon"]}
            for station in self.env.stations
        }

    def reset(self, request):
        with self.lock:
            return self._reset_unlocked(request)

    def _reset_unlocked(self, request):
        named = {"anchor": TRAINING_IGNITION_POINT, **VALIDATION_IGNITION_POINTS}
        if request.ignition_point:
            row, col = request.ignition_point
            if not 0 <= row < self.env.height or not 0 <= col < self.env.width:
                raise ValueError(
                    f"ignition_point {request.ignition_point} is outside "
                    f"the {self.env.height}x{self.env.width} grid"
                )
            self.obs = self.env.reset(ignition_point=(row, col), seed=9100)
        elif request.scenario in named:
            self.obs = self.env.reset(ignition_point=named[request.scenario], seed=9100)
        else:
            self.obs = self.env.reset(scenario=request.scenario if request.scenario == "multi" else "single", seed=9100)
        self.total_reward = 0.0
        self.events = []
        self.done = False
        self.contained = False
        self.timeout = False
        self.last_buildings_destroyed = 0
        self.total_buildings_destroyed = 0
        return self.state()

    def autopilot_actions(self, mode="autopilot"):
        with self.lock:
            return self._autopilot_actions_unlocked(mode)

    def _autopilot_actions_unlocked(self, mode="autopilot"):
        if mode == "heuristic":
            grid = self.obs["grid"]
            scalars = np.asarray(list(self.obs["scalars"].values()), dtype=np.float32)
            return self.heuristic.decide_actions(grid, scalars)
        local_available = {r: int(self.obs["scalars"][f"{r}_available"]) for r in RESOURCE_TYPES}
        actions = []
        with torch.no_grad():
            logits, _, _, zones, _features = _forward(
                self.model, self.obs, self.env, self.device
            )
            for _ in range(MAX_DISPATCH_SLOTS):
                resource_logits = logits["resource_type"][0].clone()
                mask = torch.tensor([local_available[r] > 0 for r in RESOURCE_TYPES], dtype=torch.bool)
                resource_logits[~mask] = -1e9
                if not bool(mask.any()):
                    break
                ri = int(torch.argmax(resource_logits))
                ti = int(torch.argmax(logits["target"][0, ri]))
                action = decode_action(ri, ti, zones)
                if action is None:
                    break
                actions.append(action)
                local_available[action[0]] -= 1
        return actions

    def advance(self, mode="manual", actions=None):
        """Choose and apply one tick atomically within this session."""
        with self.lock:
            selected = (
                self._autopilot_actions_unlocked(mode)
                if mode in ("autopilot", "heuristic")
                else (actions or [])
            )
            return self._step_unlocked(selected)

    def step(self, actions):
        with self.lock:
            return self._step_unlocked(actions)

    def _step_unlocked(self, actions):
        if self.done:
            return self.state() | {"last_step": None, "done": True}
        self.obs, reward, done, info = self.env.step(actions)
        self.total_reward += reward
        self.done = bool(done)
        self.contained = bool(info["contained"])
        self.timeout = bool(info["timeout"])
        self.last_buildings_destroyed = int(info["buildings_destroyed"])
        self.total_buildings_destroyed += self.last_buildings_destroyed
        event = {"tick": info["tick"], "dispatch": info["dispatch"], "resource_events": info["resource_events"], "buildings_destroyed": info["buildings_destroyed"], "reward": reward}
        self.events.append(event)
        self.events = self.events[-50:]
        return self.state() | {"last_step": event, "done": done}

    def point(self, row, col):
        a, b, c, d, e, f = self.env.meta["transform"]
        x = a * (col + 0.5) + b * (row + 0.5) + c
        y = d * (col + 0.5) + e * (row + 0.5) + f
        lon, lat = self.transformer.transform(x, y)
        return {"lat": lat, "lon": lon}

    def resource_position(self, unit):
        start = self.station_points.get(unit["station_id"], {"lat": 34.0725, "lon": -118.5425})
        target = start
        if unit.get("target_zone") is not None:
            zone = self.env.zones[unit["target_zone"]]
            target = self.point(zone["centroid_row"], zone["centroid_col"])
        if unit["state"] in ("traveling", "preparing") and unit.get("pending_travel_ticks"):
            total = max(1, int(unit["pending_travel_ticks"]))
            remaining = min(total, max(0, int(unit.get("remaining_ticks", total))))
            progress = 1.0 - remaining / total
            return {"lat": start["lat"] + (target["lat"] - start["lat"]) * progress,
                    "lon": start["lon"] + (target["lon"] - start["lon"]) * progress,
                    "height_m": 350 if unit.get("station_id") == "114" else 15}
        return {"lat": target["lat"], "lon": target["lon"], "height_m": 350 if unit.get("station_id") == "114" else 15}

    def state(self):
        with self.lock:
            return self._state_unlocked()

    def _state_unlocked(self):
        fire = self.env.sim.state
        active = np.argwhere(np.isin(fire, (2, 3)))
        fire_cells = [{**self.point(int(r), int(c)), "row": int(r), "col": int(c), "state": int(fire[r, c])} for r, c in active]
        resources = []
        for rtype in RESOURCE_TYPES:
            for index, unit in enumerate(self.env.resources[rtype]):
                resources.append({"id": f"{rtype}_{index}", "resource_type": rtype, "station_id": unit["station_id"], "state": unit["state"], "remaining_ticks": unit["remaining_ticks"], "target_zone": unit["target_zone"], "position": self.resource_position(unit)})
        return {
            "tick": self.env.tick_count, "total_reward": self.total_reward,
            "done": self.done, "contained": self.contained,
            "fire_cells": fire_cells, "state_counts": self.env.sim.state_counts(),
            "buildings_destroyed": self.last_buildings_destroyed,
            "buildings_destroyed_total": self.total_buildings_destroyed,
            "timeout": self.timeout, "resources": resources,
            "weather": {"wind_speed_mph": self.obs["scalars"]["wind_speed_mph"], "wind_direction_deg": self.obs["scalars"]["wind_direction_deg"], "humidity_pct": self.obs["scalars"]["humidity_pct"]},
            "available": {r: self.obs["scalars"][f"{r}_available"] for r in RESOURCE_TYPES},
            "traffic": {"mean_load": self.obs["scalars"]["traffic_mean_load"], "max_load": self.obs["scalars"]["traffic_max_load"]},
            "events": self.events[-10:],
        }


shared_model = None
shared_routing_context = None


def _new_session():
    return Session(model=shared_model, routing_context=shared_routing_context)


sessions = SessionManager(_new_session, max_sessions=MAX_SESSIONS)
_, session = sessions.create("default")  # compatibility for the existing UI and imports
shared_model = session.model
shared_routing_context = session.env.routing_context


def _session_for(session_id: str) -> Session:
    try:
        return sessions.get(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(404, f"Unknown simulation session: {session_id}") from error


@app.get("/")
def index():
    return {
        "name": "InfernoTactics Simulation API",
        "documentation": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "checkpoint": CHECKPOINT,
        "world_grid": WORLD_GRID,
        "traffic_mode": session.env.traffic_mode,
        "active_sessions": len(sessions),
        "max_sessions": sessions.max_sessions,
    }


@app.get("/api/config")
def config(session_id: str = "default"):
    selected = _session_for(session_id)
    return {"simulation_bbox": {"north": 34.150, "south": 34.030, "east": -118.440, "west": -118.605}, "display_bbox": {"north": 34.105, "south": 34.030, "east": -118.485, "west": -118.605}, "grid": selected.env.meta, "roster": {r: len(selected.env.resources[r]) for r in RESOURCE_TYPES}}


@app.get("/api/static/{name}")
def static_data(name: str):
    allowed = {"buildings": "palisades_buildings.geojson", "roads": "palisades_roads.geojson", "depots": "palisades_depots.json", "config": "display_config.json"}
    if name not in allowed:
        raise HTTPException(404, "Unknown static dataset")
    return FileResponse(STATIC_DATA_DIR / allowed[name])


@app.post("/api/reset")
def reset(request: ResetRequest = ResetRequest()):
    try:
        return _session_for(request.session_id).reset(request)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/step")
def step(request: StepRequest):
    selected = _session_for(request.session_id)
    actions = [tuple(action) for action in (request.actions or [])]
    try:
        return selected.advance(request.mode, actions)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/state")
def state(session_id: str = "default"):
    return _session_for(session_id).state()


@app.post("/api/sessions", status_code=201)
def create_session():
    try:
        session_id, created = sessions.create()
    except SessionCapacityError as error:
        raise HTTPException(503, str(error)) from error
    return {"session_id": session_id, "state": created.state()}


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": sessions.ids(), "count": len(sessions), "capacity": sessions.max_sessions}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id == "default":
        raise HTTPException(409, "The backward-compatible default session cannot be deleted")
    try:
        sessions.remove(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(404, f"Unknown simulation session: {session_id}") from error
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
