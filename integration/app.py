import os
import sys
import json
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pyproj import Transformer

# Ensure repo root and best_model/src are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
best_model_dir = os.path.join(repo_root, "best_model")
best_model_src = os.path.join(best_model_dir, "src")

if best_model_src not in sys.path:
    sys.path.insert(0, best_model_src)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from env.inferno_env import InfernoEnv, TRAINING_IGNITION_POINT, RESOURCE_TYPES, SCALAR_KEYS
from models.relative_model import RelativeInfernoModel
from train.relative_actions import TARGET_TYPES, decode_action, resolve_relative_targets

MAX_DISPATCH_SLOTS = 5
MAX_EPISODE_TICKS = 30  # ~60-second total rollout experience at 2.0s/tick

app = FastAPI(title="Palisades 3D Fire Simulator API - High-Performance v10")

app.mount("/static", StaticFiles(directory=current_dir), name="static")

# Global simulation & model state
env = None
model = None
obs = None
total_reward = 0.0
tick_count = 0
is_done = False
coord_grid = None # (316, 595, 2) [lat, lon]
zone_centers = {} # zone_idx -> {lat, lon, bounds}

transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

def precompute_coordinates(meta):
    global coord_grid, zone_centers
    height = meta["height"]
    width = meta["width"]
    transform = meta["transform"]
    
    dx, x0 = transform[0], transform[2]
    dy, y0 = transform[4], transform[5]
    
    coord_grid = np.zeros((height, width, 2), dtype=np.float64)
    
    cols = np.arange(width)
    rows = np.arange(height)
    
    xs = x0 + (cols + 0.5) * dx
    ys = y0 + (rows + 0.5) * dy
    
    grid_x, grid_y = np.meshgrid(xs, ys)
    lons, lats = transformer.transform(grid_x, grid_y)
    
    coord_grid[:, :, 0] = lats
    coord_grid[:, :, 1] = lons
    
    n_rows, n_cols = 4, 8
    r_step = height / n_rows
    c_step = width / n_cols
    
    for z in range(32):
        zr = z // n_cols
        zc = z % n_cols
        r_start, r_end = int(zr * r_step), int(min(height, (zr + 1) * r_step))
        c_start, c_end = int(zc * c_step), int(min(width, (zc + 1) * c_step))
        
        r_mid = (r_start + r_end) // 2
        c_mid = (c_start + c_end) // 2
        
        z_lat = float(coord_grid[r_mid, c_mid, 0])
        z_lon = float(coord_grid[r_mid, c_mid, 1])
        
        min_lat = float(np.min(coord_grid[r_start:r_end, c_start:c_end, 0]))
        max_lat = float(np.max(coord_grid[r_start:r_end, c_start:c_end, 0]))
        min_lon = float(np.min(coord_grid[r_start:r_end, c_start:c_end, 1]))
        max_lon = float(np.max(coord_grid[r_start:r_end, c_start:c_end, 1]))
        
        zone_centers[z] = {
            "lat": z_lat,
            "lon": z_lon,
            "bounds": [min_lat, min_lon, max_lat, max_lon]
        }

def init_environment():
    global env, model, obs, total_reward, tick_count, is_done
    grid_static_path = os.path.join(best_model_dir, "data", "grid_static.npy")
    grid_meta_path = os.path.join(best_model_dir, "data", "grid_meta.json")
    checkpoint_path = os.path.join(best_model_dir, "inferno_best_model.pt")
    
    env = InfernoEnv(grid_static_path=grid_static_path, grid_meta_path=grid_meta_path, seed=42)
    precompute_coordinates(env.meta)
    
    n_grid_channels = env.grid_static.shape[0] + 1
    model = RelativeInfernoModel(
        n_grid_channels=n_grid_channels,
        n_scalars=len(SCALAR_KEYS),
        n_resources=len(RESOURCE_TYPES),
        n_zones=env.n_zones
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    
    reset_simulation()

def reset_simulation():
    global obs, total_reward, tick_count, is_done
    obs = env.reset(ignition_point=TRAINING_IGNITION_POINT, scenario="single", seed=42, use_real_weather=True)
    total_reward = 0.0
    tick_count = 0
    is_done = False

@app.on_event("startup")
async def startup_event():
    init_environment()

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = os.path.join(current_dir, "Simulation.html")
    with open(html_path, "r") as f:
        return f.read()

@app.get("/reset")
async def reset_api():
    reset_simulation()
    fire_state = env.sim.state
    
    blaze_cells, threat_cells = [], []
    rows_blaze, cols_blaze = np.where(fire_state == 3)
    rows_threat, cols_threat = np.where(fire_state == 2)
    
    for r, c in zip(rows_blaze, cols_blaze):
        blaze_cells.append({"r": int(r), "c": int(c), "lat": float(coord_grid[r, c, 0]), "lon": float(coord_grid[r, c, 1]), "state": 3})
    for r, c in zip(rows_threat, cols_threat):
        threat_cells.append({"r": int(r), "c": int(c), "lat": float(coord_grid[r, c, 0]), "lon": float(coord_grid[r, c, 1]), "state": 2})
        
    return JSONResponse({
        "status": "reset",
        "tick": 0,
        "ignition_point": {
            "r": TRAINING_IGNITION_POINT[0],
            "c": TRAINING_IGNITION_POINT[1],
            "lat": float(coord_grid[TRAINING_IGNITION_POINT[0], TRAINING_IGNITION_POINT[1], 0]),
            "lon": float(coord_grid[TRAINING_IGNITION_POINT[0], TRAINING_IGNITION_POINT[1], 1])
        },
        "zones": zone_centers,
        "blaze_cells": blaze_cells,
        "threat_cells": threat_cells
    })

@app.get("/step")
async def step_api():
    global obs, total_reward, tick_count, is_done
    
    if is_done or tick_count >= MAX_EPISODE_TICKS:
        return JSONResponse({"is_done": True, "tick": tick_count, "total_reward": total_reward})
        
    local_available = {
        rtype: int(obs["scalars"][f"{rtype}_available"])
        for rtype in RESOURCE_TYPES
    }
    actions = []
    
    with torch.no_grad():
        for _ in range(MAX_DISPATCH_SLOTS):
            grid_t, scalars_t = RelativeInfernoModel.obs_to_tensors(obs)
            target_zones, target_features = resolve_relative_targets(env, obs)
            zones_t = torch.from_numpy(target_zones).unsqueeze(0)
            features_t = torch.from_numpy(target_features).unsqueeze(0)
            
            logits, value, _ = model(grid_t, scalars_t, zones_t, features_t)
            
            resource_logits = logits["resource_type"][0].clone()
            available = torch.tensor(
                [local_available[r] > 0 for r in RESOURCE_TYPES], dtype=torch.bool
            )
            resource_logits[~available] = -1e9
            if not bool(available.any()):
                break
            resource_idx = int(torch.argmax(resource_logits))
            target_idx = int(torch.argmax(logits["target"][0, resource_idx]))
            
            action = decode_action(resource_idx, target_idx, target_zones)
            if action is None:
                break
            local_available[action[0]] -= 1
            actions.append(action)
            
        # Step env with multi-dispatch actions
        obs, reward, is_done, info = env.step(actions)
        total_reward += reward
        tick_count += 1
        
    fire_state = env.sim.state
    
    rows_blaze, cols_blaze = np.where(fire_state == 3)   # BLAZE (state 3)
    rows_threat, cols_threat = np.where(fire_state == 2) # THREAT (state 2)
    rows_burned, cols_burned = np.where(fire_state == 4) # BURNED OUT (state 4)
    rows_fuel, cols_fuel = np.where(fire_state == 1)     # FUEL AT RISK (state 1)
    
    blaze_cells, threat_cells, burned_cells, fuel_cells = [], [], [], []
    
    for r, c in zip(rows_blaze, cols_blaze):
        blaze_cells.append({"r": int(r), "c": int(c), "lat": float(coord_grid[r, c, 0]), "lon": float(coord_grid[r, c, 1]), "state": 3})
        
    for r, c in zip(rows_threat, cols_threat):
        threat_cells.append({"r": int(r), "c": int(c), "lat": float(coord_grid[r, c, 0]), "lon": float(coord_grid[r, c, 1]), "state": 2})
        
    if len(rows_burned) > 0:
        step_b = max(1, len(rows_burned) // 120)
        for i in range(0, len(rows_burned), step_b):
            r, c = rows_burned[i], cols_burned[i]
            burned_cells.append({"r": int(r), "c": int(c), "lat": float(coord_grid[r, c, 0]), "lon": float(coord_grid[r, c, 1]), "state": 4})

    # Sample fuel cells near active blazes for fuel perimeter visualization
    if len(rows_blaze) > 0 and len(rows_fuel) > 0:
        step_f = max(1, len(rows_fuel) // 80)
        for i in range(0, len(rows_fuel), step_f):
            r, c = rows_fuel[i], cols_fuel[i]
            fuel_cells.append({"r": int(r), "c": int(c), "lat": float(coord_grid[r, c, 0]), "lon": float(coord_grid[r, c, 1]), "state": 1})
            
    actions_data = []
    for act in actions:
        r_type, z_idx = act
        target_info = zone_centers.get(z_idx, {"lat": 34.0725, "lon": -118.5425})
        actions_data.append({
            "resource_type": r_type,
            "zone_idx": int(z_idx),
            "target_lat": target_info["lat"],
            "target_lon": target_info["lon"]
        })
    
    return JSONResponse({
        "tick": tick_count,
        "is_done": is_done or (tick_count >= MAX_EPISODE_TICKS),
        "reward": float(reward),
        "total_reward": float(total_reward),
        "actions": actions_data,
        "blaze_cells": blaze_cells,
        "threat_cells": threat_cells,
        "burned_cells": burned_cells,
        "fuel_cells": fuel_cells,
        "stats": {
            "active_blazes": int(len(rows_blaze)),
            "active_threats": int(len(rows_threat)),
            "burned_out": int(len(rows_burned)),
            "buildings_destroyed": info.get("buildings_destroyed", 0),
            "contained": info.get("contained", False),
            "wind_speed_mph": float(env.weather_history[env.current_weather_idx]["wind_speed_mph"]) if hasattr(env, 'weather_history') and env.weather_history else 25.3
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)