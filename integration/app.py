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

from env.inferno_env import InfernoEnv, TRAINING_IGNITION_POINT, RESOURCE_TYPES
from models.inferno_model import InfernoModel

app = FastAPI(title="Palisades 3D Fire Simulator API")

# Mount static files if needed
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

# Initialize coordinate transformation
transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

def precompute_coordinates(meta):
    global coord_grid, zone_centers
    height = meta["height"]
    width = meta["width"]
    transform = meta["transform"]
    
    # transform: [dx, 0, x0, 0, dy, y0]
    dx, x0 = transform[0], transform[2]
    dy, y0 = transform[4], transform[5]
    
    coord_grid = np.zeros((height, width, 2), dtype=np.float64)
    
    cols = np.arange(width)
    rows = np.arange(height)
    
    xs = x0 + (cols + 0.5) * dx
    ys = y0 + (rows + 0.5) * dy # dy is -30.0
    
    grid_x, grid_y = np.meshgrid(xs, ys)
    lons, lats = transformer.transform(grid_x, grid_y)
    
    coord_grid[:, :, 0] = lats
    coord_grid[:, :, 1] = lons
    
    # Precompute zone centers (32 zones = 4 rows x 8 cols)
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
    model = InfernoModel(n_grid_channels=n_grid_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
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
    blaze_coords = []
    
    # Collect initial burning cells
    rows, cols = np.where((fire_state == 2) | (fire_state == 3)) # THREAT or BLAZE
    for r, c in zip(rows, cols):
        blaze_coords.append({
            "r": int(r),
            "c": int(c),
            "lat": float(coord_grid[r, c, 0]),
            "lon": float(coord_grid[r, c, 1]),
            "state": int(fire_state[r, c])
        })
        
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
        "fire_cells": blaze_coords
    })

@app.get("/step")
async def step_api():
    global obs, total_reward, tick_count, is_done
    
    if is_done:
        return JSONResponse({"is_done": True, "tick": tick_count, "total_reward": total_reward})
        
    with torch.no_grad():
        grid_t, scalars_t = InfernoModel.obs_to_tensors(obs)
        action_logits, value, _ = model(grid_t, scalars_t)
        
        resource_idx = action_logits["resource_type"].argmax(dim=-1).item()
        zone_idx = action_logits["zone"].argmax(dim=-1).item()
        
        resource_type = RESOURCE_TYPES[resource_idx] if resource_idx < len(RESOURCE_TYPES) else None
        
        # Step env
        obs, reward, is_done, info = env.step((resource_type, zone_idx))
        total_reward += reward
        tick_count += 1
        
    fire_state = env.sim.state
    
    # Sample active fire cells (BLAZE=3, THREAT=2, BURNED_OUT=4)
    active_rows, active_cols = np.where((fire_state == 2) | (fire_state == 3))
    burned_rows, burned_cols = np.where(fire_state == 4)
    
    fire_cells = []
    for r, c in zip(active_rows, active_cols):
        fire_cells.append({
            "r": int(r),
            "c": int(c),
            "lat": float(coord_grid[r, c, 0]),
            "lon": float(coord_grid[r, c, 1]),
            "state": int(fire_state[r, c])
        })
        
    # Sample subset of burned cells for performance in 3D rendering
    burned_cells = []
    if len(burned_rows) > 0:
        step = max(1, len(burned_rows) // 150)
        for i in range(0, len(burned_rows), step):
            r, c = burned_rows[i], burned_cols[i]
            burned_cells.append({
                "r": int(r),
                "c": int(c),
                "lat": float(coord_grid[r, c, 0]),
                "lon": float(coord_grid[r, c, 1]),
                "state": 4
            })
            
    target_info = zone_centers.get(zone_idx, {"lat": 34.0725, "lon": -118.5425})
    
    return JSONResponse({
        "tick": tick_count,
        "is_done": is_done,
        "reward": float(reward),
        "total_reward": float(total_reward),
        "action": {
            "resource_type": resource_type,
            "zone_idx": int(zone_idx),
            "target_lat": target_info["lat"],
            "target_lon": target_info["lon"]
        },
        "fire_cells": fire_cells,
        "burned_cells": burned_cells,
        "stats": {
            "active_blazes": int(len(active_rows)),
            "burned_out": int(len(burned_rows)),
            "buildings_destroyed": info.get("buildings_destroyed", 0),
            "contained": info.get("contained", False),
            "wind_speed_mph": float(env.weather_history[env.current_weather_idx]["wind_speed_mph"]) if hasattr(env, 'weather_history') and env.weather_history else 25.0
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)