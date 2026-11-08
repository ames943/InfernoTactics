"""Local experimental World viewer: streamlit run src/viz/world_app.py.

Rendering and widget reruns are read-only; only explicit form submissions
create, ignite, or advance the session's authoritative World.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pyproj import Transformer
import streamlit as st

# Streamlit executes this file as a script, not as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.world.contracts import Action
from src.world.scenario import load_scenario
from src.world.synthetic import make_synthetic
from src.world.world import World


def draw_map(world: World, observation: dict):
    """Render north-up projected rasters and actual operational snapshots."""
    scenario = world.scenario
    a, _, c, _, e, f = scenario.transform
    height, width = scenario.landscape["valid"].shape
    extent = (c, c + width * a, f + height * e, f)
    fig, ax = plt.subplots(figsize=(10, 7))
    fuel = np.ma.masked_where(~scenario.landscape["valid"], scenario.landscape["fuel_model"])
    image = ax.imshow(fuel, extent=extent, origin="upper", cmap="terrain", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Fuel model code", shrink=0.7)
    burned = observation["fire"]["burned"]
    ax.imshow(np.ma.masked_where(~burned, np.ones(burned.shape)), extent=extent,
              origin="upper", cmap=ListedColormap(["#df382c"]), alpha=0.65,
              interpolation="nearest")
    for unit in observation["units"]:
        for segment in unit.get("route", []):
            ax.plot([segment["start_x"], segment["x"]],
                    [segment["start_y"], segment["y"]], "--", color="royalblue", linewidth=1)
        ax.scatter(unit["x"], unit["y"], marker="^", s=70, color="navy", zorder=5)
        ax.annotate(f"{unit['id']} ({unit['status']})", (unit["x"], unit["y"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel=f"Projected x (m), {scenario.crs}", ylabel="Projected y (m)",
           title="Fuel, cumulative burned footprint (red), units and assigned routes (blue)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Wildfire World", layout="wide")
    st.title("Wildfire World — local research viewer")
    st.warning("EXPERIMENTAL: suppression effects and synthetic inputs are not validated operational predictions. Not for emergency dispatch.")
    st.caption("No external viewer service or credentials required. Reruns only render; time advances only with Step or Run to horizon.")
    with st.sidebar.form("scenario_source"):
        st.subheader("Create / reset World")
        source = st.selectbox("Scenario source", ["Synthetic (illustrative)", "Local scenario directory"])
        directory = st.text_input("Local scenario directory", placeholder="/absolute/path/to/scenario")
        seed = st.number_input("World / synthetic seed", min_value=0, value=0, step=1)
        duration = st.number_input("Synthetic horizon (seconds)", min_value=60, value=3600, step=60)
        create = st.form_submit_button("Load / reset World")
    if create:
        try:
            scenario = (make_synthetic(seed=int(seed), duration_s=int(duration))
                        if source.startswith("Synthetic") else load_scenario(Path(directory).expanduser()))
            # Do not discard the working session if construction fails.
            new_world = World(scenario, seed=int(seed))
            st.session_state["wildfire_world"] = new_world
            st.session_state.pop("wildfire_error", None)
        except Exception as exc:
            st.error(f"Could not load scenario: {exc}")
    world = st.session_state.get("wildfire_world")
    if world is None:
        st.info("Choose synthetic or local inputs and click Load / reset World. Loading does not ignite or advance the simulation.")
        return
    scenario = world.scenario
    st.subheader(scenario.name)
    st.caption(f"Start UTC: {scenario.start_utc} · CRS: {scenario.crs} · Seed: {world.seed}")
    a, _, c, _, e, f = scenario.transform
    height, width = scenario.landscape["valid"].shape
    center_x, center_y = c + a * width / 2, f + e * height / 2
    lon, lat = Transformer.from_crs(scenario.crs, "EPSG:4326", always_xy=True).transform(center_x, center_y)
    with st.form("ignition"):
        st.subheader("Geographic ignition")
        left, right = st.columns(2)
        longitude = left.number_input("Ignition longitude", value=float(lon), format="%.7f", key="ignition_lon_" + scenario.name)
        latitude = right.number_input("Ignition latitude", value=float(lat), format="%.7f", key="ignition_lat_" + scenario.name)
        ignite = st.form_submit_button("Ignite at longitude / latitude")
    if ignite:
        try:
            world.ignite(longitude, latitude)
            st.success("Ignition submitted without advancing time.")
        except Exception as exc:
            st.error(f"Ignition rejected: {exc}")
    observation = world.observe()
    st.metric("Elapsed simulated time", f"{world.time_s:g} / {scenario.duration_s:g} s")
    st.caption(f"Cumulative burned area: {observation['burned_area_m2']:,.0f} m² · Asset exposure: {observation['asset_exposure_m2']:,.0f} m²")
    with st.form("operations"):
        st.subheader("Explicit resource action")
        unit_ids = [unit["id"] for unit in observation["units"]]
        unit_id = st.selectbox("Unit", unit_ids) if unit_ids else None
        operation = st.selectbox("Operation", ["wait", "water", "line", "return"])
        left, right = st.columns(2)
        target_x = left.number_input("Target projected x (metres)", value=float(center_x), format="%.2f", key="target_x_" + scenario.name)
        target_y = right.number_input("Target projected y (metres)", value=float(center_y), format="%.2f", key="target_y_" + scenario.name)
        st.caption("Water / line use the projected target above; wait advances without dispatch. Return interrupts an eligible unit. Run submits this action once, then waits to the scenario horizon.")
        left, right = st.columns(2)
        step = left.form_submit_button("Step (up to 60 seconds)", disabled=world.done)
        run = right.form_submit_button("Run to horizon", disabled=world.done)
    if step or run:
        try:
            action = Action(unit_id=unit_id if operation != "wait" else None,
                            operation=operation,
                            x=target_x if operation in {"water", "line"} else None,
                            y=target_y if operation in {"water", "line"} else None)
            with st.spinner("Advancing the World…"):
                world.step([action])
                if run:
                    while not world.done:
                        world.step([])
            # Re-render metrics and all panels against the same updated clock.
            st.rerun()
        except Exception as exc:
            st.error(f"Could not advance World at {world.time_s:g} s: {exc}. Earlier completed steps, if any, remain in this session.")
    observation = world.observe()
    figure = draw_map(world, observation)
    st.pyplot(figure)
    plt.close(figure)
    st.subheader("Resource state")
    st.dataframe([{key: unit.get(key) for key in ("id", "kind", "status", "x", "y", "supply_l", "remaining_work_s", "blocked")}
                  for unit in observation["units"]], use_container_width=True)
    st.subheader("Events")
    start_utc = datetime.fromisoformat(scenario.start_utc.replace("Z", "+00:00"))
    st.dataframe([{"time_s": event.time_s, "timestamp_utc": (start_utc + timedelta(seconds=event.time_s)).isoformat(), "kind": event.kind, "unit_id": event.unit_id,
                   "details": json.dumps(event.details, default=str, sort_keys=True)}
                  for event in observation["events"]], use_container_width=True)
    with st.expander("Scenario provenance and validity"):
        st.json(scenario.provenance)
        st.caption("Provenance labels document inputs; they do not certify fire spread or suppression validity. Red shows arrival history, not current flame or structure combustion.")
        st.caption("Observation mode: oracle_current — perfect current simulated state, not an operational sensor feed. Fire/operations coupling delay is bounded by 60 seconds.")


if __name__ == "__main__":
    main()
