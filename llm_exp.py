# llm_obstacle_experiment.py
"""
Rewritten experiment runner for LLM-assisted obstacle avoidance in CARLA using Ollama + LangChain.
- Updated to work with langchain_community (Ollama) + langchain_core runnable API.
- Strict single-token action prompt + robust parsing.
- Logs metrics needed for research paper:
    model, scenario, trial, initial_distance, speed_at_trigger, inference_time,
    action, ttc_at_intervention, collision(bool), success(bool)
Run:
    python llm_obstacle_experiment.py
Notes:
 - Run this inside a Python 3.10 venv with CARLA Python API installed.
 - Start the CARLA simulator (CarlaUnreal.exe) before running.
 - Pull Ollama models beforehand (e.g. `ollama pull mistral:7b`).
"""

import carla
import random
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
# LangChain Ollama + core runnable API (ensure langchain-community and langchain-core installed)
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------
# Experiment hyperparameters
# ------------------------------
CARLA_HOST = "localhost"
CARLA_PORT = 2000
NUM_TRIALS_PER_MODEL = 10
SIM_TICKS_PER_TRIAL = 800          # max ticks per trial
SIM_TICK_SECONDS = 0.05            # real time per tick (approx)
SAFE_DISTANCE = 10.0               # meters - LLM trigger distance
TRIGGER_TTC_THRESHOLD = 3.0        # seconds - alternative trigger (time-to-collision)
RESPONSE_TIMEOUT = 6.0             # seconds max to wait for a model response
LOG_CSV = "llm_metrics.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------------------
# Models to test (Ollama model names)
# ------------------------------
# Make sure to `ollama pull <ollama_ref>` before running.
LLM_MODELS = [
    {"name": "mistral-7b", "ollama_ref": "mistral:7b"},
    {"name": "orca2-mini", "ollama_ref": "orca2-mini:3b"},
    # add other local Ollama refs as needed
]

# ------------------------------
# Prompt template (strict single-token output)
# ------------------------------
PROMPT_TEMPLATE = (
    "You are an autonomous driving decision module.\n"
    "Context:\n"
    "- Ego speed: {speed_mps:.2f} m/s\n"
    "- Obstacle distance ahead: {distance_m:.2f} m\n"
    "- Obstacle relative speed (closing): {rel_speed_mps:.2f} m/s\n\n"
    "Choose exactly ONE token from the list below that best describes the immediate action to avoid collision.\n\n"
    "TOKENS: STEER_LEFT, STEER_RIGHT, BRAKE_SOFT, BRAKE_HARD, MAINTAIN\n\n"
    "Important: reply with ONLY the token (one word) and nothing else.\n"
)

# ------------------------------
# Utility helpers
# ------------------------------
def vec_length(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

def time_to_collision(distance, rel_speed):
    if rel_speed <= 0.0:
        return float("inf")
    return distance / rel_speed

VALID_ACTIONS = {"STEER_LEFT", "STEER_RIGHT", "BRAKE_SOFT", "BRAKE_HARD", "MAINTAIN"}

def parse_decision(text):
    """Parse LLM output and return one of VALID_ACTIONS or None."""
    if not text:
        return None
    t = text.strip().upper()
    # keep only first token (in case LLM returns extra)
    first = t.split()[0] if len(t.split()) > 0 else t
    if first in VALID_ACTIONS:
        return first
    # fallback: find any valid token in text
    for a in VALID_ACTIONS:
        if a in t:
            return a
    return None

def apply_action_to_vehicle(vehicle, action):
    """Map high-level action to CARLA VehicleControl and apply immediately."""
    if action == "BRAKE_HARD":
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    elif action == "BRAKE_SOFT":
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5)
    elif action == "STEER_LEFT":
        control = carla.VehicleControl(throttle=0.25, steer=-0.6, brake=0.0)
    elif action == "STEER_RIGHT":
        control = carla.VehicleControl(throttle=0.25, steer=0.6, brake=0.0)
    elif action == "MAINTAIN":
        control = carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0)
    else:
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    vehicle.apply_control(control)

# ------------------------------
# Scenario definitions
# ------------------------------
def setup_static_obstacle(world, spawn_point, forward_offset=30.0):
    """Spawn a parked vehicle as obstacle ahead of ego. Returns obstacle actor or None."""
    bps = world.get_blueprint_library().filter("vehicle.*")
    if len(bps) < 2:
        return None
    bp = bps[1]
    obs_transform = carla.Transform(
        carla.Location(x=spawn_point.location.x + forward_offset,
                       y=spawn_point.location.y,
                       z=spawn_point.location.z),
        spawn_point.rotation
    )
    obstacle = world.try_spawn_actor(bp, obs_transform)
    if obstacle:
        obstacle.set_autopilot(False)
    return obstacle

def setup_sudden_stop_vehicle(world, spawn_point, forward_offset=40.0):
    """Spawn a vehicle that we'll command to stop later - for simplicity we spawn stationary obstacle."""
    bps = world.get_blueprint_library().filter("vehicle.*")
    if len(bps) < 3:
        return None
    bp = bps[2]
    obs_transform = carla.Transform(
        carla.Location(x=spawn_point.location.x + forward_offset,
                       y=spawn_point.location.y,
                       z=spawn_point.location.z),
        spawn_point.rotation
    )
    obstacle = world.try_spawn_actor(bp, obs_transform)
    if obstacle:
        obstacle.set_autopilot(False)
    return obstacle

def query_ollama(model_ref, prompt, timeout=6.0):
    """
    Low-level direct call to Ollama HTTP API.
    Always returns a string or raises an exception.
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_ref, "prompt": prompt},
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        print("OLLAMA ERROR:", e)
        return ""

# ------------------------------
# Core trial runner
# ------------------------------
def run_trial(client, world, model_ref, scenario_name, trial_id):
    """Run single trial; returns metric dict."""
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_ref,
        "scenario": scenario_name,
        "trial": trial_id,
        "initial_distance": None,
        "speed_at_trigger": None,
        "inference_time": None,
        "action": None,
        "ttc_at_intervention": None,
        "collision": False,
        "success": False,
    }

    # choose random spawn
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in map")
    spawn_point = random.choice(spawn_points)

    # spawn ego
    bps = world.get_blueprint_library().filter("vehicle.*")
    ego_bp = bps[0]
    ego = world.try_spawn_actor(ego_bp, spawn_point)
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle")
    ego.set_autopilot(False)
    # give initial forward motion
    ego.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

    # spawn obstacle
    obstacle = None
    if scenario_name == "static_obstacle":
        obstacle = setup_static_obstacle(world, spawn_point, forward_offset=30.0)
    elif scenario_name == "sudden_stop":
        obstacle = setup_sudden_stop_vehicle(world, spawn_point, forward_offset=40.0)
    else:
        obstacle = setup_static_obstacle(world, spawn_point, forward_offset=30.0)

    if obstacle is None:
        # cleanup
        try:
            ego.destroy()
        except Exception:
            pass
        return metrics

    # build prompt runnable (PromptTemplate returns a runnable-like callable in core API)
    prompt = PromptTemplate(input_variables=["speed_mps", "distance_m", "rel_speed_mps"], template=PROMPT_TEMPLATE)
    llm = Ollama(model=model_ref, timeout=RESPONSE_TIMEOUT)
    parser = StrOutputParser()

    # Create a runnable pipeline: format -> llm -> parser
    # Note: prompt(...) | llm(...) | parser(...) style is available in core; we use .invoke below
    chain = prompt | llm | parser

    triggered = False

    # main simulation loop
    for tick in range(SIM_TICKS_PER_TRIAL):
        world.tick()

        # pose & kinematics
        ego_loc = ego.get_location()
        ego_vel = ego.get_velocity()
        ego_speed = vec_length(ego_vel)

        obs_loc = obstacle.get_location()
        rel_vec = carla.Location(obs_loc.x - ego_loc.x, obs_loc.y - ego_loc.y, obs_loc.z - ego_loc.z)
        distance = math.sqrt(rel_vec.x * rel_vec.x + rel_vec.y * rel_vec.y + rel_vec.z * rel_vec.z)

        obs_vel = obstacle.get_velocity()
        # conservative relative closing speed estimate
        rel_speed = max(0.0, vec_length(ego_vel) - vec_length(obs_vel))

        ttc = time_to_collision(distance, rel_speed)

        if metrics["initial_distance"] is None:
            metrics["initial_distance"] = distance

        # trigger condition
        if (not triggered) and (distance <= SAFE_DISTANCE or ttc <= TRIGGER_TTC_THRESHOLD):
            triggered = True
            metrics["speed_at_trigger"] = ego_speed
            metrics["ttc_at_intervention"] = ttc if math.isfinite(ttc) else -1.0

            # call LLM
            try:
                t0 = time.time()
                # chain.invoke expects mapping for inputs matching PromptTemplate variables
                prompt_text = PROMPT_TEMPLATE.format(
                speed_mps=ego_speed,
                distance_m=distance,
                rel_speed_mps=rel_speed
            )

                t0 = time.time()
                raw_output = query_ollama(model_ref, prompt_text, timeout=RESPONSE_TIMEOUT)
                inference_time = time.time() - t0
            except Exception as e:
                raw_output = ""
                inference_time = RESPONSE_TIMEOUT

            metrics["inference_time"] = round(inference_time, 4)
            action = parse_decision(raw_output)
            metrics["action"] = action if action is not None else "NONE"

            # fallback on failure
            if action is None:
                action = "BRAKE_HARD"

            apply_action_to_vehicle(ego, action)

            # After intervention observe short window (3s) to evaluate success/collision
            observe_ticks = int(3.0 / SIM_TICK_SECONDS)
            collided = False
            for _ in range(observe_ticks):
                world.tick()
                ego_loc = ego.get_location()
                obs_loc = obstacle.get_location()
                distance_after = ego_loc.distance(obs_loc)
                if distance_after < 2.0:
                    collided = True
                    break

            metrics["collision"] = bool(collided)
            # success = not collided and distance increased beyond safe margin
            if not collided and distance_after > (SAFE_DISTANCE + 2.0):
                metrics["success"] = True
            else:
                metrics["success"] = False

            # end trial after intervention window
            break

    # cleanup actors
    try:
        if obstacle is not None and obstacle.is_alive:
            obstacle.destroy()
    except Exception:
        pass
    try:
        if ego is not None and ego.is_alive:
            ego.destroy()
    except Exception:
        pass

    # defaults
    if metrics["inference_time"] is None:
        metrics["inference_time"] = -1.0
    if metrics["action"] is None:
        metrics["action"] = "NONE"
    if metrics["ttc_at_intervention"] is None:
        metrics["ttc_at_intervention"] = -1.0

    return metrics

# ------------------------------
# Main experiment loop
# ------------------------------
def main():
    # connect to CARLA server
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(60.0)  # allow more time for connection
    world = client.get_world()

    all_metrics = []
    scenarios = ["static_obstacle", "sudden_stop"]

    for model in LLM_MODELS:
        model_ref = model["ollama_ref"]
        print(f"\n=== Testing model: {model['name']} ({model_ref}) ===")
        for scenario in scenarios:
            print(f"Scenario: {scenario}")
            for trial in range(NUM_TRIALS_PER_MODEL):
                print(f" Trial {trial+1}/{NUM_TRIALS_PER_MODEL} ...", end=" ", flush=True)
                try:
                    metrics = run_trial(client, world, model_ref, scenario, trial+1)
                except Exception as e:
                    print("ERROR during trial:", e)
                    metrics = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": model_ref,
                        "scenario": scenario,
                        "trial": trial+1,
                        "initial_distance": None,
                        "speed_at_trigger": None,
                        "inference_time": -1.0,
                        "action": "ERROR",
                        "ttc_at_intervention": -1.0,
                        "collision": True,
                        "success": False,
                    }
                all_metrics.append(metrics)
                print("done. action=", metrics["action"], " inference_time=", metrics["inference_time"])

    # save metrics
    df = pd.DataFrame(all_metrics)
    df.to_csv(LOG_CSV, index=False)
    print(f"\nSaved metrics to {LOG_CSV}")

    # print summary
    grouped = df.groupby("model").agg({
        "success": ["mean", "sum"],
        "collision": ["mean", "sum"],
        "inference_time": ["mean", "median"],
    })
    print("\nSummary per model:")
    print(grouped)

if __name__ == "__main__":
    main()
