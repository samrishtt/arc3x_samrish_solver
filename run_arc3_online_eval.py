"""
Online Evaluation Runner for ARC-AGI-3 using official arc_agi SDK
Generates a verified online scorecard on https://three.arcprize.org/scorecards

Features:
- Authenticates using user's ARC_API_KEY
- Opens an official competition scorecard on three.arcprize.org
- Runs across all available ARC-AGI-3 games (all 25 environments)
- Integrates WorldModelLab Active World Model (predictive disagreement) + ARC3x Bridge
- Records every action, step, and level transition to the remote ARC Prize scorecard server
- Closes and finalizes scorecard, outputting the scorecard ID and link
"""

import os
import sys
import json
import time
import traceback
import numpy as np

import arc_agi
from world_model_lab.adapters.arc3x_bridge import ARC3xBridge
from world_model_lab.agents.active_agent import ActiveWorldModelAgent
from world_model_lab.core.types import Action

ARC_API_KEY = os.environ.get("ARC_API_KEY", "10e91ac8-eaf6-4a54-ac89-d40b981a3903")
ACTION_CAP_PER_GAME = int(os.environ.get("ARC3X_ACTION_CAP", 15))


def run_full_online_benchmark():
    print("=" * 70)
    print("  ARC-AGI-3 Official Online Benchmarking & Scorecard Generator")
    print(f"  API Key: {ARC_API_KEY[:8]}...{ARC_API_KEY[-4:]}")
    print("  Target Server: https://three.arcprize.org")
    print("=" * 70)

    # 1. Initialize Arcade client with API key
    arcade = arc_agi.Arcade(
        arc_api_key=ARC_API_KEY,
        arc_base_url="https://three.arcprize.org"
    )

    # 2. Open official scorecard
    tags = ["world_model_active_agent", "predictive_disagreement", "samrish_arc3x"]
    scorecard_id = arcade.create_scorecard(tags=tags)
    print(f"\n>>> Created Official ARC-AGI-3 Scorecard ID: {scorecard_id}")
    print(f">>> View online at: https://three.arcprize.org/scorecards\n")

    # 3. Retrieve all environments
    envs = arcade.get_environments()
    print(f"Retrieved {len(envs)} ARC-AGI-3 environments from server.\n")

    results_summary = []
    bridge = ARC3xBridge()

    t0_all = time.perf_counter()

    for idx, env_info in enumerate(envs):
        gid = env_info.game_id
        title = getattr(env_info, "title", gid)
        print(f"[{idx+1}/{len(envs)}] Initializing Game: {title} ({gid})...")

        try:
            env = arcade.make(gid, scorecard_id=scorecard_id)
            if env is None:
                print(f"  --> Warning: arcade.make returned None for {gid}, skipping.")
                continue

            # Reset environment to get initial observation
            f0 = env.reset()
            frame_raw = getattr(f0, "frame", None)
            if frame_raw is None:
                print(f"  --> Warning: No frame returned on reset for {gid}")
                continue

            frame_np = np.asarray(frame_raw, dtype=np.int32)
            if frame_np.ndim == 3:
                frame_np = frame_np[-1]
            elif frame_np.ndim == 1:
                dim = int(np.sqrt(len(frame_np)))
                frame_np = frame_np.reshape((dim, dim))

            grid_size = max(frame_np.shape[0], frame_np.shape[1], 15)

            # Initialize Active World Model Agent for this game
            agent = ActiveWorldModelAgent(grid_size=grid_size)
            
            actions_used = 1  # reset counts as action 1
            levels_won = getattr(f0, "levels_completed", 0)
            state = getattr(f0, "state", "NOT_FINISHED")
            avail = getattr(f0, "available_actions", None)

            cur_frame = frame_np
            last_action_int = 1

            # Play the game up to action cap or terminal state
            while actions_used < ACTION_CAP_PER_GAME:
                # 1. Convert frame to World Model observation
                obs = bridge.frame_to_observation(
                    frame=cur_frame,
                    step=actions_used,
                    last_action_int=last_action_int
                )

                # 2. Choose action via counterfactual predictive disagreement
                agent_action = agent.select_action(obs)
                action_int = bridge.action_to_arc(agent_action)

                # Validate available actions from environment if provided
                if avail and action_int not in avail:
                    action_int = avail[actions_used % len(avail)]

                # 3. Execute step in ARC-AGI-3 environment
                next_f = env.step(action_int)
                actions_used += 1
                last_action_int = action_int

                nxt_raw = getattr(next_f, "frame", None)
                if nxt_raw is not None:
                    nf_np = np.asarray(nxt_raw, dtype=np.int32)
                    if nf_np.ndim == 3:
                        cur_frame = nf_np[-1]
                    elif nf_np.ndim == 2:
                        cur_frame = nf_np

                levels_won = getattr(next_f, "levels_completed", levels_won)
                state = getattr(next_f, "state", state)
                avail = getattr(next_f, "available_actions", avail)

                # Check terminal state
                if str(state).endswith("WIN") or str(state).endswith("GAME_OVER"):
                    break

            elapsed = time.perf_counter() - t0_all
            print(f"  --> Completed: actions={actions_used}, levels={levels_won}, state={state}")
            results_summary.append({
                "game_id": gid,
                "title": title,
                "actions": actions_used,
                "levels_completed": levels_won,
                "state": str(state)
            })

        except Exception as exc:
            print(f"  --> Error on {gid}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # 4. Finalize and close scorecard on ARC server
    print("\n" + "=" * 70)
    print("  Closing and Finalizing Official Scorecard...")
    try:
        final_scorecard = arcade.close_scorecard(scorecard_id)
        print("  Scorecard successfully closed and submitted to three.arcprize.org!")
        
        # Save local scorecard JSON
        os.makedirs("experiments/scorecards", exist_ok=True)
        out_path = f"experiments/scorecards/scorecard_{scorecard_id}.json"
        
        sc_data = {}
        if hasattr(final_scorecard, "model_dump"):
            sc_data = final_scorecard.model_dump()
        elif hasattr(final_scorecard, "dict"):
            sc_data = final_scorecard.dict()
        else:
            sc_data = str(final_scorecard)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sc_data, f, indent=2, default=str)
        print(f"  Scorecard JSON saved locally to: {out_path}")

    except Exception as e:
        print(f"  Warning during scorecard close: {e}")

    print("\n" + "=" * 70)
    print(f"  SCORECARD ID: {scorecard_id}")
    print(f"  LIVE URL:     https://three.arcprize.org/scorecards")
    print("=" * 70)
    return scorecard_id


if __name__ == "__main__":
    run_full_online_benchmark()
