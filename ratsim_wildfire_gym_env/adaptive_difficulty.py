"""Avalon-style adaptive difficulty over world-config keys.

A single scalar difficulty ``d in [0, 1]`` slides a declared set of world-config
keys between ``from`` (d=0) and ``to`` (d=1). After every episode, d moves up on
success and down on failure (Avalon's CurriculumWrapper rule: with symmetric
steps the random walk settles where P(success) = 0.5 — the agent is held at its
frontier of competence; asymmetric steps shift the target success rate).

Success criterion: ``episode_pickups >= success_pickups`` (the env reports the
TaskTracker pickup count in the terminal step's info). E.g. with 3 rewards per
well dispense, >= 4 pickups proves the agent completed the cued Random trial
AND returned to Home for at least one more.

The ranges dict comes from the experiment def's ``adaptive_difficulty:`` block
(see ratsim_experiments/experiment_defs.py). Two primitives:

  linear:  {"from": 80, "to": 200}            optionally {"round": 1} for ints
  switch:  {"switch_at": 0.4, "below": "home_well_room", "above": "central_chamber"}

Coupled integer keys (e.g. maze/n_rooms and maze/rooms/reward_room/min =
n_rooms - 2) stay consistent under rounding as long as their from/to values
differ by an exact integer offset (same span -> same fractional part at every d).

NOTE: each wrapped env instance carries its own difficulty state. With
SubprocVecEnv and n_envs > 1 the difficulties evolve independently per env;
fine for the usual n_envs=1 setup here.
"""
from __future__ import annotations

import gymnasium as gym


def interpolate_ranges(ranges: dict, d: float) -> dict:
    """Evaluate a ranges spec at difficulty d -> flat world-config override dict."""
    d = min(1.0, max(0.0, float(d)))
    out = {}
    for key, spec in ranges.items():
        if not isinstance(spec, dict):
            raise TypeError(f"adaptive_difficulty range for '{key}' must be a dict, got {spec!r}")
        if "switch_at" in spec:
            for req in ("below", "above"):
                if req not in spec:
                    raise ValueError(f"adaptive_difficulty '{key}': switch_at needs '{req}'")
            out[key] = spec["above"] if d >= float(spec["switch_at"]) else spec["below"]
        elif "from" in spec and "to" in spec:
            lo, hi = float(spec["from"]), float(spec["to"])
            val = lo + (hi - lo) * d
            if spec.get("round"):
                val = int(round(val))
            out[key] = val
        else:
            raise ValueError(
                f"adaptive_difficulty '{key}': need either from/to or switch_at/below/above, got {spec!r}")
    return out


class AdaptiveDifficultyWrapper(gym.Wrapper):
    """Slides the wrapped WildfireGymEnv's worldgen_config with performance.

    On each terminal step, reads ``info["episode_pickups"]`` and nudges d
    (+step_up on success, -step_down on failure, clamped to [0, 1]). On each
    reset, writes ``base_worldgen | interpolate_ranges(ranges, d)`` into the
    env's worldgen_config — the env re-sends world config to Unity every
    reset, so the next episode is generated at the new difficulty. The current
    d is exposed in every step's info as ``info["difficulty"]`` and recorded
    into the per-episode JSONL via the env's extra_log_fields.
    """

    def __init__(self, env: gym.Env, ranges: dict, success_pickups: int = 4,
                 step_up: float = 0.01, step_down: float = 0.01, d0: float = 0.0):
        super().__init__(env)
        self.ranges = dict(ranges)
        self.success_pickups = int(success_pickups)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.difficulty = min(1.0, max(0.0, float(d0)))
        # Snapshot the pristine base config once; every reset recomputes
        # base | overrides so overrides never accumulate or leak.
        self._base_worldgen = dict(self.env.unwrapped.worldgen_config)
        # Fail fast on a malformed spec (don't wait for the first reset).
        interpolate_ranges(self.ranges, self.difficulty)

    def reset(self, **kwargs):
        overrides = interpolate_ranges(self.ranges, self.difficulty)
        inner = self.env.unwrapped
        inner.worldgen_config = {**self._base_worldgen, **overrides}
        if hasattr(inner, "extra_log_fields"):
            inner.extra_log_fields["difficulty"] = round(self.difficulty, 6)
        print(f"[adaptive_difficulty] world reset at difficulty d={self.difficulty:.3f}")
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            pickups = info.get("episode_pickups")
            if pickups is None:
                print("[adaptive_difficulty] WARNING: terminal info lacks "
                      "'episode_pickups'; difficulty not updated this episode")
            else:
                success = pickups >= self.success_pickups
                self.difficulty += self.step_up if success else -self.step_down
                self.difficulty = min(1.0, max(0.0, self.difficulty))
                info["difficulty_success"] = success
                print(f"[adaptive_difficulty] episode pickups={pickups} "
                      f"({'success' if success else 'failure'}) -> d={self.difficulty:.3f}")
        info["difficulty"] = self.difficulty
        return obs, reward, terminated, truncated, info
