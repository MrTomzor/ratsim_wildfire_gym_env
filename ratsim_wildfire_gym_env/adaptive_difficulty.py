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

Difficulty state: by default each wrapped env instance carries its own
difficulty (with n_envs > 1 that means K independent walks, one per env).
Pass ``state_path`` to share ONE walk across all envs of a run instead: the
value lives in a tiny text file, is re-read (under flock) at every reset and
bumped with a locked read-modify-write on every episode end from ANY env —
works both in-process (dreamer driver) and across SubprocVecEnv worker
processes. The file persists across stages and crashes, so it also subsumes
the jsonl-based resume: an existing file wins over d0.
"""
from __future__ import annotations

import fcntl
import json
import os

import gymnasium as gym


def last_logged_difficulty(jsonl_path) -> "float | None":
    """Last recorded 'difficulty' in a train_episodes.jsonl, or None.

    Used to make the difficulty walk persist across scheduler stages: each
    stage (and each crash-resume) runs in a fresh process with a fresh
    wrapper, so the trainer seeds d0 from the run's cumulative episode log
    instead of restarting at the def's d0. Tolerates missing files, partial
    trailing lines, and records without the field (non-adaptive stages).
    """
    last = None
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("difficulty") is not None:
                    last = float(rec["difficulty"])
    except OSError:
        return None
    return last


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

    With ``state_path`` set, d is SHARED across all envs of the run: the value
    lives in that file, is re-read (under flock) at each reset and bumped with
    a locked read-modify-write on each episode end — one walk fed by every
    env, valid both in-process and across SubprocVecEnv workers. An existing
    file wins over ``d0`` (automatic stage/crash resume). Without
    ``state_path`` each wrapper keeps its own independent walk.
    """

    def __init__(self, env: gym.Env, ranges: dict, success_pickups: int = 4,
                 step_up: float = 0.01, step_down: float = 0.01, d0: float = 0.0,
                 state_path=None):
        super().__init__(env)
        self.ranges = dict(ranges)
        self.success_pickups = int(success_pickups)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.difficulty = min(1.0, max(0.0, float(d0)))
        # Optional shared walk: one difficulty value for all envs of a run,
        # stored in a file and updated under flock (safe across processes).
        self.state_path = str(state_path) if state_path is not None else None
        if self.state_path is not None:
            self._init_state_file(self.difficulty)
            # Adopt the shared value — an existing file wins over d0.
            self.difficulty = self._shared_rmw(lambda d: d)
        # Snapshot the pristine base config once; every reset recomputes
        # base | overrides so overrides never accumulate or leak.
        self._base_worldgen = dict(self.env.unwrapped.worldgen_config)
        # Fail fast on a malformed spec (don't wait for the first reset).
        interpolate_ranges(self.ranges, self.difficulty)

    def _init_state_file(self, d0: float) -> None:
        """Create the shared state file with d0 iff it doesn't exist yet.

        O_EXCL makes creation atomic: exactly one env wins the race, the
        rest adopt the existing value.
        """
        try:
            fd = os.open(self.state_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return
        with os.fdopen(fd, "w") as f:
            f.write(f"{d0:.6f}")

    def _shared_rmw(self, fn) -> float:
        """flock'd read-modify-write on the shared value; returns the result.

        fn(current) -> new; the result is clamped to [0, 1] and written back.
        A corrupt/partial file falls back to this wrapper's last known value.
        """
        with open(self.state_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                d = float(f.read().strip())
            except ValueError:
                d = self.difficulty
            d = min(1.0, max(0.0, fn(d)))
            f.seek(0)
            f.truncate()
            f.write(f"{d:.6f}")
        return d

    def reset(self, **kwargs):
        if self.state_path is not None:
            # Pick up bumps made by other envs since our last episode.
            self.difficulty = self._shared_rmw(lambda d: d)
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
                delta = self.step_up if success else -self.step_down
                if self.state_path is not None:
                    # Bump the SHARED value (which other envs may have moved
                    # since this episode started), not our local copy.
                    self.difficulty = self._shared_rmw(lambda d: d + delta)
                else:
                    self.difficulty = min(1.0, max(0.0, self.difficulty + delta))
                info["difficulty_success"] = success
                print(f"[adaptive_difficulty] episode pickups={pickups} "
                      f"({'success' if success else 'failure'}) -> d={self.difficulty:.3f}")
        info["difficulty"] = self.difficulty
        return obs, reward, terminated, truncated, info
