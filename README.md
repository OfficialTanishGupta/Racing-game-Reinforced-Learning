# RL Racing

A 2D top-down racing project where cars learn to drive **from scratch** using reinforcement learning (PPO). Nothing about the track is hard-coded into the agent: it only sees distance sensors and its own speed and steering, and learns to stay on the road and go fast. Train multiple models, then race them against each other on different tracks.

## Features

- Custom [Gymnasium](https://gymnasium.farama.org/) environment with simple car physics
- PPO agents trained with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Procedurally generated closed tracks (`oval`, `wavy`, `twisty`, `zigzag`)
- Train a **generalist** (random track every episode) or a **specialist** (one fixed track)
- Real-time pygame race viewer with live leaderboard, lap counter, and crash respawn

## Project Structure

```
rl-racing/
├── track.py    # Track geometry, projection, edges, and track presets
├── env.py      # Gymnasium environment (physics, sensors, reward)
├── train.py    # PPO training script (CLI)
├── race.py     # Multi-model race viewer (pygame)
├── models/     # Saved models (created on first training run)
└── README.md
```

## Setup

Requires Python 3.9+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy gymnasium stable-baselines3 pygame
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Usage

### Train

```bash
# Generalist: random track each episode
python train.py --name generalist --seed 0

# Specialists: fixed track
python train.py --name oval_pro --track oval --seed 1
python train.py --name twisty_pro --track twisty --seed 2
```

| Argument  | Default      | Description                                            |
| --------- | ------------ | ------------------------------------------------------ |
| `--name`  | `generalist` | Output file name, saved to `models/<name>.zip`         |
| `--track` | random       | `oval`, `wavy`, `twisty`, or `zigzag`                  |
| `--steps` | `1500000`    | Total training timesteps                               |
| `--seed`  | `0`          | Random seed (different seeds give different "drivers") |
| `--lr`    | `3e-4`       | PPO learning rate                                      |

### Race

```bash
python race.py --track wavy --models models/generalist.zip models/oval_pro.zip models/twisty_pro.zip --laps 3
```

| Argument   | Default | Description                              |
| ---------- | ------- | ---------------------------------------- |
| `--track`  | `oval`  | Track to race on                         |
| `--models` | none    | One or more model paths (up to 6 colors) |
| `--laps`   | `3`     | Laps required to finish                  |

In race mode, a car that leaves the track is respawned on the centerline at zero speed. Cars do not collide with each other.

## How It Works

### Observation (9 values, normalized)

- 7 ray-distance sensors at -90°, -45°, -20°, 0°, 20°, 45°, 90° relative to heading (max range 250 px)
- Current speed
- Current steering value

### Action (continuous, 2 values in [-1, 1])

- Steering
- Throttle / brake

### Reward

| Event                         | Reward                    |
| ----------------------------- | ------------------------- |
| Progress along the centerline | `+progress / 10` per step |
| Time penalty                  | `-0.01` per step          |
| Crash (leaving the track)     | `-10`, episode ends       |
| Driving backwards             | `-5`, episode ends        |
| Completing a lap              | `+20`, episode ends       |

Progress is measured as arc length along the track centerline, so the agent is rewarded for moving forward, not just for surviving.

### Tracks

Tracks are closed centerlines generated from a base radius plus sine harmonics, with a fixed road width. Add your own in `track.py`:

```python
TRACKS["my_track"] = _make("my_track", sx=1.4, harmonics=[(3, 70, 0), (5, 25, 1)], width=65)
```

Keep harmonic amplitudes small relative to `base` to avoid self-intersecting tracks.

## Tips

- Training on all tracks (no `--track`) gives better generalization; specialists usually win on their home track but fail elsewhere.
- Try different `--seed` values: same setup, noticeably different driving styles.
- If the agent learns to crawl, increase `--steps` or reduce the time penalty in `env.py`.
- Increase `n_envs` in `train.py` if you have more CPU cores.

## Roadmap

- [ ] Evaluation script: fixed-seed lap times per model per track
- [ ] Tyre grip and slip in the car physics
- [ ] Car-vs-car interaction (other cars as sensor inputs, self-play)
- [ ] Track import and a hand-drawn track editor
- [ ] Curriculum learning (easy to hard tracks)

## License

MIT
