<div align="center">

# RL-USERS

**Multi-agent reinforcement learning simulation of Amazon shopper behavior**

50 virtual shoppers, trained on 701K real Amazon reviews, browse a live product catalog, influence each other through a social graph, and are validated against real e-commerce benchmarks — end to end, in real time.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BC%20%2B%20PPO-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-microservices-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-pub%2Fsub-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-event%20stream-231F20?logo=apachekafka&logoColor=white)](https://kafka.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![D3.js](https://img.shields.io/badge/D3.js-force%20graph-F9A03C?logo=d3dotjs&logoColor=white)](https://d3js.org/)

</div>

---

## What this is

RL-USERS simulates a shopping population, not a shopping cart. Each of the 50 agents has a persona (power buyer, average buyer, browser), a wallet, and a reinforcement-learned policy for what to look at next — trained first by behavioral cloning on real purchase sequences, then fine-tuned with PPO. Agents move through a funnel (`browse → product_detail → cart → checkout`), influence their neighbors on a social graph when they buy something, and their *aggregate behavior* — not their individual product choices — is validated against published e-commerce benchmarks.

The simulation streams live: every agent action is published to Redis and Kafka as it happens, a FastAPI gateway forwards it over a WebSocket, and the dashboard updates round by round instead of reading a static report file. The gateway starts the simulation itself on boot (`live/runner.py`) — there is no separate process to launch, on Render or locally.

## Architecture

```
                     ┌─────────────────────────┐
                     │   Amazon Reviews (701K)  │
                     │   All_Beauty category    │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Behavioral Cloning → PPO │   agents/bc_trainer.py
                     │  policy_bc.pt → policy_rl.pt│  agents/rl_trainer.py
                     └────────────┬─────────────┘
                                  │
   ┌──────────────────────────────▼──────────────────────────────┐
   │                    simulate_live.py                          │
   │  50 agents · heap scheduler · BFS social propagation ·       │
   │  topological funnel gate · friction-driven abandonment       │
   └───────┬──────────────────────────────────────────┬──────────┘
           │ publish per event                          │ publish per event
   ┌───────▼────────┐                           ┌───────▼────────┐
   │      Redis      │                           │  Apache Kafka   │
   │    pub/sub       │                           │  agent_actions  │
   └───────┬────────┘                           └────────────────┘
           │ subscribe
   ┌───────▼────────────────────┐        ┌─────────────────────────┐
   │  FastAPI gateway /ws/live   │───────▶│   Live dashboard (D3)    │
   │  + microservice cluster     │  WS    │   ● LIVE / ○ LAST RUN    │
   └─────────────────────────────┘        └─────────────────────────┘
```

The simulation loop above (`live/engine.py`) runs two ways depending on topology: the FastAPI gateway starts it on a background thread by default (`live/runner.py`), so the dashboard is live with no setup step; `simulate_live.py` runs the identical loop as a standalone process, for when you want it decoupled from the gateway. Either one publishes to Redis/Kafka when they're reachable and falls back to an in-process queue when they're not.

- **Training** — `agents/bc_trainer.py` imitation-learns from real user sequences; `agents/rl_trainer.py` fine-tunes with PPO, boosting product logits for agents that just received a social signal.
- **Simulation core** — a min-heap event scheduler, an Erdős–Rényi social graph with BFS influence propagation (`1 / 2^depth` decay), and a funnel gate ordered by `topo_sort` of the funnel graph itself. All three run in pure Python: profiling puts them at **0.83% of simulation runtime**, against **64% in ONNX policy inference**, so there is nothing here for a native port to buy.
- **Live infrastructure** — every agent action publishes to Redis (dashboard feed, throttled to ~2 updates/sec for readability) and Kafka (`agent_actions`, full-fidelity event log, retention-capped so a long-running dev session can't fill the disk). Kafka is currently write-only: no consumer exists yet, but the stream is there for future subscribers — e.g. an inventory tracker, an audit log, or another agent reacting to the first 50.
- **Backend** — a FastAPI gateway fronting `product_service`, `cart_service`, `order_service`, and `session_service`, each with its own Redis-backed state.
- **Validation** — session length, conversion rate, cart abandonment (over the cart-reaching denominator), per-persona conversion and trend concentration, each checked against real industry benchmarks with a PASS/FAIL verdict over 4,000 seeded sessions.

## Features

- 🧠 **Real RL policy inference**, not scripted behavior — every product choice is a forward pass through a PPO-trained model, with social signals live-boosting the logits of trending products.
- 🕸️ **Social contagion you can watch happen** — click any node in the dashboard's social graph to see a BFS influence pulse propagate outward in real time, with edges highlighting along the actual propagation path.
- 📡 **Genuinely live dashboard** — WebSocket-driven; the gateway starts the simulation itself on boot, and a fresh instance (e.g. Render waking from a free-tier sleep) just starts a fresh simulation with no manual step.
- ✅ **Behavioral validation against real data**, not synthetic targets — session length, conversion rate, abandonment, and trend concentration are all benchmarked against published e-commerce numbers.
- 🐳 **Fully containerized backend** — Redis, Kafka, and four FastAPI microservices behind a gateway, orchestrated with a single `docker-compose up`.

## Getting started

### Prerequisites

- Python 3.11+
- Docker Desktop (for Redis, Kafka, and the microservice cluster)
- Node.js 18+ (for the dashboard, optional — a static HTML dashboard is also included)

### 1. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`requirements.txt` is enough to serve the dashboard and run the live simulation — inference goes through `policy_rl.onnx` (`agents/policy_runtime.py`), not PyTorch, which is what keeps a Render free instance under its 512 MB limit. Training and batch validation (`train.py`, `simulate.py`) still need `torch`, `pandas`, `scikit-learn`, and `requests`; install those separately if you're doing either:

```bash
pip install torch pandas scikit-learn requests
```

### 2. Get the dataset

The training data (701K Amazon "All Beauty" reviews, ~310MB) isn't checked into this repo — grab it from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) and place it at:

```
data/raw/All_Beauty.jsonl
```

### 3. Bring up the backend

```bash
docker-compose up -d --build
```

This starts Redis, Zookeeper + Kafka, the four FastAPI microservices, and the gateway on `localhost:8000` — the gateway starts the live simulation itself and mirrors it to Redis/Kafka since both are reachable in this topology. Open `http://localhost:8000/` and the dashboard is already live; no extra step needed.

### 4. Train (optional — a pretrained policy is already checked in)

```bash
python train.py
```

Training produces `policy_rl.pt` (TorchScript). Regenerate the ONNX file the live path actually runs against with:

```bash
python scripts/export_onnx.py
```

### 5. Run the standalone simulation script (optional)

`simulate_live.py` is no longer required to see a live feed — the gateway runs the same loop itself. Run it directly only if you want a simulation independent of the gateway process (e.g. to feed Redis for other consumers, or to confirm the CLI path still matches):

```bash
python simulate_live.py
```

### 6. Open the dashboard directly (without docker-compose)

```bash
python -m uvicorn backend.gateway.main:app --port 8000
```

Visit `http://localhost:8000/`. With no Redis or Kafka configured, the gateway falls back to an in-process event queue — same feed, no broker required. It shows a brief `◌ CONNECTING` state while the first agents are seeded, then `● LIVE` and updates in real time.

## Validation results

Regenerated by `python simulate.py --sessions 4000 --seed 42`. Every number
below comes from the committed [`validation_report.json`](validation_report.json)
— nothing is hand-entered, and the run is seeded, so it reproduces exactly.

**n = 4,000 sessions.** Not 50: the simulation now models a *session* (one visit,
one terminal outcome) rather than treating an agent's entire lifetime log as one.

| Metric | Benchmark | Simulated | Verdict |
|---|---|---|---|
| Session length | within 4.0 of the real mean (8.16 ± 7.08) | 9.31 ± 5.55 | ✅ PASS (gap 1.15 of 4.0) |
| Conversion rate (all sessions) | 2–5% (industry e-commerce) | 3.07% | ✅ PASS |
| Cart abandonment (cart-reaching sessions) | 70–85% (Baymard Institute) | 81.3% | ✅ PASS |
| Conversion by shopper type | power ≥ regular ≥ browser | 8.6% / 1.9% / 0.0% | ✅ PASS |
| Social influence (top-5 concentration) | 0.20–0.40 (power-law, Amazon trend data) | 0.009 | ❌ **FAIL** — see below |

### What changed, and why the earlier numbers were not comparable

Four problems in the previous validation path have been fixed. Each one had been
inflating or invalidating a published figure.

1. **A "session" was an agent's whole lifetime.** `validation/metrics.py` treated
   each agent's complete log as one session, so every statistic had **n = 50**
   and "abandonment" meant "fraction of agents that never bought anything" —
   exactly `1 − conversion_rate` by construction. It is now one visit with one
   outcome, and abandonment is measured over sessions that *created a cart*,
   which is the denominator the industry figure uses. That is what took it from
   deliberately unscored to a scored PASS.

2. **The budget constraint never bound.** `simulate.py` priced products via
   `GET {gateway}/products/{id}`, a route the gateway does not expose. Every
   lookup 404'd, `get_price()` returned its `0.0` default, and
   `can_afford(0.0)` was always true — so no purchase in any published run was
   ever constrained by a wallet. Pricing now lives in `core/pricing.py` as a
   pure function of the product ID, which also covers all 2,985 vocabulary
   products rather than the 224 the catalog happened to carry.

3. **Conversion was roughly double the benchmark.** With the budget fixed and
   the funnel recalibrated (`Agent.advance_prob` halved), it sits at 3.07%
   inside the 2–5% band, down from 10%.

4. **Abandonment was a coin flip.** `Agent.should_quit()` drew against a fixed
   per-persona probability, independent of price, cart, wallet, stage and dwell.
   Nothing about a session predicted whether it would be abandoned, so no
   detector could have beaten chance on it. It is now a hazard — see below.

The old run-to-run variance section is gone because the cause is gone: the
simulation is seeded, and a given seed reproduces a run exactly.

### Why cart abandonment happens

Abandonment is now driven by pressures that accumulate during a session
(`core/friction.py`): the basket outrunning the wallet, stalling in a stage, and
bouncing between stages, against a latent purchase intent. Each lost session
records which pressure dominated — something a real deployment can never observe,
because the cause is latent.

| Driver | Share of cart abandonment | Can a discount move it? |
|---|---|---|
| Price | 42.2% | Yes |
| Hesitation | 22.1% | No |
| Churn | 18.3% | No |
| Intent | 17.4% | No |

Measured over 535 cart-abandoning sessions. **42.2% is the ceiling on what any
price intervention can recover** — the remaining 57.8% is lost for reasons a
discount does not address. Reporting that ceiling is the point: a recovery result
that approached 100% would mean the model had been built to be rescued.

### Known gaps

- **Trend concentration fails, and is out of scope.** The top 5 products capture
  0.9% of interactions against a 0.20–0.40 benchmark. With a 2,985-product
  vocabulary and 25% random exploration, demand spreads far more than it would
  during the category trend event that benchmark describes. This is a genuine
  mismatch between the model and the metric, left failing rather than tuned away.
- **Session length is not a like-for-like comparison.** The "real" figure is
  items per user sequence in the review data, which spans a user's whole review
  history rather than a single visit. It is the closest real quantity available,
  and the report says so in the metric's own `note` field.

## Repo layout

```
agents/          persona/agent model, BFS social graph, funnel gate, scheduler, BC + PPO trainers
                 policy_runtime.py: torch-free (ONNX) inference used by the live path
core/            shared domain: pricing, catalog, cart value, order rules, session model,
                 abandonment friction. Imported by both the HTTP services and the engine
backend/         FastAPI gateway + product/cart/order/session microservices
comms/           Redis and Kafka client helpers (both optional — see live/runner.py)
data/            data loading, persona clustering, social graph builder
live/            the live simulation loop (engine.py), its in-process fan-out (bus.py),
                 and the background runner the gateway starts on boot (runner.py)
validation/      behavioral metrics + benchmark verdicts
tests/           pytest suite (python -m pytest tests/)
visuals/         live dashboard (static HTML) and Vite/React dashboard
scripts/         export_onnx.py (policy_rl.pt → policy_rl.onnx) and check_parity.py
simulate.py      batch driver over live/engine.py → validation_report.json
simulate_live.py continuous simulation → Redis/Kafka live stream (standalone; optional)
train.py         BC + PPO training entry point
docker-compose.yml   Redis, Kafka, and the microservice cluster
```
