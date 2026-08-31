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

## The recovery pipeline

The simulator above is the substrate. The thing built on it is a
**detect → intervene → measure → audit** loop for checkout abandonment, and the
point of building it inside a simulation is that a simulation has something
production never does: **counterfactual ground truth.** Every flagged session
that goes untreated plays out anyway, so we can see what it *would* have done,
and every abandonment records which pressure actually caused it. A live system
sees neither.

```
        events ──►  recovery/detector.py     stall+hesitation, or price shock
                          │                  thresholds calibrated on a held-out seed
                          ▼
                    recovery/assign.py       50/50 at flag time, hashed not drawn
                       │        │
                 control│        │treatment
                        │        ▼
                        │   recovery/intervention.py   one bounded discount,
                        │        │                     dispatched with latency
                        │        ▼
                        └──►  core/friction.py   discount lowers the bill, which
                                  │              lowers the quit hazard. No path
                                  ▼              sets an outcome directly.
                    recovery/analysis.py     z-test, 95% CI, MDE, net margin
                                  │
                                  ▼
                    recovery/audit.py ──► SQLite (source of truth)
                                  └─────► Kafka `recovery_events` ──► recovery/stream.py
```

**Detector** (`recovery/detector.py`). Flags a cart-bearing session on either of
two routes: a stall in a stage combined with hesitation, or a basket unusually
large for that shopper's segment. Thresholds are derived from the 75th
percentile of behaviour in *converting* sessions, calibrated on a different seed
from the one measured — tuning a threshold on the sessions you then score it
against is fitting to the test set.

**Intervention** (`recovery/intervention.py`). One discount per session, enforced
by an idempotency key rather than by a comment claiming it. Scheduled into the
same event heap the agents use, so it arrives late and can miss entirely if the
session has already ended.

**Measurement** (`recovery/analysis.py`, `sim/experiment.py`). Arms are randomised
at flag time, compared like-for-like, and reported with a confidence interval and
a minimum detectable effect. `--aa` runs the identical machinery with nothing
applied; it has to come back null.

**Audit** (`recovery/audit.py`, `recovery/stream.py`). One append-only row per
decision, keeping the features the detector actually saw so a decision can be
explained months later. The experiment recomputes its headline numbers from the
log and prints whether they reconcile.

### Asking the pipeline questions (MCP)

`recovery/mcp_server.py` exposes the pipeline as MCP tools, so an agent can be
asked about it in words rather than SQL.

```bash
claude mcp add recovery -- python -m recovery.mcp_server
```

| Tool | Answers |
|---|---|
| `explain_decision` | "Why did this shopper get an offer?" — the trigger, the dwell/revisits/basket that fired it, the idempotency key, and how the session ended |
| `experiment_summary` | Arms, effect, 95% CI, MDE and net margin, recomputed from the log |
| `abandonment_drivers` | What is losing carts, and how much of it a discount could ever reach |
| `run_counterfactual` | **Simulates a different offer and measures it** — a new discount rate, dispatch latency or targeting rule, returned with a confidence interval |

The last one is why this is a server rather than a database wrapper. The harness
underneath it is validated — the A/A comes back null and the A/B does not — so
"what if we offered 20% instead of 15%" gets a measured answer instead of a
guess.

`experiment_summary` reproduces the published headline exactly from the audit
trail alone, which is the check that the record is complete rather than
decorative.

**Nothing here can touch production.** The read tools sit on an append-only
table with no UPDATE or DELETE; `run_counterfactual` starts a simulation and
returns statistics. There is no path from a tool call to a live shopper, a real
offer, or money, and a test asserts the module imports no HTTP, socket or
subprocess machinery. Session counts are bounded so a tool call cannot run for
an unbounded time.

Tool logic is tested directly; the stdio transport is not, since no MCP client
runs in CI.

### Two properties that make the result trustworthy

**Arm assignment consumes no randomness.** The whole simulation runs off one
seeded stream, so drawing a number to pick an arm would shift every later draw
and the control arm would stop being the run you would have got with no
experiment at all. `recovery/assign.py` hashes the session id instead.

**A discount can only reach an outcome through the bill.** It lowers the
effective cart value, which lowers the quit hazard and the willingness to
proceed. There is no branch anywhere that marks a session recovered, and a test
asserts it.

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

## What we measured

`python -m sim.experiment --sessions 43000 --budget-aware`, thresholds calibrated
on seed 7 and measured on seed 42. Both runs are seeded and reproduce exactly.

| | treatment | control | delta |
|---|---|---|---|
| **A/B** — 15% discount, 0.5s dispatch | 6.07% | 3.25% | **+2.82%** |
| **A/A** — identical machinery, nothing applied | 3.89% | 4.43% | −0.54% |

```
A/B  +2.82%  (95% CI +1.29% to +4.35%, z=+3.61, n=1433/1476)  SIGNIFICANT, MDE 1.84%
A/A  -0.54%  (95% CI -1.98% to +0.90%, z=-0.74, n=1467/1491)  not significant, MDE 2.12%
```

**The A/A is the load-bearing number.** The same detector, the same arm
assignment, the same scheduling — with nothing applied — reports no effect. That
is the evidence the harness measures rather than manufactures, and without it
the A/B would just be a number the machinery produced. It matters here
specifically because treated sessions perturb a shared random stream, so
"the arms are comparable" is a claim that has to be tested rather than assumed.

### Net margin

Gross recovered revenue is the wrong number, and reporting it alone would have
called several earlier failures a success. Every treated session that converts
is billed at a discount, including the shoppers who would have bought anyway.

```
incremental conversions      40.4
gross recovered           8,936.47
discount cost             2,808.56
NET                       6,127.91    positive
```

At a 15% discount the programme clears its own cost. It does not always: on an
earlier configuration the same arithmetic gave cost at 2.4x recovered revenue
and a net of −1,374.62. Which side of that line a campaign lands on depends on
the discount rate and on how much of the treated population could ever respond —
which is why the detector's *recoverable precision* (roughly a third of
treatments, flat across every operating point) is reported alongside its
accuracy.

### What this does and does not show

The recovery rate is **2.9%** of carts that would otherwise have been abandoned.
Published cart-recovery campaigns are usually quoted nearer 10%, so this model
under-recovers by roughly a factor of three even after calibration. The effect is
real and significant; the magnitude is not a prediction about Razorpay traffic.

All order values are synthetic — product prices are an md5 of the ASIN — and the
behavioural model is calibrated to published aggregate benchmarks rather than
fitted to real transactions. **The method transfers; the magnitudes do not.**

Every number above is recomputable from the append-only audit trail, and the
experiment prints whether the log and the analysis agree. They do.

## What did not work, and why

Four attempts to make a discount move the needle failed before one worked. They
are documented because the failures locate the problem more precisely than the
fix does, and because each one looked reasonable beforehand.

**1. Raise the price coefficient in the quit hazard.** No effect at any value.
Conversion turned out to be gated on *reaching* checkout, so a lower quit hazard
only bought a session more time to wander until it hit the action cap. Survival
is not conversion.

**2. Make price drag on funnel progression.** Also no effect. A smooth channel
moves in proportion to the price change: 15% off a basket shifted conversion by
about a quarter of a point. Only a threshold — a price above which a shopper
will not buy — makes a discount matter, and then only for shoppers it carries
across that line.

**3. A reservation price with a soft band.** Correct mechanism, no population.
Measurement showed baskets were landing at a median **2.04x** what the shopper
would pay: far past anything a 15% offer could bridge. Only **5.7%** of flagged
carts sat close enough to the line to be crossable.

The cause was upstream of the friction model entirely. The policy chooses
products with no notion of price — its state is three product ids, its reward is
sequence plausibility, and price is an md5 of the ASIN, statistically
independent of anything the model can see. Baskets were being assembled with no
reference to a budget.

**4. Budget-aware baskets.** Fixed the cart-to-reservation ratio (2.04x → 0.89x)
and immediately traded one blocker for another: shoppers who only add what they
can afford do not abandon over price. Conversion rose to 9.18% and cart
abandonment fell to 45.3%, both far out of band, and no value of the bias weight
recovered them.

**5. What worked: a checkout fee shock.** The most-cited real reason carts are
abandoned is extra costs at checkout. Shipping and tax are now revealed *after*
the shopper has committed to a subtotal they were happy with, which is how it
works in life — cart pages show subtotals, checkout shows the rest. An
affordable basket becomes an unaffordable total, and that is exactly the band a
discount can carry back across.

A near miss worth recording separately. Targeting price-shock flags appeared to
give **+6.17%** until the comparison was checked: it was pitting price-shock
*treated* sessions against *all* control sessions, and price-shock sessions
convert at 13–18% in control against 8% overall. The entire effect was selection
bias. Like-for-like it was +0.89% (z=0.17).

## Path to production

**Where this would attach.** The detector consumes an event stream; Razorpay
already emits one. `payment.failed` and `order.paid` webhooks, plus Magic
Checkout's abandonment events, carry the stage transitions and basket values
this model reads. The intervention would dispatch through the Offers API rather
than by mutating a simulated cart. Nothing in the pipeline needs the simulator
once it is calibrated — the simulator is how you choose thresholds before
touching live traffic.

**What the simulation is for.** Calibrating a detector against counterfactual
ground truth, which production cannot provide. You cannot observe what a flagged
session would have done if you always treat it, and you cannot observe why
someone left. Here you can do both, which is the argument for calibrating in
simulation and shipping the thresholds.

**Honest limits.** Every number here comes from synthetic data with declared,
synthetic order values. Product prices are an md5 of the ASIN. The behavioural
model is calibrated to published aggregate benchmarks, not fitted to Razorpay
traffic, so the *magnitudes* are illustrative and only the *method* transfers.
The detector's latency is optimistic: the simulation observes a session only
when it acts, so a stall is detected on the action that ends it rather than
while it is happening.

**Data governance.** Every detector feature is behavioural — dwell, revisits,
basket size, segment. No product identifiers, no free text, nothing identifying
a shopper reaches the audit record, and a test asserts the schema stays that
way. That is a design property rather than a claim, and it is what would let
this shape of record survive a DPDP review. Real deployment would additionally
need: consent boundaries for behavioural profiling, RBI data-localisation for
anything touching transaction data, a rate limit and a kill switch on the offer
engine, and reason codes on every decision — the last of which the audit trail
already carries.

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
recovery/        detector, arm assignment, intervention, analysis, audit trail,
                 Kafka consumer, and the MCP tool server
sim/             batch drivers: experiment, sweeps, detector evaluation
validation/      behavioral metrics + benchmark verdicts
tests/           pytest suite (python -m pytest tests/)
visuals/         live dashboard (static HTML) and Vite/React dashboard
scripts/         export_onnx.py (policy_rl.pt → policy_rl.onnx) and check_parity.py
simulate.py      batch driver over live/engine.py → validation_report.json
simulate_live.py continuous simulation → Redis/Kafka live stream (standalone; optional)
train.py         BC + PPO training entry point
docker-compose.yml   Redis, Kafka, and the microservice cluster
```
