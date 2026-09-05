<div align="center">

# Revenue Recovery Engine

**Detecting at-risk checkouts, intervening once, and measuring honestly whether it worked.**

Razorpay AI Builder Buildathon · Track 03, Revenue Recovery

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-312%20passing-34D399)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-gateway%20%2B%20services-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-4%20tools-A78BFA)](recovery/mcp_server.py)

</div>

---

## The result

A 15% discount, offered once to sessions flagged as at risk, recovers a small
but real share of carts that would otherwise have been abandoned.

```
                    treatment    control      difference
A/B  (15% off)         5.31%      3.74%    +1.57%   z=+2.75   significant
A/A  (nothing)         3.96%      4.28%    -0.33%   z=-0.60   null
```

**+1.57%** &nbsp;·&nbsp; 95% CI **+0.45% to +2.69%** &nbsp;·&nbsp; bootstrap CI **+0.47% to +2.70%**
&nbsp;·&nbsp; pooled over **16 seeds**, 14 of them positive &nbsp;·&nbsp; n = 2,599 / 2,698

The **A/A check** is the number that makes the first one mean anything. Same
detector, same randomisation, same scheduling, nothing applied — and it returns
null, with 6 of 16 seeds positive, which is the coin-flip you want from noise.
Without it, the A/B is just a number the machinery produced.

> All order values are synthetic and the behavioural model is calibrated to
> published aggregate benchmarks, not fitted to real transactions.
> **The method transfers; the magnitudes are not a forecast about live traffic.**

---

## The problem this solves

Most cart-recovery work cannot answer *"how do you know it worked?"* You cannot
observe the counterfactual: the shopper you discounted either bought or did not,
and you never see what the same shopper would have done untreated.

A simulator can. Every session here records whether its loss was actually
price-driven — ground truth no production system ever has. That makes it
possible to measure not just whether an intervention moved outcomes, but the
ceiling on what any intervention *could* have moved.

---

## How it works

Four components over a multi-agent shopping simulation.

| | |
|---|---|
| **Detect** | Flags a cart-bearing session on either of two routes: a stall in a stage combined with hesitation, or a basket unusually large for that shopper's segment. Thresholds are derived from converting sessions, calibrated on a **held-out seed**. |
| **Diagnose** | Reads the detector's recorded features — basket against the segment, dwell, bounces, pause length — and names the dominant pressure. It never sees the true driver the simulation records; that field is kept to score the guess. |
| **Decide** | Picks the action the diagnosis calls for: a discount for price, a reminder for hesitation, a simplified checkout for churn. Each acts on the pressure it targets, so an action aimed at the wrong cause does very little. |
| **Intervene** | One bounded action per attempt, each with its own idempotency key. Dispatched into the event scheduler with latency, so it can arrive too late and miss. |
| **Escalate** | Cheapest rung first; a discount only follows if the free action landed and the session is still at risk. Stops on an attempt cap, an exhausted ladder, the spend guard, the kill switch, or the session closing. |
| **Measure** | Sessions randomised into arms **at flag time**. Two-proportion test with confidence intervals, minimum detectable effect, and an A/A null check. |
| **Audit** | Append-only SQLite, one row per decision, keeping the features the detector saw. Kafka's first consumer materialises the same records. |

**The intervention can only reach an outcome through the model.** A discount
lowers effective cart value, which lowers affordability pressure, which lowers
the quit hazard. There is no branch anywhere that marks a session recovered, and
a test asserts it.

### Detector performance

Calibrated on one seed, measured on another:

| precision | recall | base rate | **lift over base rate** |
|---|---|---|---|
| 94.8% | 51.7% | 73.7% | **1.29×** |

Precision is reported against the base rate deliberately. Cart abandonment is
~74%, so flagging *everything* would score 74% precision while carrying no
information at all. Lift is the number that means something.

---

## Evidence

### Dose-response

Eight discount rates, five seeds each.

| discount | effect | z | net margin | 95% CI on margin |
|---|---|---|---|---|
| 5% | +2.82% | +2.60 | 4,790 | 1,444 → 9,246 |
| 10% | +2.97% | +2.84 | 4,548 | 929 → 8,454 |
| 15% | +2.38% | +2.20 | 3,154 | −603 → 7,175 |
| 20% | +2.87% | +2.48 | 2,585 | −694 → 6,283 |
| 30% | +4.54% | +3.89 | 2,956 | −124 → 6,278 |
| 40% | +6.75% | +5.38 | 2,470 | −382 → 5,430 |
| 50% | +6.61% | +5.26 | 1,158 | −1,809 → 4,623 |

A bigger offer recovers more and earns less. A monotone dose-response is much
harder to produce by accident than a single significant cell.

**Break-even is not established.** Six of eight margin intervals straddle zero,
so no crossing can be located. An earlier single-run version of this sweep put
break-even at 34%; a properly pooled run at 40% came back solidly
margin-positive. That figure was sampling error and has been withdrawn.

### Net margin, not recovered revenue

The discount is billed on **every** treated conversion, including shoppers who
would have bought anyway. Gross recovered revenue counts the upside and ignores
the bill — and there are configurations here where a programme recovers carts
and loses money on every one. Margin carries a bootstrap interval because it
depends on the rate difference *and* on what the recovered baskets were worth.

### Does choosing the action beat one discount for everyone?

`python -m sim.playbook_experiment --sessions 15000` — 5 seeds, pooled.

| policy | actions | effect | net |
|---|---|---|---|
| flat 15% discount to everyone flagged | 2,457 | **+2.05%** (z=+3.27, significant) | **+6,186** |
| playbook — action chosen by diagnosis, escalating | 2,500 (942 escalations) | +0.49% (z=+0.84, not significant) | +782 |

**No.** The flat discount wins by 5,404 in net margin, and it is the only one of
the two whose effect is significant.

The reason is measurable, and it is the more useful finding. The playbook's
diagnosis is scored against the true driver it never sees:

```
guessed          churn   hesitation   intent    price
churn              615        1,169      489      112
hesitation          61          157      114       30
price              377          477      365      818

agreement  33.2% of 4,784 abandoned sessions
```

Four classes, so chance is 25%. **The diagnosis is barely better than chance.**
Nearly half of everything it calls churn is really hesitation. Routing to the
right remedy cannot work when the routing is close to a coin toss — and the
playbook duly spends less (1,657 against 5,111) and recovers far less (2,439
against 11,297).

The honest reading is that **detection and diagnosis are different problems.**
The detector finds at-risk sessions well (1.29× lift over base rate); telling
*why* a particular session is at risk, from the same observables, is much harder
and this one does not do it. That is a bottleneck worth naming rather than a
mechanism worth abandoning: better diagnosis would make the same playbook
worthwhile, and the escalation machinery, the caps and the audit trail are all
independent of it.

### Is the effect an artefact of the calibration?

Every one of ten behavioural constants perturbed ±20%, five seeds per cell:
**the effect never turns significantly negative.** Nineteen of twenty cells stay
positive; the largest swing is 1.88pp.

### Do the detector's thresholds transfer?

Calibrated in one world, evaluated in four materially different ones:

| world | lift (home thresholds) | lift (native) | loss |
|---|---|---|---|
| home | 1.286 | 1.286 | 0.000 |
| price-sensitive | 1.326 | 1.319 | −0.007 |
| patient shoppers | 1.300 | 1.298 | −0.002 |
| expensive delivery | 1.114 | 1.115 | 0.000 |
| impulsive | 1.332 | 1.355 | +0.023 |

Thresholds beat the base rate in every world, worst transfer loss 0.023 lift.
This is robustness across *simulated* worlds, not evidence about real shoppers —
every world shares the same generative model.

### Behavioural validation

| metric | benchmark | simulated | |
|---|---|---|---|
| Session length | within 4.0 of real 8.16 | 9.45 | PASS |
| Conversion rate | 2–5% | 3.79% | PASS |
| Cart abandonment | 70–85% | 77.7% | PASS |
| Conversion by segment | power ≥ regular ≥ browser | 10.6 / 2.4 / 0.0% | PASS |
| Trend concentration | 0.20–0.40 | 0.008 | **FAIL** |

Trend concentration is left failing. With a 2,985-product catalogue and 25%
exploration, demand cannot concentrate the way a category trend event does —
a genuine model/metric mismatch rather than something to tune away.

---

## What did not work

Reported because the failures are more informative than the successes, and
because a project that only lists wins has not been measured.

**Uplift targeting does not pay.** Predicting *treatability* rather than risk is
the textbook answer to a detector that wastes two treatments in three. The model
learns the ranking — at a 40% discount it puts 29.4% recoverable losses in its
top bucket against 7.7% in the rest — but the derived decision rule loses to
simply treating everyone flagged, at both 15% (3,159 vs 6,900 net) and 40%
(8,706 vs 9,279). The ranking works; the policy does not convert it into money.

**A reminder that only relieved quit pressure made things worse.** Actions
originally lowered the pressure they targeted and nothing else, so a reminder
kept a stalled session alive without moving it forward — straight into the
action cap, counted as abandoned. Measured at −3.41% (z=−2.56) before actions
were given a progression effect as well. A real reminder pushes a shopper
onward; modelling it as merely a reason to stay was wrong.

**Four claims were withdrawn during development**, all for one reason: a single
5,000-session run carries a standard deviation near 1.7pp, so one run measures
the seed as much as the effect.

| claim | first reported | after pooling |
|---|---|---|
| Price-shock targeting | +6.17% | +0.89%, not significant — the rest was selection bias |
| Sensitivity sign-flips | 5 of 10 parameters flipped | 0 — all noise |
| Break-even discount | 34% | not established |
| Uplift policy | "beats treating everyone" | it does not |

Every comparison in the repository is now pooled across seeds.

---

## Try it

```bash
pip install -r requirements.txt
python -m uvicorn backend.gateway.main:app --port 8000
```

Open **http://localhost:8000/** — the dashboard shows the pooled result and its
A/A check, the margin curve with its uncertainty band, what is actually losing
carts, and an audit trail you can click into to see why any one session was
flagged. The sliders re-measure a different offer on demand: the page runs a real
experiment in a subprocess and returns a measured delta with an interval.

Reproduce the numbers directly:

```bash
python -m sim.robustness      --seeds 16 --sessions 5000   # the headline
python -m sim.experiment      --aa                          # the null check
python -m sim.sensitivity     --workers 8                   # is it calibration?
python -m sim.transfer                                      # do thresholds move?
python -m sim.precompute_grid                               # dose-response
python -m sim.playbook_experiment --sessions 15000           # action selection
```

Everything is seeded and reproduces exactly. Independent experiments run across
processes — which is also required for correctness, since the simulation reseeds
the global RNG.

### Ask it questions

```bash
claude mcp add recovery -- python -m recovery.mcp_server
```

| tool | answers |
|---|---|
| `explain_decision` | why this shopper got an offer, and how the session ended |
| `experiment_summary` | arms, effect, CI and margin, recomputed from the audit trail |
| `abandonment_drivers` | what is losing carts, and how much a discount could reach |
| `run_counterfactual` | **measures a different offer** and returns a CI |

`experiment_summary` reproduces the published headline from the log alone, which
is the check that the audit trail is a complete record rather than decoration.
No tool can affect production: the read tools sit on an append-only table, and
`run_counterfactual` starts a simulation.

---

## Path to production

**Where it plugs in.** `integrations/razorpay_webhooks.py` maps documented
webhook events onto the funnel stages the pipeline already models.
`payment.failed` is the event that matters: the shopper reached checkout, tried,
and did not finish. Signature verification gates the mapping, and amounts are
converted from paise. *It has never seen a live webhook* — it is written against
the published payload shape and tested against fixtures.

**Controls before it touches money.** A kill switch that takes effect without a
deploy and records who threw it; a rate limit; a spend cap; and a ceiling on any
single discount. Every refusal is counted and attributed. These are in-process
and reset with the process — adequate for a simulation, **not** for production,
where the cap must hold across instances.

**Data governance.** Every detector feature is behavioural — dwell, revisits,
basket size. No customer identity crosses the ingestion boundary, and a test
asserts the audit schema carries no product identifiers or free text. Decisions
are replayable from stored features with reason codes, which is what a DPDP-era
review actually asks for.

**What would have to be true.** A real clickstream with view/cart/purchase
labels, to validate the detector on sessions it did not generate. That is the
honest answer to "this is a simulator measuring itself", and this repository does
not contain it.

---

## Limitations

- **Synthetic order values.** Prices are an md5 of the product ID.
- **Calibrated, not fitted.** The model reproduces published aggregate
  benchmarks; it is not fitted to any real transaction data.
- **Recovery is under-modelled.** ~2.9% of would-be-abandoned carts against a
  published ~10% for real campaigns.
- **Robustness ≠ realism.** The transfer test varies coefficients within one
  generative model.
- **The magnitudes do not transfer.** The method does.

---

## Architecture

```
                    Amazon reviews (701K)  ─→  BC + PPO  ─→  policy_rl.onnx
                                                                   │
   ┌───────────────────────────────────────────────────────────────▼─────────┐
   │  live/engine.py — sessions · simulated clock · seeded · funnel gate     │
   │  core/ — pricing · cart · orders · session · friction (the quit hazard) │
   └───────┬────────────────────────────────────────────────────┬────────────┘
           │ events                                    scheduler │ nudge
   ┌───────▼──────────┐  ┌──────────────┐  ┌──────────────┐  ┌───┴─────────┐
   │ recovery/detector│─→│ assign (arm) │─→│ intervention │─→│  scheduler  │
   └──────────────────┘  └──────────────┘  └──────┬───────┘  └─────────────┘
                                                   │
   ┌───────────────────────────────────────────────▼─────────────────────────┐
   │  audit (append-only SQLite)  →  Kafka `recovery_events`  →  analysis     │
   └──────────────────────┬──────────────────────────────┬───────────────────┘
                          │                              │
                  FastAPI gateway ─→ dashboard    MCP server ─→ agents
```

`core/` is shared by the HTTP services and the simulation engine, so the two
cannot drift apart — they had, and it hid a bug where every price was 0.00.

```
core/          pricing, catalog, cart, orders, session, friction
recovery/      detector, assign, intervention, playbook, analysis, audit,
               stream, uplift, safety, jobs, mcp_server
sim/           experiment, robustness, sensitivity, transfer, precompute_grid,
               uplift_experiment, playbook_experiment, parallel
integrations/  razorpay webhook adapter
agents/        agent model, funnel, scheduler, social graph, policy runtime
live/          engine, runner, in-process bus
backend/       FastAPI gateway + product/cart/order/session services
visuals/       recovery dashboard, simulator dashboard, generated charts
results/       committed output of every harness — see results/README.md
tests/         312 tests
```

The dashboards are single-file pages that need no build step: open the gateway
and they run. There is deliberately no npm toolchain in this repository.

---

<div align="center">
<sub>Built on the RL-USERS multi-agent shopping simulator.<br/>
Synthetic data throughout — the method is the deliverable, not the magnitudes.</sub>
</div>
