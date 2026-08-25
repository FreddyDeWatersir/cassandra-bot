# CassandraBot

A multi-model ensemble forecasting bot competing in the Metaculus FutureEval Bot Tournament, a bot-only competition where entrants predict real-world events across economics, geopolitics, science and technology.

The bot has run autonomously across two seasons. Every 30 minutes a GitHub Actions job checks for newly opened tournament questions, routes each one through six frontier models under three different reasoning frames, aggregates their forecasts in log-odds space, applies a calibration adjustment scaled to the question's time horizon, and posts both the forecast and its reasoning back to Metaculus via API. No human reviews or overrides any forecast, which is a tournament rule and also the point: the interesting question is whether the scaffolding around the models improves on the models alone.

Forecasts are scored by **spot peer score**, a proper log-based scoring rule measured against every other bot in the tournament. Because it is proper, you maximise it by reporting your true belief. Because it is peer-relative, it rewards being right when others are wrong and punishes confident errors heavily. That scoring rule shapes most of the design decisions below.

---

## Operating record

| | |
|---|---|
| Seasons run | Two (Spring 2026, Summer 2026) |
| Deployment | GitHub Actions, polling every 30 minutes, unattended |
| Ensemble | Six models across three providers |
| Question types | Binary, numeric, multiple-choice, date, conditional |
| Cost | ~$0.80 per question, ~$159 across the Summer season |

**On accuracy numbers:** I am not citing a peer score yet, and the reason matters more than the number would. Summer questions resolve through early September, and most tournament questions resolve at the end of a season, so a mid-season score is dominated by which questions happened to close early. Metaculus documents bots moving ten or more leaderboard places in the final days. Separately, the research layer was degraded for most of the season (see below), so a score from that period measures a pipeline I have already diagnosed as broken rather than the design described here.

I would rather show the reasoning and the failure analysis now and report a score once the season closes and the fix has run, than quote a number I would have to caveat into meaninglessness.

---

## Architecture

```
                    ┌─────────────────────────┐
   Metaculus API ──▶│  new open questions     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  RESEARCH (tiered)      │
                    │  1. AskNews live search │
                    │  2. LLM fallback        │
                    │  3. Foresight reasoning │
                    └────────────┬────────────┘
                                 │  research context
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
   ┌─────▼─────┐           ┌─────▼─────┐           ┌─────▼─────┐
   │  OUTSIDE  │           │  INSIDE   │           │ ADVOCATE  │
   │  T = 0.2  │           │  T = 0.4  │           │  T = 0.7  │
   │           │           │           │           │           │
   │ base rate │           │  causal   │           │ argue vs  │
   │ status quo│           │ mechanism │           │ consensus │
   └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
         │                       │                       │
    foresight-v3              o3                    gpt-5.4
    opus-4.6              sonnet-4.6                o4-mini
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │  6 independent forecasts
                    ┌────────────▼────────────┐
                    │  AGGREGATE              │
                    │  trim min + max         │
                    │  mean of log-odds       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  EXTREMIZE              │
                    │  factor scaled by       │
                    │  days to resolution     │
                    └────────────┬────────────┘
                                 │
                    Metaculus  ◀──┘  forecast + reasoning comment
```

Handles binary, numeric, multiple-choice, date and conditional questions. Six models run in parallel behind a concurrency limiter, each wrapped with three-attempt exponential backoff. A model that fails all three attempts is dropped from the ensemble for that question rather than failing the forecast, so a provider outage degrades the ensemble instead of breaking it.

---

## Design decisions

The interesting parts are not the LLM calls. They are the four places where a naive implementation gives you a worse forecast.

### Aggregate in log-odds, not probability space

Averaging probabilities directly is the obvious move and it is wrong. The arithmetic mean of 1% and 10% is 5.5%, which treats a tenfold difference in odds as a small gap. Forecast disagreement is multiplicative in the tails, so the mean is taken over log-odds and mapped back through the logistic:

```python
logits = [math.log(p / (1 - p)) for p in clipped]
return 1.0 / (1.0 + math.exp(-sum(logits) / len(logits)))
```

This matters most exactly where peer scoring punishes hardest. A log-based scoring rule assigns enormous penalties to confident wrong answers near 0 and 1, which is the region where probability-space averaging distorts most.

### Trim before aggregating

With five or more responses, the highest and lowest forecasts are discarded before aggregation. Parsing a probability out of free-form model reasoning occasionally goes wrong (a model states a conditional probability mid-argument, or the regex catches the wrong percentage), and a single spurious 95% shifts a log-odds mean substantially. Trimming costs a little information and buys robustness against parse failures I cannot fully prevent.

### Extremize, because ensembles are systematically underconfident

Averaging any set of forecasts pulls the result toward 0.5. Each model already hedges, and averaging hedged forecasts compounds the hedging, so the ensemble is less confident than any justified belief. The standard correction raises the odds to a power:

```python
odds = prob / (1 - prob)
result = odds ** factor / (1 + odds ** factor)
```

This is well established in the forecasting literature and is roughly the highest-return single intervention available for LLM forecasting calibration.

### Scale extremization by horizon (the part I have not seen elsewhere)

A fixed extremization factor applied to every question is wrong in a specific way. It assumes all underconfidence is an artifact of averaging. But on a question resolving in eighteen months, a model's uncertainty is often *correct*: the world genuinely has not decided yet. Sharpening those forecasts converts real uncertainty into false confidence, and under a log scoring rule that is the most expensive mistake available.

So the factor decays with the resolution horizon: full strength (1.4) inside 30 days, decaying linearly, and by a year out retaining only a quarter of the adjustment.

```python
if days <= 30:   return base
if days >= 365:  return 1.0 + (base - 1.0) * 0.25
frac = (365 - days) / (365 - 30)
return 1.0 + (base - 1.0) * (0.25 + 0.75 * frac)
```

The thresholds are reasoned, not fitted, and that is a real weakness. Validating this against flat extremization on the same question set is the main experiment for next season, and if it does not beat flat, it is complexity with no payoff and should be deleted.

### Diversity by frame, not just by model

Six models given identical prompts produce correlated errors, because they share training data and failure modes. Averaging correlated forecasts buys you very little. So prompts are cycled across three genuinely different reasoning procedures, with sampling temperature matched to each:

| Frame | Temp | Instruction |
|---|---|---|
| Outside view | 0.2 | Find the reference class, anchor on base rates, weight the status quo heavily |
| Inside view | 0.4 | Trace causal mechanisms, current trajectory, what specifically must happen |
| Devil's advocate | 0.7 | State the consensus, then argue against it, surface what others underweight |

Low temperature for base-rate reasoning where you want the modal answer, high temperature for adversarial reasoning where you want the unusual scenario. The intent is decorrelated errors rather than six restatements of the same prior.

---

## What broke, and what it taught me

The research layer failed during Summer in a way I think is more instructive than any of the parts that worked.

AskNews access lapsed at the season boundary, so the pipeline fell through to its second tier: an LLM asked to summarise relevant news. That model has no web access. Rather than reporting that it could not help, it produced fluent, confident, well-structured research describing market conditions from its training data and presented them as current. That text then entered all six forecasting prompts under the heading `Your research assistant says:`.

Extract from a real run, forecasting a Bitcoin price question in August 2026:

> "as of October 2023 ... Bitcoin fluctuating between $25,000 and $35,000"

The actual price that week was around $77,000.

Two things are worth drawing out. The first is that the failure was invisible in the logs. Every stage reported success. `AskNews research failed` appeared as a warning, the fallback returned 2,147 characters of plausible text, and the bot posted a forecast with no error. A monitoring setup that checks whether stages completed would have shown a fully green pipeline.

The second is that the ensemble partially saved itself. The models mostly ignored the false context and reasoned from figures in the Metaculus question background instead. That is luck, not design: on a question with a thinner background there is nothing to correct against, and six frontier models would have confidently anchored on a fabrication.

The lesson I actually took is about failure modes rather than about news APIs. A fallback that silently produces *worse-than-nothing* output is more dangerous than one that fails loudly, because a degraded output looks exactly like a working one. The fixes are a search-backed fallback so the degraded path still touches reality, provenance logging so every forecast records which research tier produced its context, and a fail-loud path when no tier had live search.

---

## Known limitations

- **Extremization thresholds are hand-chosen.** Reasoned rather than fitted, and unvalidated against a flat factor.
- **No offline validation harness.** Every design choice here is currently justified by reasoning rather than measurement on held-out questions. A pastcasting harness (forecasting resolved questions the models have not seen) is the obvious missing piece, and until it exists the calibration work below is argued rather than demonstrated.
- **No per-member attribution.** Individual model forecasts are logged but not persisted in structured form, so I cannot yet decompose ensemble error into member disagreement versus shared bias.
- **Ensemble size is unjustified.** Six models were chosen for coverage, not measured contribution. It is entirely possible four perform comparably at half the cost, and I do not currently have the data to know.
- **Cost tracking is broken.** The direct API wrappers bypass litellm, so the framework's cost manager reports $0.00. Real spend is only visible through the provider dashboard.
- **Parsing is regex over free text.** Layered fallbacks handle the common formats, but structured output would be more robust.

---

## Running it

```bash
poetry install
cp .env.template .env      # add METACULUS_TOKEN, OPENROUTER_API_KEY, others
poetry run python main.py --mode test_questions
```

Deployment is GitHub Actions on a schedule. Tournament questions open for a few hours at unpredictable times, so a polling cron is the appropriate design rather than a persistent service.

| File | |
|---|---|
| `main.py` | Bot, ensemble, aggregation, calibration, prompts |
| `.github/workflows/` | Scheduled runs |

Built on the Metaculus [forecasting-tools](https://github.com/Metaculus/forecasting-tools) package, which handles the API client and question objects. Started from the tournament's template bot; the ensemble, aggregation, calibration and research pipeline are my own.