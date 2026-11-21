# Semiconductor Affairs: LSTM lead-lag stock prediction

Working doc. It holds the context, the decisions and the next steps. Update it as we go.

## 1. Why this project

Main goal: **learn LSTMs**, both how to use them and how they work inside (gates, hidden state, backprop through time).
Second goal: **deploy a model behind a web page**, later on.

Stock prediction is the use case, not the goal. A careful negative result ("the LSTM doesn't beat a linear baseline") is a fine
outcome, as long as we can explain *why*.

## 2. Hypothesis

Stocks in one industry are linked through the supply chain. News hits the large, closely watched names first (ASML, TSMC,
Lam, Applied Materials, Nvidia). Smaller, less-covered companies further down the chain react **with a lag**.
If that lag is around an hour or longer, the recent returns of the leaders should help predict the follower's next-hour return.

There is literature behind this: large caps lead small caps (Lo & MacKinlay, 1990), and the effect is stronger inside one
industry (Hou, 2007). Most of that evidence is at **daily/weekly** horizons, and the effect is strongest in exactly the
illiquid small caps where trading costs are highest. Expect a weak signal at best.

## 3. Target and inputs

### Recommended target: a US small-cap equipment *subsystem* supplier

| Candidate | What they do | Why it's a good "last responder" |
|---|---|---|
| **ICHR** (Ichor) | Fluid delivery subsystems for etch/deposition tools | Revenue heavily concentrated on Lam Research and Applied Materials. A clear causal chain: TSMC capex → LRCX/AMAT orders → Ichor |
| UCTT (Ultra Clean) | Gas/fluid subsystems, parts cleaning | Same customer concentration, a bit larger |
| KLIC (Kulicke & Soffa) | Packaging/bonding equipment | Direct BESI competitor, so the advanced-packaging/hybrid-bonding story |

**First pick: ICHR.** Reasons: (1) it's small (roughly $0.5–1B, check this) with little analyst coverage, (2) its customer
dependency is explicit and documented, and (3) it trades in **the same hours as all its drivers**, which removes most of the
timezone problem (see §5).

The European option (Suss MicroTec `SMHN.DE`, Aixtron `AIXA.DE`, Soitec `SOI.PA`, alongside `ASML.AS`, `BESI.AS`, `ASM.AS`)
fits the original framing better. The catch is that the most important US drivers only overlap with European trading for
~2h a day. Keep it as a v2 experiment.

### Inputs (v1)

All are US-listed, so every series shares one clock:

- **Target's own history**: `ICHR`
- **Direct customers**: `LRCX`, `AMAT`
- **Equipment peers**: `KLAC`, `ASML` (US ADR)
- **Demand side**: `TSM` (US ADR), `NVDA`
- **Peer small caps**: `UCTT` (moves with ICHR, useful as a sanity check)
- **Sector and market**: `SOXX` or `SMH` (semi ETF), `QQQ`

To check in step 2: customer concentration figures and current market caps (from 10-K filings).

### Target variable

Next-hour **log return** of ICHR, as a percentage. Variant to try later: the **excess return** over the semi ETF
(ICHR minus β·SOXX). That strips out "the whole sector moved" and leaves the lag effect we actually care about.

## 4. Data

- **Source: Yahoo Finance via `yfinance`.** It's the only free source with hourly bars and needs no API key. Downsides: it's
  unofficial (it scrapes Yahoo's endpoints), it can break or rate-limit, and **hourly data only goes back ~730 days**.
- In practice the limit is ~730 *trading* days, so ≈ **5,000 hourly samples per ticker**. That's *small* for an LSTM, so we'll
  need to watch overfitting closely.
- **Building up history:** every ingest run fetches the full ~730-trading-day window and merges it, so we only lose
  data if we don't run it for ~2.9 years. **No scheduler or server needed for now.** If wanted: a systemd user timer with
  `Persistent=true` (catches up after the laptop was off). A server only becomes necessary for live predictions (step 9).
- **More granular data (not needed now; prices from memory, so check them):**
  - Yahoo (free): 5–30-min bars for the last ~60 days, 1-min bars for the last ~7–30 days. Only useful if collected
    continuously going forward, which *would* need a scheduler.
  - Alpaca (free tier): years of 1-min history, but only from one exchange (IEX), a small part of all trading. Full data is paid.
  - Polygon.io and similar: roughly $30–80/month for years of minute bars; tick data costs more.
  - Databento: pay per amount of data downloaded. Good for a one-off historical download.
  - Trade-off: 60× more rows and lead-lag effects are more likely at minute scale, but more bid-ask-bounce noise.
  - A paid provider is a new vendor, so it goes through Samotics' security review first.
- Storage: **Parquet** files, one per ticker. Parquet keeps types and timezones and is small. No database yet, because
  there's no concurrent access to manage.
- Store everything in **UTC**. Use adjusted prices (splits/dividends).
- Samotics note: this runs on a work laptop. New third-party packages and data sources may need a security review under
  Samotics policy. Check before relying on it. Only public market data comes *in*; no company data goes out.

## 5. Pitfalls / what the first plan was missing

1. **Show that the signal exists before building an LSTM.** Start with cross-correlations of leader returns at lag *k*
   against the target's return. If there's no lagged relationship at all, the LSTM can't find one either.
2. **Baselines.** Without them, any LSTM result means nothing. Compare against: predict 0, predict last return, and ridge
   regression on the same lagged features. The LSTM has to beat the *linear* model to justify itself.
3. **Timezones / market hours.** Bars from different exchanges only line up where trading overlaps. The first bar of each
   day contains the overnight gap, which behaves very differently from intraday bars. Add a "first bar of day" feature, or
   model it separately.
4. **Bar alignment and leakage.** Yahoo timestamps hourly bars at their *start*. The last US bar is a half hour
   (15:30–16:00). Only use information that's complete by the time of the prediction.
5. **Time-series validation.** No random splits. Use chronological train/val/test and walk-forward evaluation. Fit
   scalers on train only.
6. **Use returns, not prices.** Prices aren't stationary. Use log returns, plus normalised volume as an extra feature.
7. **Metrics that matter here.** MSE alone hides everything because the noise dominates. Also track directional accuracy,
   information coefficient (rank correlation of prediction vs actual) and a toy long/flat backtest **with trading costs**
   (small caps have wide spreads).
8. **Non-stationarity.** Market regimes change (the AI boom, export controls). A model trained on 2024–25 may not hold in
   2026. Walk-forward evaluation makes this visible.

## 6. Tooling (planned)

- **uv**: fast, lockfile-based environments. It fits a small project.
- **pandas + pyarrow**: time-series wrangling and Parquet.
- **PyTorch**: we write the training loop and later an LSTM cell by hand, so we see the mechanics. Keras would hide them.
- **scikit-learn**: baselines (ridge) and scaling only.
- Later: **FastAPI** to serve predictions (small, typed, auto docs). Docker once there's a service to reproduce.
  Experiment tracking only once we're comparing many runs.

## 7. Roadmap

- [x] **0. Setup**: uv project, `src/` layout, ruff.
- [x] **1. Ingest**: `src/semiconductor_affairs/ingest.py` fetches hourly bars into `data/raw/<TICKER>.parquet` (UTC),
      merging with what's stored. Inspection in `notebooks/01_inspect_raw_data.ipynb`. Findings (2026-11-19):
      - The limit is ~730 *trading* days: ~2.9 years, **~5,070 bars** per ticker (Dec 2023 → now).
      - 7 bars/day, labelled by start time 09:30–15:30 NY; the 15:30 bar is a half hour.
      - Early-close days have 3 bars; 2026-11-05 has 6 (a hole in Yahoo's data); a mid-session fetch gives a partial
        day (overwritten on the next run).
      - Almost perfectly aligned: only 1 bar missing (UCTT, 2025-10-09 17:30 UTC).
      - **First bar of the day has ~3–4× the return std of other bars** (overnight gap): ICHR 4.4% vs ~1.0%.
      - ICHR median dollar volume ≈ **$1.3M/hour** (UCTT 1.7M) vs LRCX 93M, NVDA 2.6B. Thinly traded, as intended.
- [x] **2. EDA + go/no-go**: `notebooks/02_lead_lag.ipynb`, helpers in `src/semiconductor_affairs/data.py`. Findings:
      - Linear lead-lag is **weak**: best intraday corr ≈ 0.03–0.05 vs noise band ±0.030 (~70 pairs tested); gone by lag 4.
      - **Unstable**: 2025 ≈ +0.08–0.11, 2024 small, 2026 *negative*. QQQ predicts as well as the customers, so this is
        market momentum rather than supply-chain lag.
      - **Direction**: ICHR doesn't lag its customers (LRCX/AMAT/KLAC both ways ≈ equal). Only NVDA, TSM and QQQ lead
        one-way (e.g. NVDA 0.040 vs −0.016), so the hint flows from end demand, not customers. All near the noise band.
      - **Overnight reversal**: last hour vs next first bar ≈ −0.06…−0.13 (band ±0.073), mostly in the gap itself,
        **same sign every year** and also from liquid SOXX, so not just bid-ask bounce. Most consistent pattern so far.
      - **Big moves**: after a >2σ leader up-hour, ICHR's next hour averages +0.1…+0.3% (>3σ: up to +0.7%), but every
        row is within ~2 s.e. of zero (s.e. 0.2–0.65%) and medians are small. A hint worth testing, not a finding.
      - Facts check: ICHR market cap ≈ $1.3B, 8 analysts (LRCX/AMAT ≈ $270–280B, 30–36). Lam + Applied ≈ 73–79% of
        Ichor's sales (10-Ks FY2016–22: 79% in FY2022; ~73–76% for FY2024–25 per third-party summaries).
      - **Decision: go, with tempered expectations.**
- [x] **3. Baselines**: `features.py` (shared 3D windows: samples × 7 hours × features), `evaluate.py` (walk-forward
      by quarter from 2025, expanding train window, 1-bar embargo; metrics MAE/RMSE/skill vs zero/corr/direction),
      `baselines.py`, `notebooks/03_baselines.ipynb`. Results (test 2025-01 → 2026-11, 2,820 hours):
      - **Nothing beats zero overall**: ridge −0.09% skill, ridge+big −0.12%, direction ≈ 49.5%.
      - Persistence −370% skill: large predictions with no signal are very costly.
      - Ridge shrinks to ~0 itself (alpha 5.6k–31.6k, last fold 1M): prediction std 0.07% vs actual 1.3%.
      - Regime flip again: 2025 skill +0.4% (corr 0.07), 2026 −0.5%. Big-move features don't help.
      - **Bar for the LSTM: skill > 0 on the walk-forward, in both years.**
- [x] **4. LSTM v1**: `lstm.py` (nn.LSTM, hidden 16, ~1.8k weights, explicit loop, chronological 20% validation,
      early stopping, target scaled by train std), `notebooks/04_lstm.ipynb`. Windows 7 / 35 / 140 h × 3 seeds, scored on
      identical hours. Results:
      - **No window beats zero**: skill ≈ −0.3% for all (2025 +0.23%, 2026 −0.7%); ridge on the same hours −0.19%.
      - **Window length doesn't matter**: the forget gate stays ≈ 0.5 (its initial value), so memory halves every hour
        and the 140 h model only uses the last ~3–7 h (wiping older hours doesn't change predictions).
      - Seed spread (−0.24…−0.51%) ≫ window differences (~0.02%), so always use several seeds.
      - Overfitting visible: train loss falls, validation ≈ 0.99 × zero.
- [x] **Overnight gap** (pulled forward from step 7): `overnight.py` (daily features incl. earnings-tonight flags from
      Yahoo's calendar, `night_days`), earnings dates added to `ingest.py`, `backtest()` in `evaluate.py`,
      `notebooks/05_overnight_gap.ipynb`. Test 2025-01 → 2026-11, 472 nights. Results:
      - First models to beat zero, on **normal nights**: reversal +1.5%, LSTM +1.1%, ridge +0.6%. Earnings nights
        (±18%, ~40% of squared movement) wreck error metrics, so the flag is used as a skip-trading filter.
      - "Always long" (overnight drift) = +108% gross, +62% at 5 bp/side: the real bar.
      - LSTM (10-day daily sequence, hidden 8, **20-seed average**): +186% gross, +140% at 5 bp, beats 99.3% of random
        strategies. But mostly 2026 (2025 < always long), single seeds −26…+157% (median 74%), best of 6 models,
        idealised fills. **Promising, not proven** → confirm with live paper trading.
      - **Long or flat** (`backtest(..., long_only=True)`: buy at the close if the prediction > 0, else stay in cash;
        never short): LSTM +147% gross, +118% at 5 bp/side, same Sharpe as long/short, 290 vs 464 trades. The short
        nights only paid off in the last ~6 weeks of the test. Preferred variant for any real use: shorting means
        borrow fees, possible recall of the shares, and unlimited loss on a takeover-style jump. Holding a permanent
        base position (e.g. 10 units, sell 1 on "down" nights) avoids shorting but adds full buy-and-hold ICHR risk,
        including earnings nights, so rejected.
      - **Minimum-gain threshold** (only buy if prediction > threshold, chosen each quarter from earlier quarters only):
        worse at every level (no threshold +135%, 0.1% +94%, adaptive +107% at 2 bp/side). The LSTM's prediction
        *size* carries little information, mainly its sign. Keep threshold 0.
      - Wallet simulation (whole shares, compounding, $323 start = 10 shares): LSTM long or flat ~$1,200, always long
        ~$790, buy and hold ~$355; buy and hold dipped to $140 on the way.
- [ ] **5. LSTM internals** (deprioritised: not interested in a hand-written cell for now): write an LSTM cell from scratch, check it matches `nn.LSTM` numerically, and look at gate
      activations on real data. Try a forget-gate bias init of ≥1 (longer starting memory).
- [ ] **6. Evaluate**: LSTM vs baselines on metrics from §5.7, including the toy backtest with costs.
- [ ] **7. Iterate**: excess-return target, time-of-day / overnight features, Taiwan-close TSMC (`2330.TW`) as an
      overnight feature, European variant, daily-frequency comparison.
- [ ] **8. MLOps**: scheduled data refresh, model versioning, retraining.
- [ ] **9. Deploy**: FastAPI + a simple page showing the latest prediction and a **live track record** (predicted vs
      realised). That's the honest way to show a model like this. Track three paper strategies side by side:
      LSTM long/short, LSTM **long or flat**, and always long. Decide on real money only after months of live results.

## 8. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-19 | Target = ICHR, US-listed inputs only for v1 | Clear supply-chain link, small cap, single trading clock |
| 2026-11-19 | Hourly data via yfinance | Only free hourly source; 730-day limit accepted, own history built up over time |
| 2026-11-19 | Signal check + baselines before any LSTM | Otherwise LSTM results can't be interpreted |
| 2026-11-19 | Ingest as a module, inspection in notebooks | Ingest gets scheduled and reused by the service; exploration doesn't |
| 2026-11-19 | Go after step 2, overnight bar out of v1 target | Weak/unstable linear signal, but nonlinear big-move hints; overnight bar is a different process |
| 2026-11-20 | Prefer long or flat over long/short for any real use | Keeps most of the backtest gain without borrow fees or unlimited loss; a permanent base position would add buy-and-hold risk instead |

## 9. Open questions

- Regression (return) or classification (up/down/flat)? Start with regression, report sign accuracy too.
- Predict one target or several small caps at once (multi-output)? More targets means more training signal per model.
- How much of the 730 days to keep for the test set? Proposal: last ~3 months.
