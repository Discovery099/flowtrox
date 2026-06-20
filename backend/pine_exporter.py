"""Faithful Pine Script v6 exporter for FLOWTOX_REGIME_01.

Unlike the static FLOWTOX_REGIME_01.pine (which approximates the regime model
with a logistic proxy), this module BAKES the actual fitted model parameters
into a self-contained Pine strategy so the regime posteriors are computed with
the *exact* same math as the Python engine:

  * HAR-RV forecast      -> fitted OLS coefficients (intercept, beta_d/w/m)
  * Volatility regime     -> fitted train-set p33/p67 thresholds
  * Feature scaler        -> fitted StandardScaler mean_ / scale_
  * Regime emissions      -> multivariate-normal log-likelihood using baked
                             inverse-covariances + log-normalisation constants
  * GMM posterior         -> log(weight_k) + emission_k, softmax over 3 states
  * HMM posterior         -> forward FILTERING recursion using baked startprob
                             + transition matrix (no look-ahead, matches the
                             Python _filtered_posteriors exactly)

Honest residual differences vs. the Python backtest:
  - TradingView's data feed (continuous/front-month contract, session handling,
    volume) differs from the user's RTH CSV, so trade-for-trade results will not
    match. The STRATEGY LOGIC and FITTED MODEL are faithful.
  - Rolling-window warmup uses min_periods=1 in Python vs. na-until-full in
    Pine; the first ~20-22 days differ slightly.
"""

from datetime import datetime, timezone

import numpy as np

import strategy_service as svc


def _f(x) -> str:
    """Format a float for Pine source with high precision."""
    xf = float(x)
    if xf == 0.0:
        return "0.0"
    return f"{xf:.12g}"


def _safe_log(x) -> float:
    return float(np.log(float(x) + 1e-300))


def _emission_fn(name: str, mu, invcov, logconst) -> str:
    """Generate a Pine function computing the MVN log-likelihood for one state.

    logpdf(z) = logconst - 0.5 * (z - mu)^T invcov (z - mu)
    The quadratic form is fully unrolled (5x5 = 25 terms) to avoid any Pine
    array-size limits and keep the script self-contained.
    """
    lines = [f"{name}(z0, z1, z2, z3, z4) =>"]
    for i in range(5):
        lines.append(f"    d{i} = z{i} - ({_f(mu[i])})")
    terms = []
    for i in range(5):
        for j in range(5):
            coef = invcov[i][j]
            if abs(coef) < 1e-18:
                continue
            terms.append(f"({_f(coef)})*d{i}*d{j}")
    quad = " + ".join(terms) if terms else "0.0"
    lines.append(f"    quad = {quad}")
    lines.append(f"    ({_f(logconst)}) - 0.5*quad")
    return "\n".join(lines)


def _state_params(cov, mu, jitter: float = 0.0):
    """Return (inv_cov, log_norm_const) for a single Gaussian state.

    log_norm_const = -0.5 * (d*log(2*pi) + logdet(cov))

    ``jitter`` mirrors the exact regularization the engine applies for that
    model path so the Pine emissions match Python to machine precision:
      * GMM: jitter=0.0   (sklearn predict_proba uses covariances_ as-is)
      * HMM: jitter=1e-6  (engine _emission_logprob adds eye*1e-6)
    """
    d = cov.shape[0]
    cov_f = np.asarray(cov, dtype=np.float64) + np.eye(d) * jitter
    try:
        inv = np.linalg.inv(cov_f)
        sign, logdet = np.linalg.slogdet(cov_f)
        if sign <= 0 or not np.isfinite(logdet):
            raise np.linalg.LinAlgError("non-positive-definite covariance")
    except np.linalg.LinAlgError:
        cov_f = cov_f + np.eye(d) * 1e-9
        inv = np.linalg.inv(cov_f)
        sign, logdet = np.linalg.slogdet(cov_f)
    logconst = -0.5 * (d * np.log(2.0 * np.pi) + logdet)
    return inv, float(logconst)


def generate_pine(symbol: str, regime_model: str = "gmm") -> str:
    """Fit/load the frozen model for (symbol, regime_model) and render Pine v6."""
    symbol = symbol.upper()
    cache = svc.ensure_models(symbol, regime_model=regime_model)
    fitted = cache["fitted"]
    inst = cache["inst"]
    mi = cache["model_info"]

    model = fitted["hmm_model"]
    scaler = fitted["scaler"]
    label_map = list(fitted["label_map"])          # [normal, continuation, reversal]
    har = fitted["har_params"]
    vt = fitted["vol_thresholds"]

    means = np.asarray(model.means_)               # (K, 5) standardized space
    if regime_model == "hmm":
        covars = np.asarray(model.covars_)         # (K, 5, 5)
        startprob = np.asarray(model.startprob_)   # (K,)
        transmat = np.asarray(model.transmat_)     # (K, K)
    else:
        covars = np.asarray(model.covariances_)    # (K, 5, 5)
        weights = np.asarray(model.weights_)       # (K,)

    # Reorder everything into semantic-state order: 0=Normal, 1=Cont, 2=Rev.
    sem = label_map  # sem[s] = component index for semantic state s
    mu_s = [means[sem[s]] for s in range(3)]
    # Match the exact regularization of each engine path (GMM none, HMM 1e-6).
    jitter = 1e-6 if regime_model == "hmm" else 0.0
    inv_s, logc_s = [], []
    for s in range(3):
        inv, lc = _state_params(covars[sem[s]], mu_s[s], jitter=jitter)
        inv_s.append(inv)
        logc_s.append(lc)

    scaler_mean = np.asarray(scaler.mean_)
    scaler_scale = np.asarray(scaler.scale_)

    point_value = float(inst["contract_specs"]["point_value"])
    commission_rt = float(inst["commission_rt"])
    tick_size = float(inst["contract_specs"]["tick_size"])

    # ----- Emission functions -----
    emis_fns = "\n\n".join(
        _emission_fn(f"f_emis_s{s}", mu_s[s], inv_s[s], logc_s[s]) for s in range(3)
    )

    # ----- Regime-posterior block (GMM softmax vs HMM forward filter) -----
    if regime_model == "gmm":
        logw = [_safe_log(weights[sem[s]]) for s in range(3)]
        posterior_block = f"""
// ---- GMM posterior: log(weight_k) + emission_k, softmax over 3 states ----
r0 = ({_f(logw[0])}) + e0
r1 = ({_f(logw[1])}) + e1
r2 = ({_f(logw[2])}) + e2
mlse = math.max(r0, math.max(r1, r2))
er0 = math.exp(r0 - mlse)
er1 = math.exp(r1 - mlse)
er2 = math.exp(r2 - mlse)
sden = er0 + er1 + er2
pNormal = validBar ? er0 / sden : na
pCont   = validBar ? er1 / sden : na
pRev    = validBar ? er2 / sden : na
"""
    else:
        ls = [_safe_log(startprob[sem[s]]) for s in range(3)]
        # logA[i][j] in semantic order: A_sem[i][j] = transmat[sem[i], sem[j]]
        logA = [[_safe_log(transmat[sem[i], sem[j]]) for j in range(3)] for i in range(3)]
        posterior_block = f"""
// ---- HMM posterior: forward FILTERING recursion (no look-ahead) ----
// Persisted log-alpha across bars; matches Python _filtered_posteriors().
var float la0 = na
var float la1 = na
var float la2 = na

f_lse3(a, b, c) =>
    m = math.max(a, math.max(b, c))
    m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))

float pNormal = na
float pCont = na
float pRev = na
if validBar
    if na(la0)
        // t = 0: initialise with start probabilities.
        la0 := ({_f(ls[0])}) + e0
        la1 := ({_f(ls[1])}) + e1
        la2 := ({_f(ls[2])}) + e2
    else
        // predict (sum over previous states) then update with emission.
        pred0 = f_lse3(la0 + ({_f(logA[0][0])}), la1 + ({_f(logA[1][0])}), la2 + ({_f(logA[2][0])}))
        pred1 = f_lse3(la0 + ({_f(logA[0][1])}), la1 + ({_f(logA[1][1])}), la2 + ({_f(logA[2][1])}))
        pred2 = f_lse3(la0 + ({_f(logA[0][2])}), la1 + ({_f(logA[1][2])}), la2 + ({_f(logA[2][2])}))
        la0 := pred0 + e0
        la1 := pred1 + e1
        la2 := pred2 + e2
    nrm = f_lse3(la0, la1, la2)
    la0 := la0 - nrm
    la1 := la1 - nrm
    la2 := la2 - nrm
    pNormal := math.exp(la0)
    pCont := math.exp(la1)
    pRev := math.exp(la2)
"""

    # ----- Header metadata -----
    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    train_start = mi.get("train_start", "?")[:10]
    train_end = mi.get("train_end", "?")[:10]

    header = f"""//@version=6
// =============================================================================
// FLOWTOX_REGIME_01 - FAITHFUL fitted-model export
// Instrument : {symbol}  ({inst['name']})
// Regime     : {regime_model.upper()}  (3-state, full covariance)
// Generated  : {gen_ts}
// Train span : {train_start} -> {train_end}   ({mi.get('train_rows', '?')} bars)
// label_map  : {label_map}   (component -> [Normal, Continuation, Reversal])
//
// This script BAKES the exact fitted parameters of the Python engine:
//   HAR-RV OLS coefficients, train-set vol-regime thresholds, StandardScaler
//   mean/scale, and the {regime_model.upper()} Gaussian means + inverse-covariances. The
//   regime posteriors are therefore computed with the SAME math as Python
//   ({'softmax over GMM responsibilities' if regime_model=='gmm' else 'HMM forward filtering, no look-ahead'}).
//
// HONEST CAVEAT: TradingView's data feed differs from the RTH CSV used to fit
//   the model (contract continuity, session, volume), so trade-for-trade P&L
//   will not match the Python backtest exactly. The logic + model are faithful.
//   For exact research results, use the Python engine; use this to validate the
//   strategy on TradingView and (later) drive alerts / paper trading.
// =============================================================================

strategy(
     title              = "FLOWTOX_REGIME_01 {symbol} {regime_model.upper()}",
     shorttitle         = "FLOWTOX_{symbol}",
     overlay            = false,
     pyramiding         = 0,
     initial_capital    = 100000,
     default_qty_type   = strategy.fixed,
     default_qty_value  = 1,
     commission_type    = strategy.commission.cash_per_order,
     commission_value   = {_f(commission_rt / 2.0)},   // per order; round-turn = {_f(commission_rt)}
     slippage           = 1,
     calc_on_every_tick = false,
     process_orders_on_close = true)
"""

    inputs = f"""
// ---------------------------------------------------------------------------
// INPUTS (the 3 optimizable params + sizing). Model params are baked below.
// ---------------------------------------------------------------------------
grpSig = "Signal thresholds (optimizable)"
tau1   = input.float(0.55, "tau1  toxic-continuation thresh", minval=0.0, maxval=1.0, step=0.05, group=grpSig)
tau2   = input.float(0.55, "tau2  toxic-reversal thresh",     minval=0.0, maxval=1.0, step=0.05, group=grpSig)
N      = input.int(15,     "N     max hold bars",             minval=1,   maxval=200, group=grpSig)
useRegimeExit = input.bool(true, "Enable regime early-exit (p < tau/2)", group=grpSig)

grpSize = "Position sizing (inverse-vol target)"
acctVal    = input.float(100000, "Account value ($)", group=grpSize)
sigmaTgt   = input.float(0.15,   "Target volatility (annual frac)", step=0.01, group=grpSize)
pointValue = input.float({_f(point_value)}, "Point value ($/pt) for {symbol}", group=grpSize)
maxSize    = input.int(10,       "Max contracts", minval=1, group=grpSize)

grpWin = "Backtest window"
useTestWindow = input.bool(false, "Trade only after test-start date", group=grpWin)
testStart     = input.time(timestamp("{mi.get('test_start','2024-04-01')[:10]} 00:00"), "Test start", group=grpWin)

lookback = 20  // HMM_LOOKBACK
"""

    features = f"""
// ---------------------------------------------------------------------------
// 2.1-2.4  BVC, tick rule, toxicity, Parkinson per-bar variance
// ---------------------------------------------------------------------------
rng      = high - low
bvc      = rng > 0 ? volume * (2 * close - low - high) / rng : 0.0
bvcSign  = bvc > 0 ? 1.0 : bvc < 0 ? -1.0 : 0.0
bvcDir   = bvcSign

tickDir  = close > close[1] ? 1.0 : close < close[1] ? -1.0 : 1.0
toxicity = bvcSign != 0 ? bvcSign * tickDir : 0.0   // -1 toxic, +1 informed, 0

parkinson = rng > 0 ? math.pow(math.log(high / low), 2) / (4 * math.log(2)) : 0.0

// ---------------------------------------------------------------------------
// 2.5  Realized variance grouped by actual session/day
// ---------------------------------------------------------------------------
var float[] dailyRVs = array.new_float()
var float   rvAccum  = 0.0
newDay = ta.change(time("D")) != 0
if newDay and bar_index > 0
    array.push(dailyRVs, rvAccum)
    rvAccum := 0.0
rvAccum += parkinson

f_lastmean(arr, n) =>
    sz  = array.size(arr)
    cnt = math.min(n, sz)
    s   = 0.0
    if cnt > 0
        for i = 0 to cnt - 1
            s += array.get(arr, sz - 1 - i)
    cnt > 0 ? s / cnt : na

rv1d = array.size(dailyRVs) > 0 ? array.get(dailyRVs, array.size(dailyRVs) - 1) : na
rv1w = f_lastmean(dailyRVs, 5)
rv1m = f_lastmean(dailyRVs, 22)

// ---------------------------------------------------------------------------
// 2.6  HAR volatility forecast (BAKED fitted coefficients)
// ---------------------------------------------------------------------------
harC  = {_f(har['intercept'])}
harBd = {_f(har['beta_d'])}
harBw = {_f(har['beta_w'])}
harBm = {_f(har['beta_m'])}
harForecast = na(rv1d) ? na : math.max(harC + harBd * rv1d + harBw * rv1w + harBm * rv1m, 1e-12)

// ---------------------------------------------------------------------------
// 2.7  Volatility regime (BAKED train-set p33/p67 thresholds)
// ---------------------------------------------------------------------------
p33 = {_f(vt['p33'])}
p67 = {_f(vt['p67'])}
volRegime = na(harForecast) ? 1 : harForecast <= p33 ? 0 : harForecast > p67 ? 2 : 1

// ---------------------------------------------------------------------------
// 2.8  Observable features (rolling over lookback=20)
// ---------------------------------------------------------------------------
isToxic = toxicity < 0 ? 1.0 : 0.0
f1 = ta.sma(isToxic, lookback)                          // toxicity rate
f2 = ta.sma(bvcSign, lookback)                          // mean BVC sign
f3 = math.log(1 + ta.sma(parkinson, lookback))          // log rolling Parkinson var
f4 = close - close[5]                                    // 5-bar momentum
f5 = volume / math.max(ta.sma(volume, lookback), 1)     // volume intensity

// ---------------------------------------------------------------------------
// 2.9  Standardize with BAKED StandardScaler (train mean/scale)
// ---------------------------------------------------------------------------
z0 = (f1 - ({_f(scaler_mean[0])})) / ({_f(scaler_scale[0])})
z1 = (f2 - ({_f(scaler_mean[1])})) / ({_f(scaler_scale[1])})
z2 = (f3 - ({_f(scaler_mean[2])})) / ({_f(scaler_scale[2])})
z3 = (f4 - ({_f(scaler_mean[3])})) / ({_f(scaler_scale[3])})
z4 = (f5 - ({_f(scaler_mean[4])})) / ({_f(scaler_scale[4])})

warmup   = math.max(lookback, 25)
validBar = not na(harForecast) and not na(f1) and not na(f4) and bar_index > warmup
"""

    emissions = f"""
// ---------------------------------------------------------------------------
// Gaussian emission log-likelihoods per semantic state (BAKED inv-cov)
//   state 0 = Normal, 1 = Toxic-Continuation, 2 = Toxic-Reversal
// ---------------------------------------------------------------------------
{emis_fns}

e0 = validBar ? f_emis_s0(z0, z1, z2, z3, z4) : na
e1 = validBar ? f_emis_s1(z0, z1, z2, z3, z4) : na
e2 = validBar ? f_emis_s2(z0, z1, z2, z3, z4) : na
{posterior_block}"""

    sizing_logic = """
// ---------------------------------------------------------------------------
// 3.6  Inverse-volatility position sizing (1..maxSize)
// ---------------------------------------------------------------------------
fVol = na(harForecast) ? na : math.sqrt(math.max(harForecast, 1e-12))
contractVolUsd = na(fVol) ? na : fVol * close * pointValue
sizeRaw = na(contractVolUsd) or contractVolUsd <= 0 ? 1.0 : (acctVal * sigmaTgt) / contractVolUsd
qty = math.max(1, math.min(maxSize, math.round(sizeRaw)))

// ---------------------------------------------------------------------------
// 3.7  Entry / exit logic (faithful to spec)
// ---------------------------------------------------------------------------
inWindow = not useTestWindow or time >= testStart
canTrade = validBar and not na(pCont) and inWindow and barstate.isconfirmed

longCond  = canTrade and pCont > tau1 and bvcDir > 0 and volRegime != 2
shortCond = canTrade and pRev  > tau2 and bvcDir < 0 and volRegime != 2

var int barsInTrade = 0
flat = strategy.position_size == 0
isLong  = strategy.position_size > 0
isShort = strategy.position_size < 0

if not flat
    barsInTrade += 1

if flat and longCond
    strategy.entry("L", strategy.long, qty = qty)
    barsInTrade := 0
else if flat and shortCond
    strategy.entry("S", strategy.short, qty = qty)
    barsInTrade := 0

if isLong
    timeExit  = barsInTrade >= N
    regimeOut = useRegimeExit and pCont < tau1 / 2.0
    if timeExit or regimeOut
        strategy.close("L", comment = timeExit ? "time" : "regime")
if isShort
    timeExit  = barsInTrade >= N
    regimeOut = useRegimeExit and pRev < tau2 / 2.0
    if timeExit or regimeOut
        strategy.close("S", comment = timeExit ? "time" : "regime")

// ---------------------------------------------------------------------------
// PLOTS - regime posteriors + vol regime + entry markers
// ---------------------------------------------------------------------------
plot(pNormal, "P(normal)",     color = color.new(color.gray, 0))
plot(pCont,   "P(toxic-cont)", color = color.new(color.lime, 0))
plot(pRev,    "P(toxic-rev)",  color = color.new(color.orange, 0))
hline(tau1, "tau1", color = color.new(color.lime, 60), linestyle = hline.style_dashed)
hline(tau2, "tau2", color = color.new(color.orange, 60), linestyle = hline.style_dashed)
bgcolor(volRegime == 2 ? color.new(color.red, 90) : na, title = "High-vol regime (no-trade)")
plotchar(flat and longCond,  "long sig",  "▲", location.bottom, color.lime,   size = size.tiny)
plotchar(flat and shortCond, "short sig", "▼", location.top,    color.orange, size = size.tiny)
"""

    return header + inputs + features + emissions + sizing_logic


def filename_for(symbol: str, regime_model: str) -> str:
    return f"FLOWTOX_REGIME_01_{symbol.upper()}_{regime_model.lower()}.pine"
