# =========================================================
# Three-model training + full SHAP beeswarm (optional ordering) + RF scenario simulation (ΔGM% + 95% CI)
# + Three figure sets (pathway/total exposure/extended multi-lever) — Nature-style
# =========================================================
import os, warnings, inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import matplotlib as mpl
mpl.use("Agg")            # Must be before importing pyplot (headless backends)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ===== Nature-style figure settings (global) =====
def use_nature_style(font_family: str = "Arial", base_size: float = 8.0):
    """
    Nature-like minimal style:
    - Sans-serif font (Arial/Helvetica), 8 pt base.
    - No grid. Thin axes, ticks inward on all sides.
    - Compact legend. 600 dpi export suggested.
    """
    mpl.rcParams.update({
        # font
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "Helvetica", "DejaVu Sans"],
        "font.size": base_size,
        "mathtext.default": "regular",

        # axes & spines
        "axes.linewidth": 0.8,
        "axes.labelsize": base_size,
        "axes.labelpad": 2.5,
        "axes.titlepad": 2.0,

        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        # legend
        "legend.frameon": False,
        "legend.fontsize": base_size,

        # lines
        "lines.linewidth": 1.4,
        "lines.markersize": 4,

        # grid
        "axes.grid": False,

        # save
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
    })

def style_axes(ax):
    """Apply Nature-like spine & ticks; keep all spines; ticks on both sides."""
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
    ax.tick_params(which="both", top=True, right=True)
    ax.minorticks_on()

# Activate style globally
use_nature_style()

# ---------------- Basic configuration ----------------
FILE = r'H:/shuikoushanData/Human/ML2025.xlsx'
RANDOM_STATE = 1
REVERSE_ORDER = True
np.random.seed(RANDOM_STATE)

# ---------------- Data loading & preprocessing ----------------
if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

df = pd.read_excel(FILE).copy()

# Binning
if 'Age' in df.columns:
    df['Age'] = pd.cut(df['Age'], [0,60,65,70,75,100], labels=[0,1,2,3,4], right=False).astype(float)
if 'Edu' in df.columns:
    df['Edu'] = pd.cut(df['Edu'], [0,6,9,12,100], labels=[0,1,2,3], right=False).astype(float)

# log1p (consistent with prior pipeline)
for col in ['UREA/CREA','Distance','Urine_Cd_CREA']:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
feature_names = ['Age', 'Gender', 'Weight','BMI',
                 'Distance','Smoke', 'UREA/CREA',
                 'Soil_ingestion','Water_ingestion','Water_dermal','Diet_other',
                 'Diet_rice','Diet_solanaceous','Diet_root_tuber','Diet_legumes'
                 ]
X=df[feature_names]
y=df.Urine_Cd_CREA

vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
# Show VIF results
index = X.columns.tolist()
vif_df = pd.DataFrame(vif, index = index, columns = ['vif']).sort_values(by = 'vif', ascending=False)
print(vif_df)

# Construct fine-grained + aggregate pathways
diet_cols    = ['Diet_rice','Diet_solanaceous','Diet_legumes','Diet_root_tuber','Diet_other']
env_ing_cols = ['Soil_ingestion','Water_ingestion','Water_dermal']
missing = [c for c in diet_cols+env_ing_cols if c not in df.columns]
if missing:
    raise KeyError(f'Missing columns: {missing}')

df[diet_cols]    = df[diet_cols].fillna(0.0)
df[env_ing_cols] = df[env_ing_cols].fillna(0.0)
df['Diet_total']     = df[diet_cols].sum(axis=1)
df['Env_total_calc'] = df[env_ing_cols].sum(axis=1)

SELECT_FEATURES = [
    'Age','Gender','Smoke',
    'Soil_ingestion','Water_ingestion','Water_dermal',
    'Diet_rice','Diet_solanaceous','Diet_legumes','Diet_root_tuber',
    'Diet_total','Env_total_calc'
]



nunq = df[SELECT_FEATURES].nunique()
const_cols = nunq[nunq <= 1].index.tolist()
if const_cols:
    print("⚠️ Constant features removed:", const_cols)
    SELECT_FEATURES = [c for c in SELECT_FEATURES if c not in const_cols]

X_all  = df[SELECT_FEATURES].copy()
y_log  = df['Urine_Cd_CREA'].astype(float).values
y_raw  = np.expm1(y_log)


def stratify_bins(y, n_bins=8):
    y = pd.Series(np.asarray(y).ravel())
    if y.nunique(dropna=True) <= 1:
        return np.repeat('bin0', len(y))
    q = pd.qcut(y, q=np.linspace(0, 1, n_bins + 1), duplicates='drop')
    return q.astype(str).to_numpy()

bins_all = stratify_bins(y_raw, n_bins=8)
X_trv, X_test, y_trv_log, y_test_log, y_trv_raw, y_test_raw, bins_trv, _ = train_test_split(
    X_all, y_log, y_raw, bins_all, test_size=0.20, random_state=RANDOM_STATE, stratify=bins_all
)
bins_trv2 = stratify_bins(y_trv_raw, n_bins=8)
X_train, X_val, y_train_log, y_val_log, y_train_raw, y_val_raw = train_test_split(
    X_trv, y_trv_log, y_trv_raw, test_size=0.20, random_state=RANDOM_STATE, stratify=bins_trv2
)


def _has_kwarg(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False

def rmse_compat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def metrics(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': rmse_compat(y_true, y_pred)}

def safe_xgb_fit(model, X_tr, y_tr, X_va=None, y_va=None, rounds=200):
    fit_kwargs = {}
    if X_va is not None and y_va is not None and _has_kwarg(model.fit, 'eval_set'):
        fit_kwargs['eval_set'] = [(X_va, y_va)]
    if _has_kwarg(model.fit, 'early_stopping_rounds'):
        fit_kwargs['early_stopping_rounds'] = rounds
    elif _has_kwarg(model.fit, 'callbacks'):
        try:
            from xgboost.callback import EarlyStopping
            fit_kwargs['callbacks'] = [EarlyStopping(rounds=rounds, save_best=True, maximize=False)]
        except Exception:
            pass
    if _has_kwarg(model.fit, 'verbose'):
        fit_kwargs['verbose'] = False
    return model.fit(X_tr, y_tr, **fit_kwargs)

def xgb_predict_best(model, X):
    best_it = getattr(model, 'best_iteration', None)
    if best_it is not None:
        try:
            return model.predict(X, iteration_range=(0, best_it + 1))
        except TypeError:
            pass
    best_ntree_limit = getattr(model, 'best_ntree_limit', None)
    if best_ntree_limit:
        try:
            return model.predict(X, ntree_limit=best_ntree_limit)
        except TypeError:
            pass
    return model.predict(X)

def lgbm_predict_best(model, X):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None:
        try:
            return model.predict(X, num_iteration=best_iter)
        except TypeError:
            pass
    return model.predict(X)

def gm(arr, eps=1e-9):
    arr = np.asarray(arr, float)
    arr = np.clip(arr, eps, None)
    return float(np.exp(np.mean(np.log(arr))))

# ---------------- Train three models ----------------
models = {}

models['XGB'] = XGBRegressor(
    random_state=RANDOM_STATE, eval_metric='rmse',
    n_estimators=3000, learning_rate=0.03,
    max_depth=5, min_child_weight=15,
    subsample=0.7, colsample_bytree=0.7, colsample_bynode=0.8,
    reg_alpha=0.2, reg_lambda=2.0
)
safe_xgb_fit(models['XGB'], X_train, y_train_log, X_val, y_val_log, rounds=200)

models['RF'] = RandomForestRegressor(
    n_estimators=800, max_depth=12, min_samples_split=4, min_samples_leaf=3,
    max_features='sqrt', bootstrap=True, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1
)
models['RF'].fit(X_train, y_train_log)

models['LGBM'] = LGBMRegressor(
    n_estimators=5000, learning_rate=0.02,
    num_leaves=31, max_depth=-1,
    min_data_in_leaf=25, min_split_gain=0.0,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.0, lambda_l2=1.0,
    feature_pre_filter=False, random_state=RANDOM_STATE, verbose=-1
)
lgb_fit_kwargs = {}
if _has_kwarg(models['LGBM'].fit, 'eval_set'):
    lgb_fit_kwargs['eval_set'] = [(X_val, y_val_log)]
if _has_kwarg(models['LGBM'].fit, 'early_stopping_rounds'):
    lgb_fit_kwargs['early_stopping_rounds'] = 200
models['LGBM'].fit(X_train, y_train_log, **lgb_fit_kwargs)

# ---------------- Unified prediction (log1p scale) ----------------
def predict_log(model_name, mdl, X):
    if model_name == 'XGB':
        return xgb_predict_best(mdl, X)
    elif model_name == 'LGBM':
        return lgbm_predict_best(mdl, X)
    else:
        return mdl.predict(X)

# ---------------- Metrics on splits ----------------
def eval_on_splits(name, mdl):
    rows_log, rows_raw = [], []
    for split, (X_, y_log_true, y_raw_true) in {
        'Train': (X_train, y_train_log, y_train_raw),
        'Val'  : (X_val,   y_val_log,   y_val_raw),
        'Test' : (X_test,  y_test_log,  y_test_raw),
    }.items():
        y_log_pred = predict_log(name, mdl, X_)
        m_log = metrics(y_log_true, y_log_pred); m_log.update({'Model':name,'Split':split})
        rows_log.append(m_log)

        y_raw_pred = np.expm1(y_log_pred)
        m_raw = metrics(y_raw_true, y_raw_pred); m_raw.update({'Model':name,'Split':split})
        rows_raw.append(m_raw)
    return pd.DataFrame(rows_log), pd.DataFrame(rows_raw)

all_log, all_raw, test_preds = [], [], {}
for name, mdl in models.items():
    df_log, df_raw = eval_on_splits(name, mdl)
    all_log.append(df_log); all_raw.append(df_raw)
    test_preds[name] = np.expm1(predict_log(name, mdl, X_test))

log_df = pd.concat(all_log, ignore_index=True)[['Model','Split','R2','MAE','RMSE']]
raw_df = pd.concat(all_raw, ignore_index=True)[['Model','Split','R2','MAE','RMSE']]

print("\n=== Metrics (training scale: log1p) ===")
print(log_df.pivot(index='Model', columns='Split', values=['R2','MAE','RMSE']))
print("\n=== Metrics (original units µg/g CREA) ===")
print(raw_df.pivot(index='Model', columns='Split', values=['R2','MAE','RMSE']))
if hasattr(models['RF'], 'oob_score_'):
    print(f"\nRF OOB R² (log1p 尺度): {models['RF'].oob_score_:.4f}")

# ---------------- SHAP beeswarm (full data, triptych; Nature-style) ----------------
try:
    import shap
    shap_vals = {}
    shap_abs_mean = {}
    for name, mdl in models.items():
        explainer = shap.TreeExplainer(mdl)
        vals = explainer.shap_values(X_all)             # (n, p)
        shap_vals[name] = vals
        shap_abs_mean[name] = np.abs(vals).mean(axis=0) # (p,)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharex=False, sharey=False)
    cmap = mpl.cm.get_cmap('coolwarm')


    Xv = X_all.values
    Z = (Xv - Xv.mean(axis=0)) / (Xv.std(axis=0) + 1e-9)

    for ax, (name, vals) in zip(axes, shap_vals.items()):
        style_axes(ax)
        abs_mean = shap_abs_mean[name]
        order = np.argsort(abs_mean) if REVERSE_ORDER else np.argsort(abs_mean)[::-1]
        feat_order = [X_all.columns[i] for i in order]
        y_pos = np.arange(len(order))


        ax.barh(y_pos, abs_mean[order], color="#d6e4f0", edgecolor="none", height=0.7)


        for row_i, j in enumerate(order):
            x_vals = vals[:, j]
            y_jit  = row_i + np.random.normal(0, 0.07, size=len(x_vals))
            ax.scatter(x_vals, y_jit, c=Z[:, j], cmap=cmap, s=6, alpha=0.65, linewidths=0)

        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_order)
        ax.set_xlabel(f"{name}-SHAP value ")



    cax = fig.add_axes([0.92, 0.18, 0.012, 0.64])
    norm = mpl.colors.Normalize(vmin=-2, vmax=2)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=[-2,-1,0,1,2])
    cb.set_label("Feature value (low → high)")

    plt.tight_layout(rect=[0.02, 0.02, 0.90, 0.98])
    plt.savefig("SHAP_beeswarm_all_3models_reversed.png")
    plt.savefig("SHAP_beeswarm_all_3models_reversed.pdf")
    plt.close(fig)
except Exception as e:
    print("SHAP computation/plotting failed: ", e)

# ---------------- RF scenario simulation (GM% change; negative = reduction) ----------------
def recompute_aggregates(df_):
    df_[diet_cols]    = df_[diet_cols].fillna(0.0)
    df_[env_ing_cols] = df_[env_ing_cols].fillna(0.0)
    df_['Diet_total']     = df_[diet_cols].sum(axis=1)
    df_['Env_total_calc'] = df_[env_ing_cols].sum(axis=1)
    return df_

def rf_predict_raw(df_feat):
    ylog = models['RF'].predict(df_feat[SELECT_FEATURES])
    return np.expm1(ylog)

def apply_rice_cap(df_, cap):
    """Cap rice Cd at `cap` (mg/kg). If `rice_Cd_mgkg` is missing, skip."""
    if 'rice_Cd_mgkg' not in df_.columns:
        print(f"⚠️ 缺少 rice_Cd_mgkg，跳过 cap={cap} mg/kg 情景")
        return df_
    df2 = df_.copy()
    cd = df2['rice_Cd_mgkg'].replace(0, np.nan)
    factor = np.minimum(1.0, cap / cd).fillna(1.0)
    df2['Diet_rice'] = df2['Diet_rice'] * factor
    return recompute_aggregates(df2)

def apply_scale(df_, changes: dict):
    """Scale specified columns by a factor; silently ignore missing columns."""
    df2 = df_.copy()
    for k, v in changes.items():
        if k in df2.columns:
            df2[k] = df2[k] * float(v)
    return recompute_aggregates(df2)

def apply_override(df_, overrides: dict):
    """Override specified columns with given values; silently ignore missing columns."""
    df2 = df_.copy()
    for k, v in overrides.items():
        if k in df2.columns:
            df2[k] = float(v)
    return recompute_aggregates(df2)

# Scenario set
scenarios = [
    {'code':'B1',   'desc':'Rice Cd ≤ 0.4 mg/kg (Codex)',         'fn': lambda d: apply_rice_cap(d, 0.4)},
    {'code':'B2',   'desc':'Rice Cd ≤ 0.2 mg/kg (China)',         'fn': lambda d: apply_rice_cap(d, 0.2)},
    {'code':'B3',  'desc':'Rice Cd = 0 (clean rice; Diet_rice→0)','fn': lambda d: apply_override(d, {'Diet_rice':0.0})},
    {'code':'C1',  'desc':'Diet diversification (rice -20%)',    'fn': lambda d: apply_scale(d, {'Diet_rice':0.8})},
    {'code':'C2',  'desc':'Diet diversification (rice -40%)',    'fn': lambda d: apply_scale(d, {'Diet_rice':0.6})},
    {'code':'D1',   'desc':'PPE: Soil ingestion -50%',            'fn': lambda d: apply_scale(d, {'Soil_ingestion':0.5})},
    {'code':'D2',   'desc':'Liming: Soil ingestion -30%',         'fn': lambda d: apply_scale(d, {'Soil_ingestion':0.7})},
    # {'code':'F',   'desc':'Drinking water ingestion = 0',        'fn': lambda d: apply_override(d, {'Water_ingestion':0.0})},
    {'code':'D3',   'desc':'Soil ingestion = 0',                  'fn': lambda d: apply_override(d, {'Soil_ingestion':0.0})},
]

# Quick check: GM (observed vs predicted)
obs_raw_full = np.expm1(df['Urine_Cd_CREA'].astype(float).values)
pred_raw_full = rf_predict_raw(df)
print(f"[Check] Observed GM (µg/g CREA): {gm(obs_raw_full):.4f}")
print(f"[Check] RF-pred GM (µg/g CREA):  {gm(pred_raw_full):.4f}")
print(f"[Check] Test Observed GM: {gm(y_test_raw):.4f}")
print(f"[Check] Test RF-pred GM:  {gm(test_preds['RF']):.4f}")

def eval_scenarios_rf(df_base, scenarios, n_boot=500):
    """
    返回每个情景的 ΔGM%(带95%CI)以及情景下的 GM(µg/g CREA 与其95%CI)。
    Baseline 使用观测值(反变换后的尿Cd)作为基线，并对 baseline 也做 bootstrap CI。
    情景模拟仍用 RF 预测来算 ΔGM 与情景 GM。
    """
    results = []
    n = len(df_base)


    obs_raw = np.expm1(df_base['Urine_Cd_CREA'].astype(float).values)
    base_gm = gm(obs_raw)
    base_boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.random.randint(0, n, n)
        base_boot[b] = gm(obs_raw[idx])
    base_lo, base_hi = np.percentile(base_boot, [2.5, 97.5])

    results.append({
        'Code': 'Baseline',
        'Scenario': 'Observed baseline',
        'ΔGM (%)': 0.0,
        'CI low (%)': 0.0,
        'CI high (%)': 0.0,
        'GM (µg/g CREA)': base_gm,
        'GM CI low': float(base_lo),
        'GM CI high': float(base_hi),
    })


    for sc in scenarios:
        df2 = sc['fn'](df_base)

        if df2 is df_base and sc['code'] in ('A','B'):
            continue

        preds = rf_predict_raw(df2)
        gm_abs = gm(preds)

        # bootstrap for absolute GM and delta
        gms = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = np.random.randint(0, n, n)
            gms[b] = gm(preds[idx])

        gm_lo, gm_hi = np.percentile(gms, [2.5, 97.5])
        delta = (gms - base_gm) / base_gm * 100.0
        mean_change = float(delta.mean())
        ci_low, ci_high = np.percentile(delta, [2.5, 97.5])

        results.append({
            'Code': sc['code'],
            'Scenario': sc['desc'],
            'ΔGM (%)': mean_change,
            'CI low (%)': float(ci_low),
            'CI high (%)': float(ci_high),
            'GM (µg/g CREA)': gm_abs,
            'GM CI low': float(gm_lo),
            'GM CI high': float(gm_hi),
        })


    order = ['A','B1','B2','B3','C1','C2','D1','D2','D3']
    df_res = pd.DataFrame(results)
    df_res['__ord'] = df_res['Code'].apply(lambda c: order.index(c) if c in order else 999)
    df_res = df_res.sort_values(['__ord','Code']).drop(columns='__ord').reset_index(drop=True)
    return df_res


sc_summary = eval_scenarios_rf(df, scenarios, n_boot=500)
print("\n=== RF scenario simulation (by geometric mean; negative = reduction) ===")
print(sc_summary[['Code','Scenario','ΔGM (%)','CI low (%)','CI high (%)','GM (µg/g CREA)']].to_string(
    index=False, float_format=lambda x: f"{x:,.2f}"
))

# -------- Scenario visualization (ΔGM%+CI and GM+CI; remove Code in y-labels) --------
try:
    import re
    if not sc_summary.empty:
        y_pos   = np.arange(len(sc_summary))

        codes      = sc_summary['Code'].astype(str).values
        raw_labels = sc_summary['Scenario'].astype(str).values


        y_labels = []
        for lbl, cc in zip(raw_labels, codes):

            pattern = rf'^\s*\[?{re.escape(cc)}\]?\s*[:\-|、．。·]*\s*'
            cleaned = re.sub(pattern, '', lbl).strip()
            y_labels.append(cleaned)


        dgm      = sc_summary['ΔGM (%)'].values
        err_low  = dgm - sc_summary['CI low (%)'].values
        err_high = sc_summary['CI high (%)'].values - dgm


        gm_vals  = sc_summary['GM (µg/g CREA)'].values
        gm_lo    = sc_summary['GM CI low'].values
        gm_hi    = sc_summary['GM CI high'].values
        gm_err_l = gm_vals - gm_lo
        gm_err_r = gm_hi - gm_vals

        thr_list  = [1.0, 3.07, 5.0]
        thr_label = {1.0: "EFSA", 3.07: "CN", 5.0: "WHO"}

        gm_max = float(np.nanmax(gm_hi))
        x_max  = max(gm_max * 1.15, 5.8)

        fig = plt.figure(figsize=(6.2, 3.2))
        gs  = fig.add_gridspec(1, 3, width_ratios=[0.22, 1.28, 1.10], wspace=0.08)

        axC = fig.add_subplot(gs[0,0])  # Codes
        axL = fig.add_subplot(gs[0,1])  # ΔGM%
        axR = fig.add_subplot(gs[0,2])  # GM
        style_axes(axL); style_axes(axR)


        axC.set_xlim(0, 1); axC.set_ylim(-0.5, len(sc_summary)-0.5)
        axC.set_xticks([]); axC.set_yticks([])
        for yy, cc in enumerate(codes):
            if cc != 'Baseline' and re.match(r'^[A-Z]+[0-9]+$', cc):
                axC.text(0.98, yy, cc, ha='right', va='center', fontsize=8)
        for side in ["top","right","bottom","left"]:
            axC.spines[side].set_visible(False)


        axL.barh(
            y_pos, dgm, xerr=[err_low, err_high],
            capsize=2.5, color="#7AA6D1", edgecolor="none", alpha=0.95
        )
        axL.axvline(0, color="k", lw=0.8)
        axL.set_yticks(y_pos); axL.set_yticklabels(y_labels)
        axL.set_xlabel("Change in urinary Cd (%)")


        axR.barh(
            y_pos, gm_vals, xerr=[gm_err_l, gm_err_r],
            capsize=2.5, color="#9CC3E6", edgecolor="none", alpha=0.95,
            ecolor="k", height=0.65
        )
        for thr in thr_list:
            axR.axvline(thr, color='k', linestyle='--', linewidth=0.8)
        y_top = len(sc_summary) - 0.24
        for thr, lab in thr_label.items():
            axR.text(thr, y_top, lab, ha='center', va='bottom', fontsize=7)

        for yy, gm_v in zip(y_pos, gm_vals):
            axR.text(2.0, yy, f"{gm_v:.2f}", va='center', ha='center', fontsize=7,
                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

        axR.set_xlabel("Urinary Cd GM (µg/g CREA)")
        axR.set_xlim(0, x_max)
        axR.set_yticks(y_pos); axR.set_yticklabels([])

        axL.set_ylim(-0.5, len(sc_summary)-0.2)
        axR.set_ylim(axL.get_ylim())

        plt.tight_layout()
        plt.savefig("RF_Scenario_DeltaGM_and_GM_split_CI_v4.png")
        plt.savefig("RF_Scenario_DeltaGM_and_GM_split_CI_v4.pdf")
        plt.close(fig)
    else:
        print("No scenarios to evaluate")
except Exception as e:
    print("Scenario plotting failed: ", e)




# =========================
# 8) Curves: each 10% reduction (GM+95% CI) + 1 µg/g threshold — Rice / Soil / Both
# =========================

# Custom parameters
N_BOOT = 500
CURVE_USE_ENSEMBLE = False

# Prediction (returns original units µg/g CREA)
def predict_vector(model, Xfeat: pd.DataFrame):
    if isinstance(model, XGBRegressor):
        y_log = xgb_predict_best(model, Xfeat)
    elif isinstance(model, LGBMRegressor):
        y_log = lgbm_predict_best(model, Xfeat)
    elif isinstance(model, RandomForestRegressor):
        y_log = model.predict(Xfeat)
    else:
        y_log = model.predict(Xfeat)
    return np.expm1(y_log)

# ==== helpers for sections 8–10 ====
# Choose model set for curves: ensemble or RF only
if CURVE_USE_ENSEMBLE:
    models_for_curve = [models['XGB'], models['RF'], models['LGBM']]
else:
    models_for_curve = [models['RF']]

def _ensemble_pred(df_):
    preds = [predict_vector(m, df_[SELECT_FEATURES]) for m in models_for_curve]
    return np.mean(np.column_stack(preds), axis=1)

def _bootstrap_geo_mean_ci(vec, n_boot=N_BOOT, alpha=0.05):
    vec = np.asarray(vec, dtype=float)
    n = len(vec)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot[b] = gm(vec[idx])
    mean = float(boot.mean())
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return mean, float(lo), float(hi)

# Section 8: three curves (rice / soil / both)
x_ticks_pct = np.arange(0, 101, 10)
fracs = x_ticks_pct / 100.0

def _apply_reduction(df_, mode, frac):
    out = df_.copy()
    if mode in ('rice', 'both') and 'Diet_rice' in out.columns:
        out['Diet_rice'] = out['Diet_rice'] * (1.0 - frac)
    if mode in ('soil', 'both') and 'Soil_ingestion' in out.columns:
        out['Soil_ingestion'] = out['Soil_ingestion'] * (1.0 - frac)
    return recompute_aggregates(out)

curves = {}
for mode in ['rice', 'soil', 'both']:
    means, los, his = [], [], []
    for f in fracs:
        df_scn = _apply_reduction(df, mode, f)
        pred_vec = _ensemble_pred(df_scn)
        m, lo, hi = _bootstrap_geo_mean_ci(pred_vec, n_boot=N_BOOT, alpha=0.05)
        means.append(m); los.append(lo); his.append(hi)
    curves[mode] = {"mean": np.array(means), "lo": np.array(los), "hi": np.array(his)}

# -------- Nature-style curve plot (Rice / Soil / Both; GM ± 95% CI) --------
fig, ax = plt.subplots(figsize=(3.5, 2.6))
style_axes(ax)

def _plot_with_ci(ax, x, ymean, ylo, yhi, label, marker):
    yerr = np.vstack([ymean - ylo, yhi - ymean])
    ax.errorbar(
        x, ymean, yerr=yerr, marker=marker, linestyle='-',
        linewidth=1.4, capsize=2.5, label=label
    )

_plot_with_ci(ax, x_ticks_pct, curves['rice']['mean'], curves['rice']['lo'], curves['rice']['hi'],
              label='Rice only', marker='o')
_plot_with_ci(ax, x_ticks_pct, curves['soil']['mean'], curves['soil']['lo'], curves['soil']['hi'],
              label='Soil only', marker='s')
_plot_with_ci(ax, x_ticks_pct, curves['both']['mean'], curves['both']['lo'], curves['both']['hi'],
              label='Rice + soil', marker='^')

# Threshold lines: 1, 4, 5 µg/g
for thr, off in zip([1.0, 4.0, 5.0], [0.0, 0.0, 0.0]):
    ax.axhline(thr, color='k', linestyle='--', linewidth=0.8)
    ax.text(x_ticks_pct[-1], thr + off, f"  {thr:g} µg/g", va='bottom', ha='left', fontsize=7)


ax.set_xlim(0, 100)
ax.set_xticks(x_ticks_pct)
ax.set_xlabel("Reduction in pathway exposure (%)")
ax.set_ylabel("Predicted urinary Cd, GM (µg/g CREA)")
ax.legend(frameon=False, ncol=1, loc='upper right', handlelength=1.6, handletextpad=0.4, columnspacing=0.8, markerscale=0.9)


plt.tight_layout()
plt.savefig("Curve_UrineCd_vs_Reduction_GM.png")
plt.savefig("Curve_UrineCd_vs_Reduction_GM.pdf")
plt.close(fig)

# =========================
# 9) Total exposure (diet/environment/both) 10% reduction: GM+95% CI + 1 µg/g threshold
# =========================
def _apply_total_reduction(df_, group: str, frac: float):
    """
    group: 'diet' | 'env' | 'both'
    frac : 降幅比例(0~1)
    规则：对组成该总暴露的所有路径同比例缩放，然后重算 Diet_total/Env_total_calc
    """
    out = df_.copy()
    diet_cols_g = ['Diet_rice','Diet_solanaceous','Diet_legumes','Diet_root_tuber','Diet_other']
    env_cols_g  = ['Soil_ingestion','Water_ingestion']

    if group in ('diet', 'both'):
        for c in diet_cols_g:
            if c in out.columns:
                out[c] = out[c] * (1.0 - frac)
    if group in ('env', 'both'):
        for c in env_cols_g:
            if c in out.columns:
                out[c] = out[c] * (1.0 - frac)
    return recompute_aggregates(out)

x_ticks_pct2 = np.arange(0, 101, 10)
fracs2 = x_ticks_pct2 / 100.0

curves_tot = {}
for group in ['diet', 'env', 'both']:
    means, los, his = [], [], []
    for f in fracs2:
        df_scn = _apply_total_reduction(df, group, f)
        pred_vec = _ensemble_pred(df_scn)
        m, lo, hi = _bootstrap_geo_mean_ci(pred_vec, n_boot=N_BOOT, alpha=0.05)
        means.append(m); los.append(lo); his.append(hi)
    curves_tot[group] = {'mean': np.array(means), 'lo': np.array(los), 'hi': np.array(his)}

# -------- Nature-style curve plot (Total diet / Total environmental / Both) --------
fig, ax = plt.subplots(figsize=(3.5, 2.6))
style_axes(ax)

def _plot_with_ci(ax, x, ymean, ylo, yhi, label, marker):
    yerr = np.vstack([ymean - ylo, yhi - ymean])
    ax.errorbar(
        x, ymean, yerr=yerr, marker=marker, linestyle='-',
        linewidth=1.4, capsize=2.5, label=label
    )

_plot_with_ci(ax, x_ticks_pct2, curves_tot['diet']['mean'], curves_tot['diet']['lo'], curves_tot['diet']['hi'],
              label='Total diet', marker='o')
_plot_with_ci(ax, x_ticks_pct2, curves_tot['env']['mean'], curves_tot['env']['lo'], curves_tot['env']['hi'],
              label='Total environmental', marker='s')
_plot_with_ci(ax, x_ticks_pct2, curves_tot['both']['mean'], curves_tot['both']['lo'], curves_tot['both']['hi'],
              label='Diet + environmental', marker='^')

for thr, off in zip([1.0, 4.0, 5.0], [0.0, 0.0, 0.0]):
    ax.axhline(thr, color='k', linestyle='--', linewidth=0.8)
    ax.text(x_ticks_pct2[-1], thr + off, f"  {thr:g} µg/g", va='bottom', ha='left', fontsize=7)


ax.set_xlim(0, 100)
ax.set_xticks(x_ticks_pct2)
ax.set_xlabel("Reduction in total exposure (%)")
ax.set_ylabel("Predicted urinary Cd, GM (µg/g CREA)")

ax.legend(frameon=False, ncol=1, loc='upper right', handlelength=1.6, handletextpad=0.4, columnspacing=0.8, markerscale=0.9)


plt.tight_layout()
plt.savefig("Curve_UrineCd_vs_TotalExposureReduction_GM_3curves.png")
plt.savefig("Curve_UrineCd_vs_TotalExposureReduction_GM_3curves.pdf")
plt.close(fig)

# =========================
# 10) Extended curves: Smoke / Water / Combo (along with rice/soil/both)
# =========================
def _apply_smoke(df_, frac):
    """Proportional reduction in smoking (frac∈[0,1], 1=100% cessation)."""
    out = df_.copy()
    if 'Smoke' in out.columns:
        out['Smoke'] = out['Smoke'] * (1.0 - frac)
    return recompute_aggregates(out)

def _apply_water(df_, frac):
    """Proportional reduction in drinking-water ingestion pathway."""
    out = df_.copy()
    if 'Water_ingestion' in out.columns:
        out['Water_ingestion'] = out['Water_ingestion'] * (1.0 - frac)
    return recompute_aggregates(out)

def _apply_rice(df_, frac):
    """Proportional reduction in rice exposure only."""
    out = df_.copy()
    if 'Diet_rice' in out.columns:
        out['Diet_rice'] = out['Diet_rice'] * (1.0 - frac)
    return recompute_aggregates(out)

def _apply_soil(df_, frac):
    """Proportional reduction in soil ingestion only."""
    out = df_.copy()
    if 'Soil_ingestion' in out.columns:
        out['Soil_ingestion'] = out['Soil_ingestion'] * (1.0 - frac)
    return recompute_aggregates(out)

def _apply_both(df_, frac):
    """Proportional reduction in both rice and soil pathways."""
    out = _apply_rice(df_, frac)
    out = _apply_soil(out, frac)
    return recompute_aggregates(out)

def _apply_combo(df_, intensity):
    """
    多杠杆组合(intensity∈[0,1])：
      食品端：米Cd限值0.2 mg/kg(部分遵循：强度越大越接近上限)+ 稻米-40% + 茄科/根茎 -30%
      环境端：食土 -50% + 饮水 -40%
      行为端：Smoke → 0
    """
    out = df_.copy()


    if 'rice_Cd_mgkg' in out.columns:
        cd = out['rice_Cd_mgkg'].replace(0, np.nan)
        cap_factor = np.minimum(1.0, 0.2 / cd).fillna(1.0)
    else:
        cap_factor = 1.0


    if 'Diet_rice' in out.columns:
        out['Diet_rice'] = out['Diet_rice'] * (1.0 - intensity * (1.0 - cap_factor))
        out['Diet_rice'] = out['Diet_rice'] * (1.0 - 0.40 * intensity)


    for c in ['Diet_solanaceous', 'Diet_root_tuber']:
        if c in out.columns:
            out[c] = out[c] * (1.0 - 0.30 * intensity)


    if 'Soil_ingestion' in out.columns:
        out['Soil_ingestion'] = out['Soil_ingestion'] * (1.0 - 0.50 * intensity)
    if 'Water_ingestion' in out.columns:
        out['Water_ingestion'] = out['Water_ingestion'] * (1.0 - 0.40 * intensity)


    if 'Smoke' in out.columns:
        out['Smoke'] = out['Smoke'] * (1.0 - 1.0 * intensity)

    return recompute_aggregates(out)

# Compute curves (0–100%, step 10%)
x_ticks_pct_ext = np.arange(0, 101, 10)
fracs_ext = x_ticks_pct_ext / 100.0

def _curve_for(mode: str):
    means, los, his = [], [], []
    for f in fracs_ext:
        if mode == 'rice':
            df_scn = _apply_rice(df, f)
        elif mode == 'soil':
            df_scn = _apply_soil(df, f)
        elif mode == 'water':
            df_scn = _apply_water(df, f)
        elif mode == 'smoke':
            df_scn = _apply_smoke(df, f)
        elif mode == 'both':
            df_scn = _apply_both(df, f)
        elif mode == 'combo':
            df_scn = _apply_combo(df, f)
        else:
            raise ValueError("unknown mode")
        pred_vec = _ensemble_pred(df_scn)
        m, lo, hi = _bootstrap_geo_mean_ci(pred_vec, n_boot=N_BOOT, alpha=0.05)
        means.append(m); los.append(lo); his.append(hi)
    return {'mean': np.array(means), 'lo': np.array(los), 'hi': np.array(his)}

modes = [
    ('rice',  'Rice only',               'o'),
    ('soil',  'Soil only',               's'),
    ('water', 'Water only',              'D'),
    ('smoke', 'Smoking cessation',       'v'),
    ('both',  'Rice + Soil',             '^'),
    ('combo', 'Multi-lever combo',       'P'),
]
curves_ext = {m: _curve_for(m) for m,_,_ in modes}

# -------- Nature-style curve plot (rice/soil/water/smoke/both/combo) --------
fig, ax = plt.subplots(figsize=(3.6, 2.8))
style_axes(ax)

def _plot_with_ci(ax, x, ymean, ylo, yhi, label, marker):
    yerr = np.vstack([ymean - ylo, yhi - ymean])
    ax.errorbar(
        x, ymean, yerr=yerr, marker=marker, linestyle='-',
        linewidth=1.4, capsize=2.5, label=label
    )

for m, label, mk in modes:
    _plot_with_ci(ax, x_ticks_pct_ext, curves_ext[m]['mean'], curves_ext[m]['lo'], curves_ext[m]['hi'], label, mk)

for thr, off in zip([1.0, 4.0, 5.0], [0.0, 0.0, 0.0]):
    ax.axhline(thr, color='k', linestyle='--', linewidth=0.8)
    ax.text(x_ticks_pct_ext[-1], thr + off, f"  {thr:g} µg/g", va='bottom', ha='left', fontsize=7)


ax.set_xlim(0, 100); ax.set_xticks(x_ticks_pct_ext)
ax.set_xlabel("Reduction / policy intensity (%)")
ax.set_ylabel("Predicted urinary Cd, GM (µg/g CREA)")

ax.legend(frameon=False, ncol=2, loc='lower left', handlelength=1.6, handletextpad=0.4, columnspacing=0.8, markerscale=0.9)


plt.tight_layout()
plt.savefig("Curve_UrineCd_Rice-Soil-Smoke-Water-Combo_GM.png")
plt.savefig("Curve_UrineCd_Rice-Soil-Smoke-Water-Combo_GM.pdf")
plt.close(fig)

# =========================
# 11) Joint reduction response surface: diet × environment (2D/triangular/3D)
# =========================

def _reduce_diet_env(df_, diet_frac: float, env_frac: float):
    """
    diet_frac/env_frac ∈ [0,1]：对应总饮食/总环境暴露的下降比例。
    """
    out = df_.copy()
    diet_cols_g = ['Diet_rice','Diet_solanaceous','Diet_legumes','Diet_root_tuber','Diet_other']
    env_cols_g  = ['Soil_ingestion','Water_ingestion']
    for c in diet_cols_g:
        if c in out.columns:
            out[c] = out[c] * (1.0 - diet_frac)
    for c in env_cols_g:
        if c in out.columns:
            out[c] = out[c] * (1.0 - env_frac)
    return recompute_aggregates(out)

# --- Compute GM on grid (no bootstrap for speed; add if needed) ---
GRID_N = 21
diet_fracs = np.linspace(0.0, 1.0, GRID_N)
env_fracs  = np.linspace(0.0, 1.0, GRID_N)

Z = np.empty((GRID_N, GRID_N), dtype=float)  # [env, diet]
for i, fe in enumerate(env_fracs):
    for j, fd in enumerate(diet_fracs):
        df_scn = _reduce_diet_env(df, diet_frac=fd, env_frac=fe)
        pred_vec = _ensemble_pred(df_scn)
        Z[i, j] = gm(pred_vec)

# Grid for plotting (percent axes)
Dd, Ee = np.meshgrid(diet_fracs * 100.0, env_fracs * 100.0)
levels_thr = [1.0, 3.07, 5.0]

# -------- 11A) 2D response surface (heatmap + contours) --------
fig, ax = plt.subplots(figsize=(3.6, 3.1))
style_axes(ax)

pcm = ax.pcolormesh(Dd, Ee, Z, shading='auto')
cs  = ax.contour(Dd, Ee, Z, levels=levels_thr, colors='k', linewidths=0.9, linestyles='--')
ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f} µg/g", fontsize=7)

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("Predicted urinary Cd, GM (µg/g CREA)")

ax.set_xlabel("Reduction in total diet exposure (%)")
ax.set_ylabel("Reduction in total environmental exposure (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("Surface_UrineCd_GM_vs_DietEnv.png")
plt.savefig("Surface_UrineCd_GM_vs_DietEnv.pdf")
plt.close(fig)

# -------- 11B) Triangular budget ≤100% (show region fd+fe≤100%) --------
mask = (Dd + Ee) > 100.0
Z_tri = np.ma.array(Z, mask=mask)

fig, ax = plt.subplots(figsize=(3.6, 3.1))
style_axes(ax)

pcm = ax.pcolormesh(Dd, Ee, Z_tri, shading='auto')

cs = ax.contour(Dd, Ee, Z, levels=levels_thr, colors='k', linewidths=0.9, linestyles='--')
ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0f} µg/g", fontsize=7)

# Budget boundary: diet% + env% = 100%
ax.plot([0,100], [100,0], color='k', linestyle='-', linewidth=0.8)

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("Predicted urinary Cd, GM (µg/g CREA)")

ax.set_xlabel("Reduction in total diet exposure (%)")
ax.set_ylabel("Reduction in total environmental exposure (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("Surface_UrineCd_GM_TriangularBudget.png")
plt.savefig("Surface_UrineCd_GM_TriangularBudget.pdf")
plt.close(fig)

# -------- 11C) 3D surface (Z axis in place; horizontal colorbar on top) --------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


fig = plt.figure(figsize=(6.2, 4.2))
gs = fig.add_gridspec(
    2, 1,
    height_ratios=[0.10, 0.90],
    left=0.10, right=0.95, bottom=0.10, top=0.98, hspace=0.06
)
cax = fig.add_subplot(gs[0, 0])
ax3 = fig.add_subplot(gs[1, 0], projection='3d')

# Viewpoint: similar to Matplotlib default; Z axis on the right
ax3.view_init(elev=28, azim=-60)

# Surface
surf = ax3.plot_surface(
    Dd, Ee, Z, rstride=1, cstride=1, cmap='viridis',
    linewidth=0, antialiased=True, alpha=0.92
)

# Threshold planes (1/3/5 µg/g)
for thr, a in zip(levels_thr, [0.20, 0.14, 0.14]):
    ZZ = np.full_like(Z, thr, dtype=float)
    ax3.plot_surface(Dd, Ee, ZZ, rstride=8, cstride=8, color='k', alpha=a, linewidth=0)

# Axis labels (Nature style, no grid)
ax3.set_xlabel("Diet reduction (%)", labelpad=10)
ax3.set_ylabel("Environmental reduction (%)", labelpad=10)
ax3.set_zlabel("Urinary Cd GM (µg/g CREA)", labelpad=12)
ax3.xaxis.pane.set_alpha(0.0)
ax3.yaxis.pane.set_alpha(0.0)
ax3.zaxis.pane.set_alpha(0.0)
ax3.grid(False)
ax3.set_box_aspect((1.35, 1.0, 0.7))

# Horizontal colorbar on top (separate from main axes)
cbar = fig.colorbar(surf, cax=cax, orientation='horizontal')
cbar.set_label("Predicted urinary Cd, GM (µg/g CREA)")
cax.tick_params(length=3)

plt.savefig("Surface3D_UrineCd_GM_vs_DietEnv.png")
plt.savefig("Surface3D_UrineCd_GM_vs_DietEnv.pdf")
plt.close(fig)




# =========================
# End
# =========================
print("\nFigures saved:")
print(" - SHAP_beeswarm_all_3models_reversed.(png|pdf)")
print(" - RF_Scenario_DeltaGM_and_GM.(png|pdf)")
print(" - Curve_UrineCd_vs_Reduction_GM.(png|pdf)")
print(" - Curve_UrineCd_vs_TotalExposureReduction_GM_3curves.(png|pdf)")
print(" - Curve_UrineCd_Rice-Soil-Smoke-Water-Combo_GM.(png|pdf)")