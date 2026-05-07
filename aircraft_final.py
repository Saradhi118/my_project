# =============================================================
# aircraft_final.py
# - Exact plot styles from cl_cd_alpha.py
# - DE Optimization per Mach×Alt combination separately
# - All 5 plots match exactly: colors, markers, labels, titles
# =============================================================

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import differential_evolution

# =============================================================
# CONSTANTS
# =============================================================
SCALER_PATH    = "feature_scaler.pkl"
CL_MODEL_PATH  = "cl_xgb.model"
CD_MODEL_PATH  = "cd_xgb.model"
XCP_MODEL_PATH = "xcp_xgb.model"

FEATURES = [
    "nose length", "body_length", "wing LE", "root chord", "tip chord",
    "semi-span", "root th", "tip th", "wing sweep", "tail LE",
    "root chord.1", "tip chord.1", "semi-span.1", "root th.1",
    "tip th.1", "MACH", "ALPHA", "ALT",
]

# Exact alpha weights from cl_cd_alpha.py
ALPHA_WEIGHT_MAP = {
    0:  0.05,  2:  0.10,  4:  0.20,  6:  0.20,
    8:  0.10,  10: 0.10,  12: 0.05,  14: 0.05,
    16: 0.05,  18: 0.05,  20: 0.05,
}

# =============================================================
# LOAD MODELS
# =============================================================
def load_xgb_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m

def load_assets():
    sc  = joblib.load(SCALER_PATH)
    cl  = load_xgb_model(CL_MODEL_PATH)
    cd  = load_xgb_model(CD_MODEL_PATH)
    xcp = load_xgb_model(XCP_MODEL_PATH) if os.path.isfile(XCP_MODEL_PATH) else None
    return sc, cl, cd, xcp

try:
    scaler, cl_model, cd_model, xcp_model = load_assets()
    MODELS_LOADED = True
    LOAD_ERROR    = ""
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR    = str(e)
    scaler = cl_model = cd_model = xcp_model = None

# =============================================================
# CANVAS HELPER
# =============================================================
_canvas_handles = {}

def draw_figure(canvas_elem, fig, key):
    global _canvas_handles
    if key in _canvas_handles:
        try:
            _canvas_handles[key].get_tk_widget().destroy()
        except Exception:
            pass
    agg = FigureCanvasTkAgg(fig, canvas_elem.TKCanvas)
    agg.draw()
    agg.get_tk_widget().pack(fill="both", expand=True)
    _canvas_handles[key] = agg
    plt.close(fig)

# =============================================================
# WEIGHT SCHEME  (exact from cl_cd_alpha.py)
# =============================================================
def assign_weights(row, scheme):
    alpha_val    = int(row["ALPHA"])
    alpha_weight = ALPHA_WEIGHT_MAP.get(alpha_val, 1.0)
    if scheme == "uniform":
        return alpha_weight
    else:                       # variable
        return alpha_weight * 11

# =============================================================
# BACKEND: load CSV → predict → compute weights
# =============================================================
def run_backend(csv_path, mach_list, alt_list, weight_scheme):
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce")

    feats    = [f for f in FEATURES if f in df.columns]
    X_scaled = scaler.transform(df[feats])

    df["CL_pred"] = cl_model.predict(X_scaled)
    df["CD_pred"] = cd_model.predict(X_scaled)

    df["CL_CD_actual"] = df["CL"] / df["CD"]
    df["CL_CD_pred"]   = df["CL_pred"] / df["CD_pred"]

    # Exact weight logic from cl_cd_alpha.py
    df["weight_uniform"]  = df.apply(assign_weights, axis=1, scheme="uniform")
    df["weight_variable"] = df.apply(assign_weights, axis=1, scheme="variable")

    # Backend always uses weight_variable (matches cl_cd_alpha.py line 109)
    df["CL_CD_pred_weighted"] = df["CL_CD_pred"] * df["weight_variable"]

    # Filter per Mach×Alt
    subsets = []
    for mach in mach_list:
        for alt in alt_list:
            mask = np.isclose(df["MACH"], mach) & (df["ALT"] == alt)
            sub  = df.loc[mask].copy().sort_values("ALPHA")
            if not sub.empty:
                subsets.append((mach, alt, sub))

    return df, subsets

# =============================================================
# METRICS  (exact from train_and_save.py evaluate())
# =============================================================
def compute_metrics_table(df):
    rows = []
    for actual_col, pred_col, label in [
        ("CL", "CL_pred", "CL"),
        ("CD", "CD_pred", "CD"),
    ]:
        if actual_col in df.columns and pred_col in df.columns:
            act  = df[actual_col].dropna()
            pred = df[pred_col].loc[act.index]
            mae  = mean_absolute_error(act, pred)
            rmse = np.sqrt(mean_squared_error(act, pred))
            r2   = r2_score(act, pred)
            pct  = (mae / max(abs(act).mean(), 1e-6)) * 100
            rows.append([label,
                         f"{mae:.5f}", f"{rmse:.5f}",
                         f"{r2:.5f}",  f"{pct:.2f}%"])
    return rows

# =============================================================
# PLOTS — EXACT MATCH to cl_cd_alpha.py
# =============================================================

# ── Plot 1: Actual vs Predicted CL/CD  (plot_original_cl_cd) ─
def make_fig_original(subset, mach, altitude):
    """
    EXACT copy of plot_original_cl_cd() from cl_cd_alpha.py
    figsize=(9,5), whitegrid, tab:blue/o, tab:pink/s, linestyle="-"
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=subset, x="ALPHA", y="CL_CD_actual",
        label="Actual (blue)", color="tab:blue",
        marker="o", linestyle="-", errorbar=None, ax=ax,
    )
    sns.lineplot(
        data=subset, x="ALPHA", y="CL_CD_pred",
        label="Predicted (pink)", color="tab:pink",
        marker="s", linestyle="-", errorbar=None, ax=ax,
    )
    ax.set_title(f"CL/CD vs. α  (Mach = {mach}, Alt = {altitude} m)")
    ax.set_xlabel("Incidence Angle α (°)")
    ax.set_ylabel("Lift-to-Drag Ratio (CL/CD)")
    ax.legend(title="Series")
    fig.tight_layout()
    return fig

# ── Plot 2: Weighted Predicted CL/CD  (plot_weighted_pred) ───
def make_fig_weighted(subset, mach, altitude):
    """
    EXACT copy of plot_weighted_pred() from cl_cd_alpha.py
    figsize=(9,5), whitegrid, tab:blue/o, tab:pink/s, linestyle="-"
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=subset, x="ALPHA", y="CL_CD_actual",
        label="Actual (blue)", color="tab:blue",
        marker="o", linestyle="-", errorbar=None, ax=ax,
    )
    sns.lineplot(
        data=subset, x="ALPHA", y="CL_CD_pred_weighted",
        label="Weighted Pred (pink)", color="tab:pink",
        marker="s", linestyle="-", errorbar=None, ax=ax,
    )
    ax.set_title(
        f"Weighted Predicted CL/CD vs. α  (Mach = {mach}, Alt = {altitude} m)")
    ax.set_xlabel("Incidence Angle α (°)")
    ax.set_ylabel("Weighted Lift-to-Drag Ratio")
    ax.legend(title="Series")
    fig.tight_layout()
    return fig

# ── Plot 3: Residuals  ────────────────────────────────────────
def make_fig_residuals(df):
    """
    Residual plot for CL and CD.
    whitegrid, darkorange scatter, red dashed zero line.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (actual_col, pred_col, label) in zip(axes, [
        ("CL", "CL_pred", "CL"),
        ("CD", "CD_pred", "CD"),
    ]):
        res = df[actual_col] - df[pred_col]
        ax.scatter(df[pred_col], res,
                   alpha=0.5, color="darkorange",
                   edgecolors="k", linewidths=0.3, s=18)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel(f"Predicted {label}", fontsize=10)
        ax.set_ylabel("Residual  (Actual − Predicted)", fontsize=10)
        ax.set_title(f"{label}  Residual Plot", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

# ── Plot 4: Feature Importance  ───────────────────────────────
def make_fig_feature_importance(feature_cols):
    """
    Horizontal bar chart, teal bars, top-10 features.
    One subplot per model (CL, CD).
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (model, label) in zip(axes, [
        (cl_model, "CL Model"),
        (cd_model, "CD Model"),
    ]):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:10]
        names = [feature_cols[i] for i in idx][::-1]
        vals  = imp[idx][::-1]
        ax.barh(names, vals, color="teal",
                edgecolor="black", linewidth=0.4)
        ax.set_title(f"{label}: Feature Importance", fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

# ── Plot 5: Predicted vs Actual scatter  ─────────────────────
def make_fig_pred_vs_actual(df):
    """
    Scatter: steelblue for CL, darkorange for CD.
    Red dashed perfect-fit line.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (actual_col, pred_col, label, color) in zip(axes, [
        ("CL", "CL_pred", "CL", "steelblue"),
        ("CD", "CD_pred", "CD", "darkorange"),
    ]):
        actual = df[actual_col].dropna().values
        pred   = df[pred_col].loc[df[actual_col].dropna().index].values
        ax.scatter(actual, pred,
                   alpha=0.5, color=color,
                   edgecolors="k", linewidths=0.3, s=18)
        mn = min(actual.min(), pred.min())
        mx = max(actual.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel(f"Actual {label}", fontsize=10)
        ax.set_ylabel(f"Predicted {label}", fontsize=10)
        ax.set_title(f"{label}: Predicted vs Actual", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

# =============================================================
# OPTIMIZATION PER Mach×Alt
# =============================================================
def optimize_for_combo(mach, alt, df_full, maxiter, popsize, bmin, bmax):
    """
    Run DE for a specific Mach×Alt.
    Fixes MACH=mach, ALT=alt in the feature vector.
    Optimizes remaining 16 geometry/flight params.
    Returns best_x, best_cl, best_cd, best_ratio, convergence_log.
    """
    # Indices of MACH and ALT in FEATURES
    mach_idx = FEATURES.index("MACH")
    alt_idx  = FEATURES.index("ALT")

    # Use data bounds for this Mach×Alt combination
    mask   = np.isclose(df_full["MACH"], mach) & (df_full["ALT"] == alt)
    subset = df_full.loc[mask]

    conv_log = []

    def objective(x_free):
        # Reconstruct full 18-dim vector
        x_full = list(x_free)
        x_full.insert(mach_idx, mach)
        x_full.insert(alt_idx,  alt)
        xa = np.array(x_full).reshape(1, -1)
        xs = scaler.transform(xa)
        cl = cl_model.predict(xs)[0]
        cd = cd_model.predict(xs)[0]
        return -(cl / (cd + 1e-8))

    def callback(xk, convergence):
        x_full = list(xk)
        x_full.insert(mach_idx, mach)
        x_full.insert(alt_idx,  alt)
        xa = np.array(x_full).reshape(1, -1)
        xs = scaler.transform(xa)
        cl = cl_model.predict(xs)[0]
        cd = cd_model.predict(xs)[0]
        conv_log.append(cl / (cd + 1e-8))

    # Bounds for 16 free parameters (exclude MACH & ALT)
    free_features = [f for f in FEATURES if f not in ("MACH", "ALT")]
    bounds = []
    for f in free_features:
        if f in subset.columns and subset[f].notna().any():
            lo = float(subset[f].min())
            hi = float(subset[f].max())
            if lo == hi:
                lo -= 0.01; hi += 0.01
            bounds.append((lo, hi))
        else:
            bounds.append((bmin, bmax))

    result = differential_evolution(
        objective, bounds,
        maxiter=maxiter, popsize=popsize,
        callback=callback, seed=42, tol=1e-6,
        mutation=(0.5, 1.0), recombination=0.7,
        strategy="best1bin",
    )

    # Reconstruct best full vector
    best_free = list(result.x)
    best_full = best_free[:]
    best_full.insert(mach_idx, mach)
    best_full.insert(alt_idx,  alt)

    xa = np.array(best_full).reshape(1, -1)
    xs = scaler.transform(xa)
    best_cl  = cl_model.predict(xs)[0]
    best_cd  = cd_model.predict(xs)[0]
    best_xcp = xcp_model.predict(xs)[0] if xcp_model else float("nan")
    best_ratio = best_cl / (best_cd + 1e-8)

    return best_full, best_cl, best_cd, best_xcp, best_ratio, conv_log

def make_fig_opt_convergence(all_logs, combos):
    """Convergence curve for each Mach×Alt combo."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.tab10.colors
    for i, ((mach, alt), log) in enumerate(zip(combos, all_logs)):
        ax.plot(log, color=colors[i % 10],
                linewidth=2, label=f"Mach={mach}, Alt={alt}m")
    ax.set_title("DE Convergence per Mach×Alt Combination", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best CL/CD")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

def make_fig_opt_bar(results_list):
    """Bar chart comparing best CL/CD across Mach×Alt."""
    sns.set_style("whitegrid")
    labels = [f"M={r[0]}\nAlt={r[1]}" for r in results_list]
    ratios = [r[5] for r in results_list]   # best_ratio
    fig, ax = plt.subplots(figsize=(max(6, 3*len(labels)), 4))
    bars = ax.bar(labels, ratios, color="steelblue",
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Best CL/CD per Mach×Alt Combination", fontweight="bold")
    ax.set_ylabel("Max CL/CD")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

# =============================================================
# TRAINING  (train_and_save.py logic)
# =============================================================
def run_training(csv_path, window):
    data = pd.read_csv(csv_path)
    const_cols = [c for c in data.columns if data[c].nunique() <= 1]
    drop_cols  = const_cols + (["CASEID"] if "CASEID" in data.columns else [])
    data_clean = data.drop(columns=drop_cols)
    data_num   = data_clean.apply(pd.to_numeric, errors="coerce")

    feats = [f for f in FEATURES if f in data_num.columns]
    X     = data_num[feats]
    xcp_col = "X-C.P." if "X-C.P." in data_num.columns else None
    y_cols  = ["CL", "CD"] + ([xcp_col] if xcp_col else [])
    y       = data_num[y_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    params = dict(n_estimators=100, learning_rate=0.1, max_depth=10,
                  subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", grow_policy="depthwise", verbosity=0)

    log = ""
    for col, tag in [("CL","CL"), ("CD","CD")] + (
            [(xcp_col, "XCP")] if xcp_col else []):
        window["train_log"].update(f"Training {tag} model...\n", append=True)
        window.refresh()

    cl_m = xgb.XGBRegressor(**params); cl_m.fit(Xtr, y_train["CL"])
    cd_m = xgb.XGBRegressor(**params); cd_m.fit(Xtr, y_train["CD"])
    xcp_m = None
    if xcp_col:
        xcp_m = xgb.XGBRegressor(**params); xcp_m.fit(Xtr, y_train[xcp_col])

    def ev(true, pred, lbl):
        mae  = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2   = r2_score(true, pred)
        pct  = (mae / max(abs(true).mean(), 1e-6)) * 100
        return (f"  {lbl:<14} MAE={mae:.4f}  RMSE={rmse:.4f}"
                f"  R²={r2:.4f}  MAE%={pct:.2f}%\n")

    log  = "\n--- Training Metrics ---\n"
    log += ev(y_train["CL"], cl_m.predict(Xtr), "CL  Train")
    log += ev(y_train["CD"], cd_m.predict(Xtr), "CD  Train")
    log += "\n--- Testing Metrics ---\n"
    log += ev(y_test["CL"],  cl_m.predict(Xte), "CL  Test")
    log += ev(y_test["CD"],  cd_m.predict(Xte), "CD  Test")

    cl_m.save_model(CL_MODEL_PATH)
    cd_m.save_model(CD_MODEL_PATH)
    if xcp_m:
        xcp_m.save_model(XCP_MODEL_PATH)
    joblib.dump(sc, SCALER_PATH)
    log += "\n✅ All models & scaler saved.\n"
    return log

# =============================================================
# GUI LAYOUT
# =============================================================

# ── Tab 1: Train ──────────────────────────────────────────────
tab_train = [
    [sg.Text("Train XGBoost Models from CSV",
             font=("Helvetica", 12, "bold"))],
    [sg.Text("CSV File:"),
     sg.Input(key="train_csv", size=(40, 1)),
     sg.FileBrowse(file_types=(("CSV", "*.csv"),))],
    [sg.Button("Train & Save Models", size=(20, 1),
               button_color=("white", "#9C27B0"))],
    [sg.Text("Training Log:", font=("Helvetica", 10, "bold"))],
    [sg.Multiline(size=(68, 14), key="train_log",
                  font=("Courier", 9), disabled=True, autoscroll=True)],
]

# ── Tab 2: Predict (single point) ────────────────────────────
input_rows = [
    [sg.Text(n, size=(17, 1)),
     sg.Input(key=f"inp_{n}", size=(12, 1))]
    for n in FEATURES
]
tab_predict = [
    [sg.Text("Single-Point Prediction",
             font=("Helvetica", 12, "bold"))],
    [sg.Column(input_rows, scrollable=True,
               vertical_scroll_only=True, size=(390, 360))],
    [sg.Button("Predict", size=(12, 1),
               button_color=("white", "#2196F3"))],
    [sg.Multiline(size=(62, 6), key="pred_out",
                  font=("Courier", 10), disabled=True)],
    [sg.Canvas(key="pred_canvas", size=(500, 290))],
]

# ── Tab 3: Evaluate (backend) ────────────────────────────────
tab_evaluate = [
    [sg.Text("Backend Evaluation  —  cl_cd_alpha.py logic",
             font=("Helvetica", 12, "bold"))],
    [sg.Text("CSV File:"),
     sg.Input(key="eval_csv", size=(37, 1)),
     sg.FileBrowse(file_types=(("CSV", "*.csv"),))],
    [sg.Text("Mach numbers (comma-sep):"),
     sg.Input("0.5,0.8", key="mach_list", size=(16, 1)),
     sg.Text("  Altitudes m (comma-sep):"),
     sg.Input("0,6000", key="alt_list", size=(14, 1))],
    [sg.Text("Weight scheme:"),
     sg.Combo(["uniform", "variable"],
              default_value="variable",
              key="weight_scheme", size=(10, 1))],
    [sg.Button("Run Backend", size=(14, 1),
               button_color=("white", "#4CAF50")),
     sg.Button("Save CSVs", size=(12, 1),
               button_color=("white", "#607D8B"),
               disabled=True, key="save_csvs")],

    # Metrics table
    [sg.Text("─" * 74)],
    [sg.Text("📊 Error Metrics", font=("Helvetica", 11, "bold"))],
    [sg.Table(
        values=[["─", "─", "─", "─", "─"]],
        headings=["Target", "MAE", "RMSE", "R²", "MAE%"],
        key="metrics_table",
        auto_size_columns=False,
        col_widths=[8, 11, 11, 11, 9],
        num_rows=3,
        justification="center",
        alternating_row_color="#f0f4ff",
        header_background_color="#4CAF50",
        header_text_color="white",
        font=("Courier", 10),
    )],
    [sg.Text("─" * 74)],

    # Plot tabs — one canvas each, exact styles
    [sg.Text("📈 Plots  (per Mach×Alt combination)",
             font=("Helvetica", 11, "bold"))],
    [sg.Text("Select Mach×Alt:"),
     sg.Combo([], key="combo_select", size=(20, 1),
              enable_events=True, readonly=True)],
    [sg.TabGroup([[
        sg.Tab("Actual vs Pred CL/CD",
               [[sg.Canvas(key="cv_orig",  size=(760, 380))]]),
        sg.Tab("Weighted Pred CL/CD",
               [[sg.Canvas(key="cv_wt",    size=(760, 380))]]),
        sg.Tab("Residuals",
               [[sg.Canvas(key="cv_res",   size=(760, 380))]]),
        sg.Tab("Pred vs Actual",
               [[sg.Canvas(key="cv_pva",   size=(760, 380))]]),
        sg.Tab("Feature Importance",
               [[sg.Canvas(key="cv_fi",    size=(760, 380))]]),
    ]])],
]

# ── Tab 4: Optimize per Mach×Alt ─────────────────────────────
tab_optimize = [
    [sg.Text("DE Optimization  —  per Mach×Alt Combination",
             font=("Helvetica", 12, "bold"))],
    [sg.Text("CSV (for data bounds):"),
     sg.Input(key="opt_csv", size=(34, 1)),
     sg.FileBrowse(file_types=(("CSV", "*.csv"),))],
    [sg.Text("Mach numbers (comma-sep):"),
     sg.Input("0.5,0.8", key="opt_mach", size=(16, 1)),
     sg.Text("  Altitudes m (comma-sep):"),
     sg.Input("0,6000",  key="opt_alt",  size=(14, 1))],
    [sg.Text("Max Iterations:"),
     sg.Input("50", key="maxiter", size=(6, 1)),
     sg.Text("  Popsize:"),
     sg.Input("10", key="popsize", size=(6, 1))],
    [sg.Text("Global Bounds  Min:"),
     sg.Input("0.1", key="bmin", size=(8, 1)),
     sg.Text("  Max:"),
     sg.Input("10.0", key="bmax", size=(8, 1))],
    [sg.Button("Optimize All Combos", size=(20, 1),
               button_color=("white", "#FF5722"))],
    [sg.ProgressBar(100, orientation="h", size=(44, 16),
                    key="opt_prog", visible=False)],
    [sg.Multiline(size=(62, 12), key="opt_out",
                  font=("Courier", 9), disabled=True, autoscroll=True)],
    [sg.TabGroup([[
        sg.Tab("Convergence per Combo",
               [[sg.Canvas(key="cv_conv", size=(760, 340))]]),
        sg.Tab("Best CL/CD Comparison",
               [[sg.Canvas(key="cv_bar",  size=(760, 340))]]),
    ]])],
]

# ── Main window ───────────────────────────────────────────────
status_text = (
    "⚠ Models NOT loaded — train first or check file paths."
    if not MODELS_LOADED
    else "✅ Models loaded successfully."
)
layout = [
    [sg.Text("✈  Aircraft Aerodynamics Optimization Tool",
             font=("Helvetica", 15, "bold"), text_color="#1a237e")],
    [sg.Text(status_text, key="model_status",
             text_color="red" if not MODELS_LOADED else "green",
             font=("Helvetica", 10, "bold"))],
    [sg.TabGroup([[
        sg.Tab("🛠  Train",    tab_train),
        sg.Tab("🔍  Predict",  tab_predict),
        sg.Tab("📊  Evaluate", tab_evaluate),
        sg.Tab("⚙  Optimize",  tab_optimize),
    ]])],
    [sg.StatusBar("Ready", key="status", size=(75, 1))],
]

window = sg.Window(
    "Aircraft Optimization Tool",
    layout,
    finalize=True,
    resizable=True,
    size=(940, 820),
)

# =============================================================
# GLOBAL STATE
# =============================================================
eval_subsets  = []      # [(mach, alt, subset_df), ...]
eval_df_full  = None    # full enriched dataframe after Run Backend
eval_feats    = []      # feature columns actually found in CSV

# =============================================================
# EVENT LOOP
# =============================================================
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    # ── Tab 1: TRAIN ──────────────────────────────────────────
    if event == "Train & Save Models":
        csv_path = values["train_csv"].strip()
        if not csv_path or not os.path.exists(csv_path):
            sg.popup_error("Select a valid CSV file first.")
            continue
        window["train_log"].update("")
        window.refresh()
        try:
            window["status"].update("⏳ Training models...")
            log = run_training(csv_path, window)
            window["train_log"].update(log, append=True)
            scaler, cl_model, cd_model, xcp_model = load_assets()
            MODELS_LOADED = True
            window["model_status"].update(
                "✅ Models loaded successfully.", text_color="green")
            window["status"].update(
                "✅ Training complete — models saved & reloaded")
        except Exception as e:
            window["train_log"].update(
                f"❌ Training error:\n{e}\n", append=True)
            window["status"].update(f"Error: {e}")

    # ── Tab 2: PREDICT ────────────────────────────────────────
    if event == "Predict":
        if not MODELS_LOADED:
            sg.popup_error("Models not loaded. Train first.")
            continue
        try:
            x = [float(values[f"inp_{n}"]) for n in FEATURES]
        except ValueError:
            window["pred_out"].update(
                "❌ Fill all 18 input fields with numbers.\n")
            continue
        try:
            xa  = np.array(x).reshape(1, -1)
            xs  = scaler.transform(xa)
            cl  = cl_model.predict(xs)[0]
            cd  = cd_model.predict(xs)[0]
            xcp = xcp_model.predict(xs)[0] if xcp_model else float("nan")
            ratio = cl / (cd + 1e-8)
            window["pred_out"].update(
                f"{'='*44}\n"
                f"  CL   (Lift Coeff.)      = {cl:>12.6f}\n"
                f"  CD   (Drag Coeff.)      = {cd:>12.6f}\n"
                f"  XCP  (Centre Pressure)  = {xcp:>12.6f}\n"
                f"  CL/CD                   = {ratio:>12.6f}\n"
                f"{'='*44}\n"
            )
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(["CL", "CD", "XCP"], [cl, cd, xcp],
                          color=["steelblue", "darkorange", "green"],
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, [cl, cd, xcp]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)
            ax.set_title("Single-Point Prediction", fontweight="bold")
            ax.set_ylabel("Value")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            fig.tight_layout()
            draw_figure(window["pred_canvas"], fig, "pred_canvas")
            window["status"].update("✅ Prediction complete")
        except Exception as e:
            window["pred_out"].update(f"❌ Error: {e}\n")

    # ── Tab 3: EVALUATE ───────────────────────────────────────
    if event == "Run Backend":
        if not MODELS_LOADED:
            sg.popup_error("Models not loaded. Train first.")
            continue
        csv_path = values["eval_csv"].strip()
        if not csv_path or not os.path.exists(csv_path):
            sg.popup_error("Select a valid CSV file.")
            continue
        try:
            mach_list = [float(v.strip())
                         for v in values["mach_list"].split(",")]
            alt_list  = [int(float(v.strip()))
                         for v in values["alt_list"].split(",")]
        except ValueError:
            sg.popup_error("Enter valid Mach / Alt values.")
            continue
        try:
            window["status"].update("⏳ Running backend evaluation...")
            window.refresh()

            df_full, subsets = run_backend(
                csv_path, mach_list, alt_list, values["weight_scheme"])
            eval_subsets = subsets
            eval_df_full = df_full
            eval_feats   = [f for f in FEATURES if f in df_full.columns]

            # Metrics table
            rows = compute_metrics_table(df_full)
            window["metrics_table"].update(
                values=rows if rows
                else [["No CL/CD cols", "", "", "", ""]])

            # Populate Mach×Alt selector
            combo_labels = [f"Mach={m}, Alt={a}m"
                            for m, a, _ in subsets]
            window["combo_select"].update(
                values=combo_labels,
                value=combo_labels[0] if combo_labels else "")

            # Draw plots for first combo
            if subsets:
                m0, a0, s0 = subsets[0]
                draw_figure(window["cv_orig"],
                            make_fig_original(s0, m0, a0), "cv_orig")
                draw_figure(window["cv_wt"],
                            make_fig_weighted(s0, m0, a0), "cv_wt")

            # Residuals + Pred vs Actual use full dataset
            draw_figure(window["cv_res"],
                        make_fig_residuals(df_full), "cv_res")
            draw_figure(window["cv_pva"],
                        make_fig_pred_vs_actual(df_full), "cv_pva")
            draw_figure(window["cv_fi"],
                        make_fig_feature_importance(eval_feats), "cv_fi")

            window["save_csvs"].update(disabled=False)
            window["status"].update(
                f"✅ Done — {len(df_full)} rows | "
                f"{len(subsets)} Mach×Alt combos")
        except Exception as e:
            sg.popup_error(f"Backend Error:\n{e}")
            window["status"].update(f"Error: {e}")

    # ── Mach×Alt selector changed → redraw plots 1 & 2 ───────
    if event == "combo_select" and eval_subsets:
        selected = values["combo_select"]
        for mach, alt, sub in eval_subsets:
            if f"Mach={mach}, Alt={alt}m" == selected:
                draw_figure(window["cv_orig"],
                            make_fig_original(sub, mach, alt), "cv_orig")
                draw_figure(window["cv_wt"],
                            make_fig_weighted(sub, mach, alt), "cv_wt")
                break

    # ── Save CSVs ─────────────────────────────────────────────
    if event == "Save CSVs":
        if not eval_subsets:
            sg.popup_error("Run Backend first.")
            continue
        saved = []
        for mach, alt, sub in eval_subsets:
            cols = [c for c in [
                "ALPHA", "CL", "CD", "CL_pred", "CD_pred",
                "CL_CD_actual", "CL_CD_pred",
                "weight_uniform", "weight_variable",
                "CL_CD_pred_weighted",
            ] if c in sub.columns]
            fname = f"results_M{mach}_ALT{alt}.csv"
            sub[cols].to_csv(fname, index=False)
            saved.append(fname)
        sg.popup("Saved:\n" + "\n".join(saved))
        window["status"].update(f"✅ Saved {len(saved)} CSV(s)")

    # ── Tab 4: OPTIMIZE per Mach×Alt ─────────────────────────
    if event == "Optimize All Combos":
        if not MODELS_LOADED:
            sg.popup_error("Models not loaded. Train first.")
            continue
        csv_path = values["opt_csv"].strip()
        if not csv_path or not os.path.exists(csv_path):
            sg.popup_error("Select a CSV file for data bounds.")
            continue
        try:
            mach_list = [float(v.strip())
                         for v in values["opt_mach"].split(",")]
            alt_list  = [int(float(v.strip()))
                         for v in values["opt_alt"].split(",")]
            maxiter   = int(values["maxiter"])
            popsize   = int(values["popsize"])
            bmin      = float(values["bmin"])
            bmax      = float(values["bmax"])
        except ValueError:
            sg.popup_error("Check Mach/Alt/iteration settings.")
            continue
        try:
            # Load data for bounds
            df_opt = pd.read_csv(csv_path)
            df_opt = df_opt.apply(pd.to_numeric, errors="coerce")

            combos = [(m, a)
                      for m in mach_list for a in alt_list]
            total  = len(combos)

            window["opt_out"].update(
                f"Optimizing {total} Mach×Alt combinations...\n\n")
            window["opt_prog"].update(visible=True, current_count=0)
            window.refresh()

            all_logs    = []
            all_results = []   # (mach, alt, best_x, cl, cd, ratio)

            for i, (mach, alt) in enumerate(combos):
                window["opt_out"].update(
                    f"{'─'*44}\n"
                    f"▶  Mach={mach}, Alt={alt}m\n",
                    append=True)
                window.refresh()

                try:
                    bx, bcl, bcd, bxcp, bratio, log = optimize_for_combo(
                        mach, alt, df_opt,
                        maxiter, popsize, bmin, bmax)

                    all_logs.append(log)
                    all_results.append((mach, alt, bx, bcl, bcd, bratio))

                    param_str = "\n".join(
                        f"    {FEATURES[k]:<18} = {bx[k]:.5f}"
                        for k in range(len(FEATURES)))

                    window["opt_out"].update(
                        f"  CL        = {bcl:.6f}\n"
                        f"  CD        = {bcd:.6f}\n"
                        f"  XCP       = {bxcp:.6f}\n"
                        f"  Max CL/CD = {bratio:.6f}\n"
                        f"  Optimal Parameters:\n{param_str}\n",
                        append=True)
                except Exception as ex:
                    window["opt_out"].update(
                        f"  ❌ Error: {ex}\n", append=True)
                    all_logs.append([])
                    all_results.append((mach, alt, [], 0, 1, 0))

                pct = int((i + 1) / total * 100)
                window["opt_prog"].update(current_count=pct)
                window.refresh()

            # Summary
            window["opt_out"].update(
                f"\n{'='*44}\n  SUMMARY\n{'='*44}\n",
                append=True)
            best_overall = max(all_results, key=lambda r: r[5])
            for r in all_results:
                window["opt_out"].update(
                    f"  Mach={r[0]}, Alt={r[1]}m  →  "
                    f"CL/CD = {r[5]:.5f}\n",
                    append=True)
            window["opt_out"].update(
                f"\n🏆 Best: Mach={best_overall[0]}, "
                f"Alt={best_overall[1]}m  →  "
                f"CL/CD = {best_overall[5]:.5f}\n",
                append=True)

            # Convergence plot
            draw_figure(
                window["cv_conv"],
                make_fig_opt_convergence(all_logs, combos),
                "cv_conv")

            # Bar comparison
            draw_figure(
                window["cv_bar"],
                make_fig_opt_bar(all_results),
                "cv_bar")

            window["status"].update(
                f"✅ Optimization done | "
                f"Best CL/CD = {best_overall[5]:.5f} "
                f"at Mach={best_overall[0]}, Alt={best_overall[1]}m")

        except Exception as e:
            window["opt_out"].update(f"❌ Error: {e}\n")
            window["status"].update(f"Error: {e}")

window.close()
