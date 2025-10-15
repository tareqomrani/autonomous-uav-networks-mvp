# ===============================================================
# Autonomous UAV Networks Simulator (Neon Green Theme)
# ---------------------------------------------------------------
# • Autonomous UAV network sim with MAC/Routing, adversaries, energy
# • 3D orbit visualization (LEO/MEO/GEO)
# • CSV / JSON / PDF Mission Report exports
# • Banner-matched neon-green on black UI
# ===============================================================

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import io, json

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF for mission report
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------------------------------------------------------------
# THEME (matches banner)
# ---------------------------------------------------------------
THEME = {
    "bg": "#0a0f0a",           # near-black
    "panel": "#0c1310",        # sidebar/panels
    "grid": "#123a28",         # subtle grid lines
    "txt": "#d8ffe9",          # light mint text
    "neon": "#22ff88",         # primary neon
    "neon_mid": "#17d473",
    "neon_deep": "#0fa25f",
    "link": "#1aff80",
}

def inject_app_css():
    st.markdown(f"""
    <style>
      .stApp {{ background: {THEME['bg']}; color: {THEME['txt']}; }}
      [data-testid="stSidebar"] {{ background: {THEME['panel']}; color: {THEME['txt']}; }}
      h1, h2, h3, h4, h5, h6 {{ color: {THEME['txt']} !important; }}
      .stDownloadButton button, .stButton button {{
        border: 1px solid {THEME['neon_mid']}; color: {THEME['txt']};
        background: transparent;
      }}
      .stDownloadButton button:hover, .stButton button:hover {{
        box-shadow: 0 0 14px {THEME['neon']};
        border-color: {THEME['neon']};
      }}
      .stDataFrame div[data-testid="stTable"] {{ color: {THEME['txt']}; }}
    </style>
    """, unsafe_allow_html=True)

def greenify(fig, title=None):
    fig.update_layout(
        title=title,
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["txt"]),
        legend=dict(font=dict(color=THEME["txt"])),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        colorway=[THEME["neon"], THEME["neon_mid"], THEME["neon_deep"]],
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=THEME["grid"],
        zeroline=False, linecolor=THEME["grid"], ticks="", showline=True
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=THEME["grid"],
        zeroline=False, linecolor=THEME["grid"], ticks="", showline=True
    )
    return fig

# ---------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------
@dataclass
class UAV:
    uid: int
    role: str
    pos: np.ndarray
    v_max: float = 12.0
    battery_Wh: float = 150.0
    payload_kg: float = 0.5
    tx_power_W: float = 1.0
    rx_noise_W: float = 1e-9
    energy_used_Wh: float = 0.0
    risk_aversion: float = 0.4
    energy_aversion: float = 0.4
    throughput_preference: float = 0.3

    def move_toward(self, target: np.ndarray, dt: float, keep_in_bounds: Tuple[float, float] = (1000, 1000)):
        vec = target - self.pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return
        step = min(self.v_max * dt, dist)
        self.pos = self.pos + (vec / dist) * step
        self.pos[0] = np.clip(self.pos[0], 0, keep_in_bounds[0])
        self.pos[1] = np.clip(self.pos[1], 0, keep_in_bounds[1])

@dataclass
class Adversary:
    kind: str
    pos: np.ndarray
    power_W: float = 2.0
    radius_m: float = 250.0

@dataclass
class ChannelModel:
    f_GHz: float = 2.4
    pl0_dB: float = 40.0
    d0_m: float = 1.0
    n: float = 2.2
    shadowing_std_dB: float = 2.0
    rng: random.Random = field(default_factory=random.Random)

    def pathloss_linear(self, d_m: float, shadow: bool = True):
        if d_m < 1e-3:
            d_m = 1e-3
        pl_dB = self.pl0_dB + 10 * self.n * math.log10(d_m / self.d0_m)
        if shadow:
            pl_dB += self.rng.gauss(0.0, self.shadowing_std_dB)
        return 10.0 ** (-pl_dB / 10.0)

# ---------------------------------------------------------------
# Utility
# ---------------------------------------------------------------
def capacity_bps(tx_power_W: float, gain_linear: float, noise_W: float, mac_share: float = 1.0):
    sinr = (tx_power_W * gain_linear) / max(noise_W, 1e-15)
    return mac_share * math.log2(1.0 + sinr)

def comm_energy_Wh(tx_power_W: float, seconds: float):
    return (tx_power_W * seconds) / 3600.0

def motion_energy_Wh(distance_m: float, mass_factor: float = 0.25):
    return distance_m * mass_factor / 1000.0

def mac_share(num_links: int, scheme: str):
    if num_links <= 0:
        return 0.0
    s = scheme.lower()
    if "tdma" in s:
        return 1.0 / num_links
    if "noma" in s:
        return min(1.0, 0.65 + 0.5 / num_links)
    if "rsma" in s:
        return min(1.0, 0.75 + 0.6 / num_links)
    return 1.0 / num_links

def eaves_risk(midpoint: np.ndarray, eaves: Optional[Adversary]):
    if not eaves:
        return 0.0
    d = np.linalg.norm(midpoint - eaves.pos)
    return float(np.clip(1.0 - d / eaves.radius_m, 0.0, 1.0))

# ---------------------------------------------------------------
# Graph + Routing
# ---------------------------------------------------------------
def build_graph(uavs, ch, jammer, noise_W, link_thresh_bps, mac_scheme):
    G = nx.DiGraph()
    for u in uavs:
        G.add_node(u.uid, pos=(u.pos[0], u.pos[1]), role=u.role)

    positions = np.vstack([u.pos for u in uavs])
    dists = cdist(positions, positions)
    N = len(uavs)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            g = ch.pathloss_linear(dists[i, j])
            extra_noise = 0.0
            if jammer and jammer.kind == "jammer":
                if np.linalg.norm(uavs[i].pos - jammer.pos) <= jammer.radius_m or np.linalg.norm(uavs[j].pos - jammer.pos) <= jammer.radius_m:
                    extra_noise += jammer.power_W * ch.pathloss_linear(np.linalg.norm(uavs[j].pos - jammer.pos), shadow=False)
            mac = mac_share(N - 1, mac_scheme)
            cap = capacity_bps(uavs[i].tx_power_W, g, noise_W + extra_noise, mac)
            if cap >= link_thresh_bps:
                G.add_edge(uavs[i].uid, uavs[j].uid, capacity_bps=cap)
    return G

def route(G, src, dst):
    for u, v, d in G.edges(data=True):
        d["w"] = 1.0 / max(d["capacity_bps"], 1e-9)
    try:
        return nx.shortest_path(G, src, dst, weight="w")
    except Exception:
        return None

# ---------------------------------------------------------------
# Simulation Core
# ---------------------------------------------------------------
def pick_waypoint(uav, targets, jammer, eaves, bounds=(1000, 1000)):
    goal = targets.get(uav.role, np.array([bounds[0] / 2, bounds[1] / 2]))
    candidate = goal.copy()

    def nudge_away(p, adv, strength=120.0):
        if not adv:
            return p
        d = np.linalg.norm(p - adv.pos)
        if d < adv.radius_m:
            vec = p - adv.pos
            if np.linalg.norm(vec) < 1e-6:
                vec = np.array([1.0, 0.0])
            return p + (vec / np.linalg.norm(vec)) * strength
        return p

    candidate = nudge_away(candidate, jammer)
    candidate = nudge_away(candidate, eaves)
    center = np.array([bounds[0] / 2, bounds[1] / 2])
    candidate += (center - uav.pos) * (0.1 * uav.energy_aversion)
    return np.clip(candidate, [0, 0], list(bounds))

def run_sim(seed, num_uav, area_xy, steps, dt, src_count, sink_count, mac_scheme,
            link_thresh_bps, jammer_cfg, eaves_cfg, ch_params):
    rng = np.random.RandomState(seed)
    random.seed(seed)

    roles = ["source"] * src_count + ["sink"] * sink_count
    roles += ["relay"] * (num_uav - len(roles))
    rng.shuffle(roles)

    uavs = [UAV(uid=i, role=roles[i], pos=rng.rand(2) * np.array(area_xy)) for i in range(num_uav)]

    jammer = Adversary("jammer", np.array(jammer_cfg["pos"]), jammer_cfg["power_W"], jammer_cfg["radius_m"]) if jammer_cfg.get("enabled") else None
    eaves = Adversary("eaves", np.array(eaves_cfg["pos"]), 0, eaves_cfg["radius_m"]) if eaves_cfg.get("enabled") else None
    ch = ChannelModel(**ch_params, rng=random.Random(seed))

    targets = {
        "source": np.array([0.15 * area_xy[0], 0.85 * area_xy[1]]),
        "sink": np.array([0.85 * area_xy[0], 0.15 * area_xy[1]]),
        "relay": np.array([0.5 * area_xy[0], 0.5 * area_xy[1]]),
    }

    metrics = []
    for t in range(steps):
        G = build_graph(uavs, ch, jammer, 1e-9, link_thresh_bps, mac_scheme)
        sources = [u.uid for u in uavs if u.role == "source"]
        sinks = [u.uid for u in uavs if u.role == "sink"]

        total_thr, total_risk, total_Ecomm, links = 0.0, 0.0, 0.0, 0
        for s in sources:
            if not sinks:
                continue
            d = random.choice(sinks)
            p = route(G, s, d)
            if not p or len(p) < 2:
                continue
            caps, risks = [], []
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                cap = G.edges[u, v]["capacity_bps"]
                u_pos = [U for U in uavs if U.uid == u][0].pos
                v_pos = [U for U in uavs if U.uid == v][0].pos
                risks.append(eaves_risk(0.5 * (u_pos + v_pos), eaves))
                caps.append(cap)
            total_thr += min(caps) if caps else 0.0
            total_risk += np.mean(risks) if risks else 0.0
            links += len(caps)
            for i in range(len(p) - 1):
                node = [U for U in uavs if U.uid == p[i]][0]
                node.energy_used_Wh += comm_energy_Wh(node.tx_power_W, dt)
                total_Ecomm += comm_energy_Wh(node.tx_power_W, dt)

        total_risk /= max(len(sources), 1)
        for u in uavs:
            wp = pick_waypoint(u, targets, jammer, eaves, area_xy)
            pre = u.pos.copy()
            u.move_toward(wp, dt, keep_in_bounds=area_xy)
            dist = np.linalg.norm(u.pos - pre)
            u.energy_used_Wh += motion_energy_Wh(dist)

        avgBatt = np.mean([max(u.battery_Wh - u.energy_used_Wh, 0.0) for u in uavs])
        metrics.append(dict(t=t, throughput_bps=total_thr, avg_eaves_risk_0to1=total_risk,
                            used_links=links, avg_remaining_battery_Wh=avgBatt,
                            total_comm_energy_Wh=total_Ecomm))
    return uavs, pd.DataFrame(metrics), area_xy, jammer, eaves, ch

# ---------------------------------------------------------------
# 3D Orbit Visualization
# ---------------------------------------------------------------
def _orbit_ring_xyz(R, tilt=0, n=360):
    t = np.linspace(0, 2 * np.pi, n)
    x, y, z = R * np.cos(t), R * np.sin(t), np.zeros_like(t)
    tilt = np.deg2rad(tilt)
    Ry = y * np.cos(tilt) - z * np.sin(tilt)
    Rz = y * np.sin(tilt) + z * np.cos(tilt)
    return x, Ry, Rz

def _sphere_mesh(R=400, nu=64, nv=32):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, nu), np.linspace(0, np.pi, nv))
    return R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v), R * np.cos(v)

def make_orbit_figure(uavs, area_xy, Rp=400, LEO=520, MEO=700, GEO=880, tilt=0, alpha=0.18):
    xs, ys, zs = _sphere_mesh(Rp)
    fig = go.Figure()

    # Planet
    fig.add_surface(
        x=xs, y=ys, z=zs,
        opacity=alpha, showscale=False,
        colorscale=[[0, "rgb(5,40,25)"], [1, "rgb(10,70,45)"]]
    )

    # Orbit rings
    for r, name in [(LEO, "LEO Orbit"), (MEO, "MEO Orbit"), (GEO, "GEO Orbit")]:
        rx, ry, rz = _orbit_ring_xyz(r, tilt)
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz, mode="lines",
            line=dict(width=3, color=THEME["neon"]),
            name=name
        ))

    # UAVs
    scale = 0.7 * Rp
    ax, ay = area_xy
    px = [(u.pos[0] / ax - 0.5) * 2 * scale for u in uavs]
    py = [(u.pos[1] / ay - 0.5) * 2 * scale for u in uavs]
    pz = [0] * len(uavs)
    roles = [u.role for u in uavs]
    col_map = {"source": THEME["neon"], "relay": THEME["neon_mid"], "sink": THEME["neon_deep"]}
    cols = [col_map.get(r, THEME["neon_mid"]) for r in roles]
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz, mode="markers+text",
        marker=dict(size=6, color=cols),
        text=[f"UAV {u.uid}" for u in uavs],
        textfont=dict(color=THEME["txt"]),
        textposition="top center",
        name="UAVs"
    ))

    fig.update_scenes(
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
        bgcolor=THEME["bg"]
    )
    fig.update_layout(
        paper_bgcolor=THEME["bg"],
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color=THEME["txt"]))
    )
    return fig

# ---------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------
def export_data(metrics_df, uavs, params):
    csv_buf = io.StringIO()
    metrics_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    u_summary = [{"uid": u.uid, "role": u.role, "x_m": float(u.pos[0]), "y_m": float(u.pos[1]),
                  "energy_used_Wh": u.energy_used_Wh,
                  "battery_remaining_Wh": max(u.battery_Wh - u.energy_used_Wh, 0)} for u in uavs]
    json_bytes = json.dumps({"parameters": params, "metrics": metrics_df.to_dict(orient="records"),
                             "uavs": u_summary}, indent=2).encode("utf-8")
    return csv_bytes, json_bytes

def summarize_metrics(df):
    if df.empty:
        return {}
    return {"steps": int(df["t"].max()) + 1,
            "avg_thr": float(df["throughput_bps"].mean() / 1e6),
            "peak_thr": float(df["throughput_bps"].max() / 1e6),
            "final_batt": float(df["avg_remaining_battery_Wh"].iloc[-1]),
            "avg_risk": float(df["avg_eaves_risk_0to1"].mean())}

def build_pdf_report(params, df, imgs):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    y = H - 0.8 * 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.8 * 72, y, "Autonomous UAV Networks – Mission Report")
    y -= 0.3 * 72
    c.setFont("Helvetica", 10)
    c.drawString(0.8 * 72, y, "Autonomy • Routing • Energy • Security • Orbit Visualization")
    y -= 0.35 * 72
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.8 * 72, y, "Parameters")
    y -= 0.2 * 72
    c.setFont("Helvetica", 10)
    for k, v in params.items():
        c.drawString(0.9 * 72, y, f"• {k}: {v}"); y -= 0.15 * 72
    s = summarize_metrics(df)
    if s:
        y -= 0.1 * 72
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.8 * 72, y, "Quick Stats"); y -= 0.18 * 72
        c.setFont("Helvetica", 10)
        for k, v in s.items():
            c.drawString(0.9 * 72, y, f"• {k}: {v:.3f}" if isinstance(v, float) else f"• {k}: {v}")
            y -= 0.15 * 72
    if imgs.get("map"):    c.drawImage(ImageReader(io.BytesIO(imgs["map"])),    0.7 * 72, 3.8 * 72, 3.5 * 72, 3 * 72)
    if imgs.get("links"):  c.drawImage(ImageReader(io.BytesIO(imgs["links"])),  4.5 * 72, 3.8 * 72, 3.0 * 72, 3 * 72)
    c.showPage()
    if imgs.get("metrics"): c.drawImage(ImageReader(io.BytesIO(imgs["metrics"])), 0.8 * 72, 3.0 * 72, 7.0 * 72, 4.0 * 72)
    if imgs.get("orbits"):  c.drawImage(ImageReader(io.BytesIO(imgs["orbits"])),  1.0 * 72, 0.8 * 72, 6.0 * 72, 2.0 * 72)
    c.save()
    return buf.getvalue()

# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Autonomous UAV Networks Simulator", layout="wide")
inject_app_css()
st.title("🛰️ Autonomous UAV Networks Simulator")
st.caption("Based on Sarkar & Gul (2023) — Artificial Intelligence-Based Autonomous UAV Networks: A Survey")

with st.sidebar:
    st.header("Scenario")
    seed = st.number_input("Seed", 0, 999999, 42)
    area_x = st.slider("Area X (m)", 300, 2000, 1000, 50)
    area_y = st.slider("Area Y (m)", 300, 2000, 1000, 50)
    num = st.slider("UAVs", 3, 40, 12)
    srcs = st.slider("Sources", 1, 10, 3)
    sinks = st.slider("Sinks", 1, 10, 3)
    steps = st.slider("Steps", 5, 300, 120, 5)
    dt = st.slider("Δt (s)", 0.5, 5.0, 1.0, 0.5)

    st.header("Comm / MAC")
    mac = st.selectbox("MAC Scheme", ["TDMA (Orthogonal)", "NOMA (Superposition)", "Rate-Splitting (RSMA)"])
    link_thresh = st.slider("Link Capacity Threshold (bps)", 0.1, 10.0, 1.0, 0.1)

    st.header("Channel")
    f = st.slider("Carrier f (GHz)", 0.9, 6.0, 2.4, 0.1)
    pl0 = st.slider("PL(d0) dB @1m", 30, 60, 40, 1)
    n = st.slider("Pathloss exponent n", 1.6, 3.5, 2.2, 0.1)
    sh = st.slider("Shadowing σ (dB)", 0.0, 6.0, 2.0, 0.5)

    st.header("Adversaries")
    jam_on = st.checkbox("Enable Jammer", True)
    jam_x = st.slider("Jammer X", 0, area_x, int(0.35 * area_x), 10)
    jam_y = st.slider("Jammer Y", 0, area_y, int(0.65 * area_y), 10)
    jam_pow = st.slider("Jammer Power (W)", 0.1, 10.0, 2.0, 0.1)
    jam_r = st.slider("Jammer Radius (m)", 50, 600, 250, 10)
    eav_on = st.checkbox("Enable Eavesdropper", True)
    eav_x = st.slider("Eaves X", 0, area_x, int(0.65 * area_x), 10)
    eav_y = st.slider("Eaves Y", 0, area_y, int(0.35 * area_y), 10)
    eav_r = st.slider("Eaves Radius (m)", 50, 600, 250, 10)

    st.header("3D Orbit View")
    show_3d = st.checkbox("Enable 3D Orbit Scene", True)
    orbit_tilt = st.slider("Ring Tilt (deg)", -40, 40, 0, 1)
    planet_R = st.slider("Planet Radius (vis)", 200, 800, 400, 20)
    leo_r = st.slider("LEO Radius", 320, 900, 520, 10)
    meo_r = st.slider("MEO Radius", 500, 1200, 700, 10)
    geo_r = st.slider("GEO Radius", 700, 1600, 880, 10)
    sphere_alpha = st.slider("Sphere Opacity", 0.0, 1.0, 0.20, 0.05)

    run = st.button("Run Simulation", type="primary")

# =========================================================
# Run Simulation and Display Results
# =========================================================
if run:
    jammer_cfg = dict(enabled=jam_on, pos=[jam_x, jam_y], power_W=jam_pow, radius_m=jam_r)
    eaves_cfg = dict(enabled=eav_on, pos=[eav_x, eav_y], radius_m=eav_r)
    ch_params = dict(f_GHz=f, pl0_dB=pl0, n=n, shadowing_std_dB=sh)

    uavs, metrics, area_xy, jammer, eaves, ch = run_sim(
        seed, num, (area_x, area_y), steps, dt, srcs, sinks,
        mac, link_thresh, jammer_cfg, eaves_cfg, ch_params
    )

    # ---- Final positions (neon)
    roles = [u.role for u in uavs]
    role_color = {"source": THEME["neon"], "relay": THEME["neon_mid"], "sink": THEME["neon_deep"]}
    colors = [role_color.get(r, THEME["neon_mid"]) for r in roles]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[u.pos[0] for u in uavs], y=[u.pos[1] for u in uavs],
        mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(width=1, color=THEME["grid"])),
        text=[f"{u.uid}:{u.role[0].upper()}" for u in uavs],
        textposition="top center",
        name="UAVs"
    ))
    # Adversary zones
    def add_circle(center, radius, name):
        th = np.linspace(0, 2*np.pi, 200)
        cx = center[0] + radius*np.cos(th)
        cy = center[1] + radius*np.sin(th)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines",
                                 line=dict(color=THEME["neon_mid"], dash="dot"),
                                 name=name))
        fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode="markers",
                                 marker=dict(symbol="x", size=10, color=THEME["neon"]),
                                 name=f"{name} center"))
    if jammer: add_circle(jammer.pos, jammer.radius_m, "Jammer Zone")
    if eaves:  add_circle(eaves.pos, eaves.radius_m, "Eaves Zone")

    fig.update_layout(xaxis_range=[0, area_xy[0]], yaxis_range=[0, area_xy[1]])
    fig = greenify(fig, "Final UAV Positions")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Connectivity
    G = build_graph(uavs, ch, jammer, 1e-9, link_thresh, mac)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        up = [U for U in uavs if U.uid == u][0].pos
        vp = [U for U in uavs if U.uid == v][0].pos
        edge_x += [up[0], vp[0], None]
        edge_y += [up[1], vp[1], None]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(color=THEME["link"], width=2), opacity=0.45, name="Links"))
    fig2.add_trace(go.Scatter(x=[u.pos[0] for u in uavs], y=[u.pos[1] for u in uavs],
                              mode="markers", marker=dict(size=9, color=colors,
                              line=dict(width=1, color=THEME["grid"])), name="UAVs"))
    fig2.update_layout(xaxis_range=[0, area_xy[0]], yaxis_range=[0, area_xy[1]])
    fig2 = greenify(fig2, "Connectivity Graph")
    st.plotly_chart(fig2, use_container_width=True)

    # ---- 3D Orbit
    if show_3d:
        fig3d = make_orbit_figure(
            uavs=uavs, area_xy=area_xy,
            Rp=planet_R, LEO=leo_r, MEO=meo_r, GEO=geo_r,
            tilt=orbit_tilt, alpha=sphere_alpha
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ---- Metrics
    metrics["throughput_Mbps"] = metrics["throughput_bps"] / 1e6
    figm = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Throughput (Mb/s)", "Avg Battery (Wh)", "Eaves Risk (0–1)"))
    figm.add_trace(go.Scatter(x=metrics["t"], y=metrics["throughput_Mbps"], mode="lines",
                              line=dict(width=2, color=THEME["neon"])), 1, 1)
    figm.add_trace(go.Scatter(x=metrics["t"], y=metrics["avg_remaining_battery_Wh"], mode="lines",
                              line=dict(width=2, color=THEME["neon_mid"])), 2, 1)
    figm.add_trace(go.Scatter(x=metrics["t"], y=metrics["avg_eaves_risk_0to1"], mode="lines",
                              line=dict(width=2, color=THEME["neon_deep"])), 3, 1)
    figm = greenify(figm, "Analytics")
    st.plotly_chart(figm, use_container_width=True)

    # ---- Export Section
    st.subheader("Export Results")
    params_dict = {
        "num_uavs": num, "sources": srcs, "sinks": sinks,
        "steps": steps, "dt": dt, "MAC_scheme": mac,
        "jammer_enabled": jam_on, "eavesdropper_enabled": eav_on,
        "area_m": (area_x, area_y), "pathloss_exponent": n,
        "shadowing_std_dB": sh, "frequency_GHz": f
    }
    csv_bytes, json_bytes = export_data(metrics, uavs, params_dict)
    st.download_button("📄 Download CSV", csv_bytes, "uav_metrics.csv", "text/csv")
    st.download_button("🧠 Download JSON", json_bytes, "uav_full_export.json", "application/json")

    # ---- PDF Report (requires kaleido installed for to_image())
    try:
        map_png = fig.to_image(format="png", scale=2)
        links_png = fig2.to_image(format="png", scale=2)
        metrics_png = figm.to_image(format="png", scale=2)
        orbits_png = None
        if show_3d:
            orbits_png = fig3d.to_image(format="png", scale=2)
        pdf_bytes = build_pdf_report(params_dict, metrics,
                                     {"map": map_png, "links": links_png,
                                      "metrics": metrics_png, "orbits": orbits_png})
        st.download_button("🗂️ Download PDF Mission Report", pdf_bytes,
                           "uav_mission_report.pdf", "application/pdf")
    except Exception as e:
        st.warning(f"PDF export unavailable: {e} (check kaleido install)")

    st.success("✅ Simulation complete — tweak sliders and rerun!")
else:
    st.info("Set parameters in the sidebar and click **Run Simulation**.")
