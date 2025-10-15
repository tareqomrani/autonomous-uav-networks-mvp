# ================================================================
#  Autonomous UAV Networks Simulator (Aerospace-Accurate Edition)
#  app.py
# ================================================================

import math, random
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from dataclasses import dataclass, field
from typing import List
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from plotly.subplots import make_subplots

# ================================================================
#  CONSTANTS & UTILITIES
# ================================================================
K_B = 1.380649e-23   # Boltzmann (J/K)
T0_K = 290.0         # Standard temperature (K)

def db_to_lin(x_db): return 10.0 ** (x_db / 10.0)
def lin_to_db(x):   return 10.0 * math.log10(max(x, 1e-30))

# ================================================================
#  NEON GREEN THEME
# ================================================================
THEME = {
    "bg": "#0a0f0a",
    "panel": "#0c1310",
    "grid": "#123a28",
    "txt": "#d8ffe9",
    "neon": "#22ff88",
    "neon_mid": "#17d473",
    "neon_deep": "#0fa25f",
    "link": "#1aff80",
}

def inject_app_css():
    st.markdown(f"""
    <style>
      .stApp {{ background:{THEME['bg']}; color:{THEME['txt']}; }}
      [data-testid="stSidebar"] {{ background:{THEME['panel']}; color:{THEME['txt']}; }}
      h1, h2, h3, h4, h5, h6 {{ color:{THEME['txt']} !important; }}
      .stButton button, .stDownloadButton button {{
        border:1px solid {THEME['neon_mid']}; background:transparent; color:{THEME['txt']};
      }}
      .stButton button:hover, .stDownloadButton button:hover {{
        box-shadow:0 0 14px {THEME['neon']}; border-color:{THEME['neon']};
      }}
    </style>
    """, unsafe_allow_html=True)

def greenify(fig, title=None):
    fig.update_layout(
        title=title,
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["txt"]),
        legend=dict(font=dict(color=THEME["txt"])),
        colorway=[THEME["neon"], THEME["neon_mid"], THEME["neon_deep"]],
        margin=dict(l=10, r=10, t=40 if title else 10, b=10)
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["grid"], linecolor=THEME["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=THEME["grid"], linecolor=THEME["grid"])
    return fig

# ================================================================
#  AEROSPACE MODELS
# ================================================================
@dataclass
class UAV:
    uid: int
    pos: np.ndarray
    role: str
    energy_used_Wh: float = 0.0
    battery_Wh: float = 150.0
    def move(self, delta: np.ndarray): self.pos += delta

@dataclass
class Jammer:
    pos: np.ndarray
    power_W: float = 5.0
    kind: str = "jammer"

@dataclass
class Eavesdropper:
    pos: np.ndarray
    radius_m: float = 300.0
    kind: str = "eaves"

# ---- Channel Model (FSPL + Shadowing) ----
@dataclass
class ChannelModel:
    f_GHz: float = 2.4
    shadowing_std_dB: float = 2.0
    n_extra: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    def pathloss_dB(self, d_m: float, shadow=True):
        d_km = max(d_m, 1e-3) / 1000.0
        f_MHz = self.f_GHz * 1e3
        fspl = 32.44 + 20.0 * math.log10(f_MHz) + 20.0 * math.log10(d_km)
        extra = 10.0 * self.n_extra * math.log10(max(d_m, 1e-3))
        pl = fspl + extra
        if shadow:
            pl += self.rng.gauss(0.0, self.shadowing_std_dB)
        return pl

    def pathgain_linear(self, d_m: float, shadow=True):
        return 10.0 ** (-self.pathloss_dB(d_m, shadow) / 10.0)

# ---- Capacity ----
def capacity_bps(EIRP_W: float, G_r_lin: float, path_gain: float,
                 B_Hz: float, NF_dB: float):
    N_W = K_B * T0_K * B_Hz * db_to_lin(NF_dB)
    rx_W = EIRP_W * path_gain * G_r_lin
    sinr = rx_W / max(N_W, 1e-30)
    return B_Hz * math.log2(1.0 + sinr)

def mac_share(N: int, scheme: str):
    if "TDMA" in scheme: return 1.0 / max(N, 1)
    if "NOMA" in scheme: return 1.0
    if "RSMA" in scheme: return 0.8
    return 1.0 / max(N, 1)

# ---- Propulsion Power ----
def propulsion_energy_Wh(speed_mps: float, dt_s: float, P_base_W: float, P_speed_coeff: float):
    P = max(P_base_W + P_speed_coeff * (speed_mps ** 3), 0.0)
    return P * dt_s / 3600.0

# ================================================================
#  GRAPH BUILDING
# ================================================================
def build_graph(uavs, ch, jammer, B_Hz, NF_dB, link_thresh_bps, mac_scheme,
                tx_power_W, tx_gain_dBi, rx_gain_dBi):
    G = nx.DiGraph()
    for u in uavs:
        G.add_node(u.uid, pos=(u.pos[0], u.pos[1]), role=u.role)

    positions = np.vstack([u.pos for u in uavs])
    dists = cdist(positions, positions)
    N = len(uavs)

    EIRP_W = tx_power_W * db_to_lin(tx_gain_dBi)
    G_r_lin = db_to_lin(rx_gain_dBi)

    for i in range(N):
        for j in range(N):
            if i == j: continue
            pg = ch.pathgain_linear(dists[i, j])
            N_W = K_B * T0_K * B_Hz * db_to_lin(NF_dB)
            if jammer:
                dJ = np.linalg.norm(uavs[j].pos - jammer.pos)
                J_pg = ch.pathgain_linear(dJ, shadow=False)
                N_W += jammer.power_W * J_pg
            mac = mac_share(N - 1, mac_scheme)
            cap = mac * capacity_bps(EIRP_W, G_r_lin, pg, B_Hz, NF_dB)
            if cap >= link_thresh_bps:
                G.add_edge(uavs[i].uid, uavs[j].uid, capacity_bps=cap)
    return G

# ================================================================
#  3D ORBIT VIEW
# ================================================================
def _sphere_mesh(r):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    return x, y, z

def _orbit_ring_xyz(r, tilt_deg=0):
    th = np.linspace(0, 2 * np.pi, 200)
    x = r * np.cos(th)
    y = r * np.sin(th)
    z = r * np.sin(np.radians(tilt_deg)) * np.ones_like(th)
    return x, y, z

def make_orbit_figure(uavs, area_xy, Rp=400, LEO=520, MEO=700, GEO=880,
                      tilt=0, alpha=0.18):
    xs, ys, zs = _sphere_mesh(Rp)
    fig = go.Figure()

    fig.add_surface(
        x=xs, y=ys, z=zs, opacity=alpha,
        colorscale=[[0, "rgb(5,40,25)"], [1, "rgb(10,70,45)"]],
        showscale=False
    )

    for r, name in [(LEO, "LEO"), (MEO, "MEO"), (GEO, "GEO")]:
        rx, ry, rz = _orbit_ring_xyz(r, tilt)
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz, mode="lines",
            line=dict(width=3, color=THEME["neon"]),
            name=f"{name} Orbit"
        ))

    ax, ay = area_xy
    scale = 0.7 * Rp
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
        textposition="top center"
    ))
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor=THEME["bg"])
    fig.update_layout(paper_bgcolor=THEME["bg"], margin=dict(l=0, r=0, t=0, b=0))
    return fig

# ================================================================
#  STREAMLIT FRONT-END
# ================================================================
st.set_page_config("Autonomous UAV Networks Simulator", layout="wide")
inject_app_css()
st.title("🛰️ Autonomous UAV Networks Simulator")
st.caption("High-Fidelity Aerospace Network Model — based on Sarkar & Gul (2023)")

# ---- Sidebar Inputs ----
st.sidebar.header("Scenario Setup")
num_uavs = st.sidebar.slider("Number of UAVs", 6, 20, 12)
sources = st.sidebar.slider("Sources", 1, 5, 3)
sinks = st.sidebar.slider("Sinks", 1, 5, 3)
steps = st.sidebar.slider("Simulation Steps", 10, 200, 120, 10)
dt = st.sidebar.number_input("Timestep (s)", 0.1, 5.0, 1.0)
area_xy = np.array([
    st.sidebar.slider("Area X (m)", 500, 3000, 1000, 100),
    st.sidebar.slider("Area Y (m)", 500, 3000, 2000, 100)
])

st.sidebar.header("Radio / RF")
bandwidth_MHz = st.sidebar.slider("Channel bandwidth (MHz)", 1.0, 20.0, 5.0, 0.5)
noise_fig_dB = st.sidebar.slider("Receiver noise figure (dB)", 2.0, 10.0, 6.0, 0.5)
tx_power_W = st.sidebar.slider("TX power (W)", 0.1, 10.0, 2.0, 0.1)
tx_gain_dBi = st.sidebar.slider("TX antenna gain (dBi)", 0.0, 15.0, 5.0, 0.5)
rx_gain_dBi = st.sidebar.slider("RX antenna gain (dBi)", 0.0, 15.0, 5.0, 0.5)
B_Hz = bandwidth_MHz * 1e6

st.sidebar.header("Channel & Link")
f_GHz = st.sidebar.slider("Frequency (GHz)", 1.0, 10.0, 4.0, 0.1)
pathloss_extra = st.sidebar.slider("Extra exponent (air clutter)", 0.0, 1.0, 0.2, 0.1)
shadowing_std = st.sidebar.slider("Shadowing σ (dB)", 0.0, 6.0, 2.0, 0.5)
link_thresh = st.sidebar.slider("Min link capacity (bps)", 1e4, 5e7, 1e6, step=1e4)
mac = st.sidebar.selectbox("MAC scheme", ["TDMA (Orthogonal)", "NOMA (Non-Orthogonal)", "RSMA (Rate-Splitting)"])
jammer_enabled = st.sidebar.checkbox("Enable jammer", True)
eaves_enabled = st.sidebar.checkbox("Enable eavesdropper", True)

st.sidebar.header("Propulsion")
P_base_W = st.sidebar.slider("Propulsion power @ cruise (W)", 40, 200, 90, 5)
P_speed_coeff = st.sidebar.slider("Speed power coeff (W/(m/s)^3)", 0.0, 0.2, 0.02, 0.01)

show_3d = st.sidebar.checkbox("Show 3D orbit", True)

# ================================================================
#  SIMULATION
# ================================================================
roles = ["source"] * sources + ["sink"] * sinks + ["relay"] * (num_uavs - sources - sinks)
uavs = [UAV(i, np.random.rand(2) * area_xy, roles[i]) for i in range(num_uavs)]

ch = ChannelModel(f_GHz=f_GHz, shadowing_std_dB=shadowing_std, n_extra=pathloss_extra)
jammer = Jammer(np.random.rand(2) * area_xy) if jammer_enabled else None
eaves = Eavesdropper(np.random.rand(2) * area_xy) if eaves_enabled else None

metrics = []
for t in range(steps):
    G = build_graph(uavs, ch, jammer, B_Hz, noise_fig_dB, link_thresh, mac,
                    tx_power_W, tx_gain_dBi, rx_gain_dBi)

    throughput = 0.0
    route_risk = []
    for s in [u.uid for u in uavs if u.role == "source"]:
        for k in [u.uid for u in uavs if u.role == "sink"]:
            try:
                path = nx.shortest_path(G, source=s, target=k)
                caps = [G.edges[path[i], path[i+1]]["capacity_bps"] for i in range(len(path)-1)]
                throughput += min(caps)
                if eaves:
                    dist = np.mean([np.linalg.norm(uavs[n].pos - eaves.pos) for n in path])
                    route_risk.append(float(np.clip(1.0 - dist / eaves.radius_m, 0.0, 1.0)))
            except nx.NetworkXNoPath:
                continue

    total_risk = float(np.mean(route_risk)) if route_risk else 0.0
    if not route_risk and eaves:
        d = [np.linalg.norm(u.pos - eaves.pos) for u in uavs]
        prox = [float(np.clip(1.0 - di / eaves.radius_m, 0.0, 1.0)) for di in d]
        total_risk = float(np.mean(prox))

    # Update energy
    for u in uavs:
        pre = u.pos.copy()
        u.move((np.random.rand(2)-0.5)*10)
        dist = np.linalg.norm(u.pos - pre)
        speed = dist / max(dt, 1e-9)
        u.energy_used_Wh += propulsion_energy_Wh(speed, dt, P_base_W, P_speed_coeff)
        u.battery_Wh = max(0.0, 150.0 - u.energy_used_Wh)

    metrics.append({
        "t": t,
        "throughput_bps": throughput,
        "throughput_Mbps": throughput / 1e6,
        "avg_eaves_risk_0to1": total_risk,
        "avg_remaining_battery_Wh": np.mean([u.battery_Wh for u in uavs]),
    })

df = pd.DataFrame(metrics)

# ================================================================
#  VISUALIZATION
# ================================================================
role_color = {"source": THEME["neon"], "relay": THEME["neon_mid"], "sink": THEME["neon_deep"]}
colors = [role_color.get(u.role, THEME["neon_mid"]) for u in uavs]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[u.pos[0] for u in uavs], y=[u.pos[1] for u in uavs],
    mode="markers+text",
    marker=dict(size=12, color=colors, line=dict(width=1, color=THEME["grid"])),
    text=[f"{u.uid}:{u.role[0].upper()}" for u in uavs],
    textposition="top center"))
fig = greenify(fig, "Final UAV Positions")
st.plotly_chart(fig, use_container_width=True)

edge_x, edge_y = [], []
for u, v in G.edges():
    up = [U for U in uavs if U.uid == u][0].pos
    vp = [U for U in uavs if U.uid == v][0].pos
    edge_x += [up[0], vp[0], None]
    edge_y += [up[1], vp[1], None]
fig2 = go.Figure()
fig2.add_trace(go.Sc
