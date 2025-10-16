# ================================================================
#  Aerospace UAV Networks Simulator v2.3 – Digital-Green (Consolidated)
#  ISA + Aerodynamics + Two-Ray + Rician + LTE-MCS + MAC presets + Eaves Risk
# ================================================================
import math, random
import numpy as np, pandas as pd, networkx as nx, streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass

# ---------------- Constants & utils ----------------
K_B=1.380649e-23; G0=9.80665; C=2.99792458e8; PI=math.pi; EPS0=8.854187817e-12
def db_to_lin(x): return 10**(x/10)
def lin_to_db(x): return 10*math.log10(max(x,1e-30))

THEME={"bg":"#0a0f0a","panel":"#0c1310","grid":"#123a28","txt":"#d8ffe9",
       "neon":"#22ff88","neon_mid":"#17d473","neon_deep":"#0fa25f"}

def inject_css():
    st.markdown(f"""
    <style>
      .stApp {{ background:{THEME['bg']}; color:{THEME['txt']}; }}
      [data-testid="stSidebar"] {{ background:{THEME['panel']}; color:{THEME['txt']}; }}
      .stButton button, .stDownloadButton button {{
        border:1px solid {THEME['neon_mid']}; background:transparent; color:{THEME['txt']}; }}
      .stButton button:hover, .stDownloadButton button:hover {{
        box-shadow:0 0 14px {THEME['neon']}; border-color:{THEME['neon']}; }}
      h1,h2,h3,h4,h5,h6 {{ color:{THEME['txt']} !important; }}
    </style>
    """, unsafe_allow_html=True)

def greenify(fig, title=None):
    fig.update_layout(
        title=title, paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["txt"]),
        legend=dict(font=dict(color=THEME["txt"])),
        colorway=[THEME["neon"], THEME["neon_mid"], THEME["neon_deep"]],
        margin=dict(l=10,r=10,t=40 if title else 10,b=10))
    fig.update_xaxes(showgrid=True, gridcolor=THEME["grid"], linecolor=THEME["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=THEME["grid"], linecolor=THEME["grid"])
    return fig

# ---------------- ISA atmosphere ----------------
def isa_density(h):
    T0, p0, L, R, g = 288.15, 101325.0, -0.0065, 287.058, 9.80665
    T = T0 + L*h
    p = p0 * (T/T0)**(-g/(R*L))
    return p/(R*T)

def isa_temp(h): return 288.15 - 0.0065*h

# ---------------- Aerodynamics & battery ----------------
@dataclass
class AeroConfig:
    mass_kg:float=11.0; S:float=0.6; AR:float=10.0; CD0:float=0.025; e:float=0.85
    eta_prop:float=0.72; motor_eff:float=0.92; battery_Wh:float=360.0
    v_cruise:float=18.0; v_max:float=30.0; v_loiter:float=11.0
    climb_angle_deg_max:float=12.0; turn_rate_deg_s_max:float=20.0
    @property
    def W_N(self): return self.mass_kg*G0

def power_required(V,h,gamma,c):
    rho=isa_density(h); W=c.W_N
    induced=(2*W**2)/(PI*c.e*c.AR*rho*c.S)/max(V,0.5)
    parasite=0.5*rho*c.S*c.CD0*V**3
    climb=W*V*math.sin(gamma)
    return induced+parasite+climb

def electrical_power(P,c): return P/max(c.eta_prop*c.motor_eff,0.05)

@dataclass
class Battery:
    energy_Wh:float
    def draw(self,P,dt): dWh=min(P*dt/3600,self.energy_Wh); self.energy_Wh-=dWh; return dWh
    @property
    def soc(self): return max(self.energy_Wh/360.0,0)

# ---------------- Vehicle kinematics ----------------
def stall_speed(h,c): rho=isa_density(h); CL=1.2; return math.sqrt(2*c.W_N/(rho*c.S*CL))

@dataclass
class UAV:
    uid:int; x:float; y:float; h:float; hdg:float; V:float; role:str
    energy_Wh:float; bat:Battery; cfg:AeroConfig
    def p3d(self): return np.array([self.x,self.y,self.h])

def update_state(u,dt,cmd_hdg,cmd_V,cmd_gam,wind,area):
    max_turn=math.radians(u.cfg.turn_rate_deg_s_max)*dt
    dpsi=(cmd_hdg-u.hdg+PI)%(2*PI)-PI
    dpsi=max(-max_turn,min(max_turn,dpsi))
    u.hdg=(u.hdg+dpsi)%(2*PI)
    V=min(max(cmd_V,stall_speed(u.h,u.cfg)*1.05),u.cfg.v_max); u.V=V
    gam=max(-math.radians(u.cfg.climb_angle_deg_max),
            min(math.radians(u.cfg.climb_angle_deg_max),cmd_gam))
    v_air=np.array([V*math.cos(u.hdg), V*math.sin(u.hdg)])
    v_g=v_air+wind[:2]
    dx,dy=v_g*dt; dh=V*math.sin(gam)*dt
    u.x=max(0,min(u.x+dx,area[0])); u.y=max(0,min(u.y+dy,area[1])); u.h=max(0,u.h+dh)

# ---------------- RF: FSPL, Two-Ray, Rician ----------------
def fspl_dB(f_GHz, d_m): return 32.44 + 20*math.log10(f_GHz*1e3) + 20*math.log10(max(d_m,1e-3)/1000)
def fspl_pg(f_GHz, d_m, sigma_dB=0): return 10**(-(fspl_dB(f_GHz,d_m)+np.random.normal(0,sigma_dB))/10)

def fresnel(eps_r, sigma, f_Hz, theta, pol):
    omega=2*PI*f_Hz; eps_c=eps_r-1j*sigma/(omega*EPS0); s,c=math.sin(theta),math.cos(theta)
    term=np.sqrt(eps_c-s**2+0j)
    if pol.startswith("para"): num,den=eps_c*c-term,eps_c*c+term
    else: num,den=c-term,c+term
    return num/den

def two_ray_pg(f_GHz, tx, rx, ht, hr, eps_r, sigma, pol, shadow_dB=0):
    f_Hz=f_GHz*1e9; k=2*PI*f_Hz/C
    d1=np.linalg.norm(tx-rx); tx_img=np.array([tx[0],tx[1],-ht]); d2=np.linalg.norm(tx_img-rx)
    horiz=np.linalg.norm(tx[:2]-rx[:2]); theta=math.atan2(abs(ht+hr),max(horiz,1e-6))
    G=fresnel(eps_r,sigma,f_Hz,theta,pol)
    E1=np.exp(-1j*k*d1)/max(d1,1e-6); E2=G*np.exp(-1j*k*d2)/max(d2,1e-6)
    Etot=E1+E2
    fspl_lin=10**(-fspl_dB(f_GHz,d1)/10)
    field_ratio=min((abs(Etot)*d1)**2,10.0)  # cap at +10 dB above FSPL
    return fspl_lin*field_ratio*db_to_lin(np.random.normal(0,shadow_dB))

def apply_rician(pg,K_dB):
    K=db_to_lin(K_dB); s=math.sqrt(K/(K+1)); sigma=math.sqrt(1/(2*(K+1)))
    g=(s+np.random.normal(0,sigma))+1j*np.random.normal(0,sigma)
    return pg*(abs(g)**2)

# ---------------- LTE MCS table ----------------
MCS_TABLE=pd.DataFrame([
    {"MCS":"QPSK 1/2","SINR_dB":-3,"Eff":0.377},
    {"MCS":"QPSK 3/4","SINR_dB": 3,"Eff":0.877},
    {"MCS":"16QAM 1/2","SINR_dB": 8,"Eff":1.476},
    {"MCS":"16QAM 3/4","SINR_dB":12,"Eff":2.406},
    {"MCS":"64QAM 2/3","SINR_dB":17,"Eff":3.672},
    {"MCS":"64QAM 5/6","SINR_dB":23,"Eff":4.877},
    {"MCS":"256QAM 3/4","SINR_dB":28,"Eff":6.377},
    {"MCS":"256QAM 5/6","SINR_dB":33,"Eff":7.377},
])
def mcs_eff(sinr_dB):
    x=MCS_TABLE["SINR_dB"].to_numpy(); y=MCS_TABLE["Eff"].to_numpy()
    return float(np.interp(sinr_dB, x, y, left=y[0], right=y[-1]))

# ---------------- MAC presets ----------------
def mac_share(num_neighbors, scheme):
    if "TDMA" in scheme:  # orthogonal time slices
        return 1.0 / max(num_neighbors, 1)
    if "NOMA" in scheme:  # power-domain multiplexing
        return 1.0
    if "RSMA" in scheme:  # rate-splitting overhead
        return 0.8
    return 1.0 / max(num_neighbors, 1)

# ---------------- Graph build + flow ----------------
def build_graph(U, ch, txP, txG, rxG, jammer, use_mcs):
    G=nx.DiGraph(); EIRP=txP*db_to_lin(txG); Gr=db_to_lin(rxG)
    for t in U:
        for r in U:
            if t.uid==r.uid: continue
            if ch["model"]=="2R":
                pg=two_ray_pg(ch["f"],t.p3d(),r.p3d(),t.h,r.h,ch["eps"],ch["sig"],ch["pol"],ch["shadow"])
            else:
                d=np.linalg.norm(t.p3d()-r.p3d()); pg=fspl_pg(ch["f"],d,ch["shadow"])
            if ch["rician"] is not None: pg=apply_rician(pg,ch["rician"])
            T=isa_temp(r.h); N=K_B*T*ch["B"]*db_to_lin(ch["NF"])
            rx_W=EIRP*pg*Gr
            I=0.0
            if jammer is not None:
                I=jammer["P"]*fspl_pg(ch["f"], np.linalg.norm(jammer["p"]-r.p3d()))
            sinr=rx_W/max(N+I,1e-30); sinr_dB=10*math.log10(max(sinr,1e-12))
            # MAC preset (rough neighbor count)
            macK = mac_share(max(len(U)-1,1), ch.get("mac_scheme","TDMA (Orthogonal)"))
            if use_mcs: C=ch["B"]*mcs_eff(sinr_dB)*macK
            else:       C=ch["B"]*math.log2(1+sinr)*macK*0.8
            if C>1e3: G.add_edge(t.uid,r.uid,cap=C)
    return G

def widest(G,res,s,t):
    import heapq
    best={n:0 for n in G}; best[s]=1e9; pq=[(-1e9,s)]; prev={}
    while pq:
        b,u=heapq.heappop(pq); b=-b
        if u==t: break
        for v in G[u]:
            c=res.get((u,v),0)
            if c>0:
                nb=min(b,c)
                if nb>best[v]:
                    best[v]=nb; prev[v]=u; heapq.heappush(pq,(-nb,v))
    if t not in prev: return None,0
    path=[t]
    while path[-1]!=s: path.append(prev[path[-1]])
    path.reverse(); return path,best[t]

def flow(G,S,T):
    res={(u,v):G[u][v]["cap"] for u,v in G.edges}; tot=0
    while True:
        best=(None,None,0); best_path=None
        for s in S:
            for t in T:
                p,c=widest(G,res,s,t)
                if p and c>best[2]: best=((s,t),p,c); best_path=p
        if best[0] is None: break
        _,p,c=best; tot+=c
        for i in range(len(p)-1): e=(p[i],p[i+1]); res[e]=max(0,res[e]-c)
    return tot,None

# ---------------- Visuals ----------------
def metrics_figure(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.t,y=df.thr,mode='lines',name='Throughput (Mbps)'))
    fig.add_trace(go.Scatter(x=df.t,y=df.bat,mode='lines',name='Battery (Wh)'))
    fig.add_trace(go.Scatter(x=df.t,y=df.SoC,mode='lines',name='SoC (frac)'))
    if "avg_eaves_risk_0to1" in df.columns:
        fig.add_trace(go.Scatter(x=df.t, y=df["avg_eaves_risk_0to1"], mode='lines', name='Eaves Risk (0..1)', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Risk (0..1)', range=[0,1]))
    fig.update_xaxes(title="Time step"); fig.update_yaxes(title="Metrics")
    return greenify(fig,"Mission Metrics")

def animation_figure(traj_df, area, U):
    cmap={"source":THEME["neon"],"relay":THEME["neon_mid"],"sink":THEME["neon_deep"]}
    role={u.uid:u.role for u in U}; tvals=sorted(traj_df["t"].unique())
    if not tvals: return go.Figure()
    base=traj_df[traj_df.t==tvals[0]]
    fig=go.Figure(
        data=[go.Scatter(x=base.x,y=base.y,mode="markers",
                         marker=dict(size=8,color=[cmap[role[int(uid)]] for uid in base.uid]))],
        layout=go.Layout(xaxis=dict(range=[0,area[0]]),yaxis=dict(range=[0,area[1]]),
                         updatemenus=[dict(type="buttons",buttons=[
                             dict(label="Play",method="animate",args=[None,{"frame":{"duration":50,"redraw":True},"fromcurrent":True}]),
                             dict(label="Pause",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False}}])
                         ])]))
    frames=[]
    for t in tvals:
        d=traj_df[traj_df.t==t]
        frames.append(go.Frame(
            data=[go.Scatter(x=d.x,y=d.y,mode="markers",
                             marker=dict(size=8,color=[cmap[role[int(uid)]] for uid in d.uid]))],
            name=str(t)))
    fig.frames=frames
    return greenify(fig,"Flight Path Animation")

def orbit_figure(U, area):
    Rp=400
    u,v=np.mgrid[0:2*PI:40j,0:PI:20j]
    x=Rp*np.cos(u)*np.sin(v); y=Rp*np.sin(u)*np.sin(v); z=Rp*np.cos(v)
    fig=go.Figure(); fig.add_surface(x=x,y=y,z=z,opacity=0.12,showscale=False,
                                     colorscale=[[0,"rgb(5,40,25)"],[1,"rgb(10,70,45)"]])
    for r,name in [(520,"LEO"),(700,"MEO"),(880,"GEO")]:
        th=np.linspace(0,2*PI,200)
        fig.add_trace(go.Scatter3d(x=r*np.cos(th),y=r*np.sin(th),z=np.zeros_like(th),
                                   mode="lines",line=dict(width=2,color=THEME["neon"]),name=name))
    px=[(uu.x/area[0]-0.5)*0.7*Rp*2 for uu in U]
    py=[(uu.y/area[1]-0.5)*0.7*Rp*2 for uu in U]
    fig.add_trace(go.Scatter3d(
        x=px,y=py,z=[0]*len(U),mode="markers+text",
        marker=dict(size=4,color=[THEME["neon"] if u.role=="source" else THEME["neon_mid"] if u.role=="relay" else THEME["neon_deep"] for u in U]),
        text=[f"U{uu.uid}" for uu in U],textfont=dict(color=THEME["txt"]),textposition="top center"))
    fig.update_scenes(xaxis_visible=False,yaxis_visible=False,zaxis_visible=False,bgcolor=THEME["bg"])
    fig.update_layout(paper_bgcolor=THEME["bg"],margin=dict(l=0,r=0,t=0,b=0))
    return fig

# ================================================================
#  STREAMLIT APP
# ================================================================
st.set_page_config("Aerospace UAV Networks v2.3", layout="wide")
inject_css()
st.title("🛰️ Aerospace-Grade UAV Networks Simulator v2.3 (Digital-Green)")

# Sidebar
a=AeroConfig()
area=np.array([st.sidebar.slider("Area X (m)",500,5000,1200),
               st.sidebar.slider("Area Y (m)",500,5000,2000)])
steps=st.sidebar.slider("Steps",10,400,120)
dt=st.sidebar.number_input("Δt (s)",0.1,5.0,0.5)
fGHz=st.sidebar.select_slider("Frequency (GHz)",[0.9,2.4,4.0,5.8],4.0)
B=st.sidebar.slider("Bandwidth (MHz)",0.5,20.0,5.0)*1e6
txP=st.sidebar.slider("TX Power (W)",0.1,10.0,2.0)
txG=st.sidebar.slider("TX Gain (dBi)",0.0,15.0,5.0)
rxG=st.sidebar.slider("RX Gain (dBi)",0.0,15.0,5.0)
NF=st.sidebar.slider("Noise Figure (dB)",2.0,10.0,6.0)
eps=st.sidebar.slider("εr",2.0,30.0,15.0)
sig=st.sidebar.slider("σ (S/m)",0.0,0.1,0.005)
rician=st.sidebar.slider("Rician K (dB)",-10.0,20.0,6.0)
shadow=st.sidebar.slider("Shadowing σ (dB)",0.0,6.0,2.0)
mac_scheme=st.sidebar.selectbox("MAC scheme", ["TDMA (Orthogonal)","NOMA (Non-Orthogonal)","RSMA (Rate-Splitting)"])
slots=st.sidebar.number_input("TDMA Slot (s)",0.1,5.0,1.0)
wind_spd=st.sidebar.slider("Wind (m/s)",0.0,20.0,3.0)
wind_dir=st.sidebar.slider("Wind Direction (° from)",0,360,270)
num=st.sidebar.slider("UAV Count",6,30,12)
src=st.sidebar.slider("Sources",1,5,3)
snk=st.sidebar.slider("Sinks",1,5,3)
use_mcs=st.sidebar.checkbox("Use LTE MCS model", False)
jam_en=st.sidebar.checkbox("Enable jammer", True)
eaves_en=st.sidebar.checkbox("Enable eavesdropper", False)
eaves_radius=st.sidebar.slider("Eavesdrop radius (m)", 50, 1000, 300, 10)
show3d=st.sidebar.checkbox("Show 3-D Orbit View", True)
show_anim=st.sidebar.checkbox("Show Flight-Path Animation", False)
export=st.sidebar.checkbox("Enable CSV Export", True)

# PHY indicator
phy_mode="📡 PHY: LTE-MCS" if use_mcs else "📡 PHY: Shannon Ideal"
st.markdown(f"### {phy_mode}")

# Channel & wind
ch=dict(f=fGHz,B=B,NF=NF,model="2R",eps=eps,sig=sig,pol="perp",
        shadow=shadow,rician=rician,mac_scheme=mac_scheme)
jammer=dict(p=np.array([np.random.rand()*area[0],np.random.rand()*area[1],0.0]), P=5.0) if jam_en else None
eaves=(np.array([np.random.rand()*area[0],np.random.rand()*area[1],0.0]) if eaves_en else None)
wd_rad=math.radians(wind_dir+180)  # “from” semantics
wind=np.array([wind_spd*math.cos(wd_rad), wind_spd*math.sin(wd_rad), 0.0])

# Fleet init
roles=["source"]*src + ["sink"]*snk + ["relay"]*(num-src-snk)
U=[UAV(i, np.random.rand()*area[0], np.random.rand()*area[1],
       50+np.random.rand()*20, np.random.rand()*2*PI, a.v_cruise,
       roles[i], a.battery_Wh, Battery(a.battery_Wh), a) for i in range(num)]

# Simulation loop
metrics=[]; traj=[]
for t in range(steps):
    G=build_graph(U,ch,txP,txG,rxG,jammer,use_mcs)
    S=[u.uid for u in U if u.role=="source"]; T=[u.uid for u in U if u.role=="sink"]
    thr,_=flow(G,S,T)
    # TDMA slot scaling per step
    slots_per_step=max(1,int(round(dt/max(slots,1e-9))))
    thr *= 1.0/slots_per_step

    # --- Eavesdropper risk (0..1 mean fleet proximity) ---
    avg_risk=0.0
    if eaves is not None:
        dists=[np.linalg.norm(u.p3d()-eaves) for u in U]
        avg_risk=float(np.mean([float(np.clip(1.0 - d/max(eaves_radius,1e-6), 0.0, 1.0)) for d in dists]))

    # Move & energy
    for u in U:
        cmd_hdg=(u.hdg+np.random.normal(0,0.05))%(2*PI)
        cmd_V=a.v_cruise+np.random.uniform(-2,2)
        cmd_gam=np.radians(np.random.uniform(-a.climb_angle_deg_max,a.climb_angle_deg_max))
        update_state(u,dt,cmd_hdg,cmd_V,cmd_gam,wind,area)
        Pm=power_required(u.V,u.h,cmd_gam,u.cfg); u.bat.draw(electrical_power(Pm,u.cfg),dt)
        u.energy_Wh=u.bat.energy_Wh
        traj.append({"t":t,"uid":u.uid,"x":u.x,"y":u.y,"h":u.h})

    metrics.append(dict(t=t, thr=thr/1e6,
                        bat=np.mean([u.bat.energy_Wh for u in U]),
                        SoC=np.mean([u.bat.soc for u in U]),
                        avg_eaves_risk_0to1=avg_risk))

df=pd.DataFrame(metrics); traj_df=pd.DataFrame(traj)

# Plots
fig_pos=go.Figure()
fig_pos.add_trace(go.Scatter(
    x=[u.x for u in U], y=[u.y for u in U], mode="markers+text",
    marker=dict(size=12, color=[THEME["neon"] if u.role=="source"
                                else THEME["neon_mid"] if u.role=="relay"
                                else THEME["neon_deep"] for u in U]),
    text=[f"{u.uid}:{u.role[0].upper()}" for u in U], textposition="top center"))
st.plotly_chart(greenify(fig_pos,"Final UAV Positions"), use_container_width=True)

st.plotly_chart(metrics_figure(df), use_container_width=True)
if show3d: st.plotly_chart(orbit_figure(U, area), use_container_width=True)
if show_anim and not traj_df.empty: st.plotly_chart(animation_figure(traj_df, area, U), use_container_width=True)

# Exports
if export:
    st.subheader("Exports")
    st.download_button("📥 Download Metrics CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="uav_metrics_v2_3.csv", mime="text/csv")
    fleet_rows=[{"uid":u.uid,"role":u.role,"x":u.x,"y":u.y,"h":u.h,
                 "mass_kg":u.cfg.mass_kg,"S":u.cfg.S,"AR":u.cfg.AR,"CD0":u.cfg.CD0,"e":u.cfg.e,
                 "eta_prop":u.cfg.eta_prop,"motor_eff":u.cfg.motor_eff,"battery_Wh":u.cfg.battery_Wh,
                 "v_cruise":u.cfg.v_cruise,"v_max":u.cfg.v_max,"v_loiter":u.cfg.v_loiter} for u in U]
    st.download_button("📥 Download Fleet CSV", pd.DataFrame(fleet_rows).to_csv(index=False).encode("utf-8"),
                       file_name="uav_fleet_v2_3.csv", mime="text/csv")
    st.download_button("📥 Download Trajectory CSV", traj_df.to_csv(index=False).encode("utf-8"),
                       file_name="uav_trajectory_v2_3.csv", mime="text/csv")

# LTE MCS table reference
with st.sidebar.expander("📘 PHY Reference (LTE MCS Table)"):
    st.markdown("""
| MCS | Modulation | Coding | SINR (dB) | Efficiency (bps/Hz) |
|:----|:-----------|:-------|:----------|:---------------------|
| QPSK 1/2 | QPSK | 1/2 | -3 | 0.377 |
| QPSK 3/4 | QPSK | 3/4 | 3 | 0.877 |
| 16QAM 1/2 | 16QAM | 1/2 | 8 | 1.476 |
| 16QAM 3/4 | 16QAM | 3/4 | 12 | 2.406 |
| 64QAM 2/3 | 64QAM | 2/3 | 17 | 3.672 |
| 64QAM 5/6 | 64QAM | 5/6 | 23 | 4.877 |
| 256QAM 3/4 | 256QAM | 3/4 | 28 | 6.377 |
| 256QAM 5/6 | 256QAM | 5/6 | 33 | 7.377 |
""")
