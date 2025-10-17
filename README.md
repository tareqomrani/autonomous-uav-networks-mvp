# 🚀 Autonomous UAV Networks – Digital-Green v2.3  
**Aerospace-Accurate Multi-UAV Communication & Networking Simulator**

---

## 🧭 Overview  
This simulator models multi-UAV networks with integrated aerodynamic, atmospheric, and RF models.  
It is inspired by and aligns with themes from the survey article _“Artificial Intelligence-Based Autonomous UAV Networks: A Survey”_ by Sarkar & Gul (2023) 📖  [oai_citation:0‡MDPI](https://www.mdpi.com/2504-446X/7/5/322)  
The goal is to operationalize key concepts in autonomous UAV communications, energy modeling, and network security.

The app fuses real-world atmospheric physics (ISA), aerodynamic power models, Two-Ray + Rician propagation, and LTE MCS/MAC layers to explore secure, efficient mission-level UAV networking scenarios.

---

## ⚙️ Features at a Glance  

| Module | Description |
|-------|-------------|
| **ISA Atmosphere** | Computes density and temperature vs. altitude using the International Standard Atmosphere. |
| **Aerodynamic Power Model** | Splits propulsion demand into induced, parasite, and climb components. |
| **Battery & Efficiency** | Models power draw, conversion losses (propeller, motor), and SoC tracking. |
| **RF Propagation (2-Ray + Rician)** | Models direct + ground-reflected paths with fading and shadowing. |
| **LTE MCS / Shannon PHY** | Supports either Shannon ideal model or empirical LTE MCS interpolation. |
| **MAC Presets (TDMA / NOMA / RSMA)** | Applies throughput scaling for orthogonal, non-orthogonal, and rate-splitting strategies. |
| **Kinematics & Dynamics** | Updates UAV position, heading, altitude with turn rate, climb limits, and wind. |
| **Security Modules** | Includes jammer, eavesdropper risk quantification, and link interference modeling. |
| **3D Orbit Visualization** | Displays LEO/MEO/GEO reference rings and UAV constellation in 3D. |
| **Flight Animation** | Time-series animation of UAV motion in 2D. |
| **Metrics & Charts** | Throughput, battery, SoC, and eavesdrop risk plotted over time. |
| **Export Suite** | CSV / JSON / ZIP export of metrics, trajectories, fleet, and simulation settings. |

---

## 🧮 Core Models & Workflow

1. **Initialization**  
   Randomly generate UAVs (sources, relays, sinks) and (optionally) adversaries (jammer, eavesdropper).  

2. **Atmospheric Computations**  
   Evaluate ISA temperature \(T(h)\), pressure \(p(h)\), and density \(\rho(h)\).  

3. **Aerodynamic Power**  
   Decompose power requirement:  
   \[
     P_{\text{total}} = P_{\text{induced}} + P_{\text{parasite}} + P_{\text{climb}}
   \]
   with standard formulas for each term.

4. **Electrical Power Conversion**  
   \[
     P_{\text{elec}} = \frac{P_{\text{total}}}{\eta_{\text{prop}}\,\eta_{\text{motor}}}
   \]

5. **State Update (Kinematics)**  
   Constrain heading changes, apply wind, climb angle, and update (x, y, h).

6. **Link Budget & Graph Construction**  
   For each transmitter–receiver pair, compute path gain (Two-Ray, FSPL) + Rician fading, then SINR → capacity.  
   Build directed capacity graph \(G(V, E)\).

7. **MAC Scaling & Flow Computation**  
   Scale link capacities by MAC share factor and run a “widest path” heuristic (resembling Ford–Fulkerson) to compute aggregate throughput.

8. **Logging, Visualization & Export**  
   Log per-step metrics, update plots/animations, and provide downloadable outputs.

---

## 🧭 Quick Start

```bash
git clone https://github.com/yourusername/autonomous-uav-networks.git
cd autonomous-uav-networks
pip install -r requirements.txt
streamlit run app.py
