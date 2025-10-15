<p align="center">
  <img src="banner3.PNG" alt="Autonomous UAV Networks Banner" width="100%">
</p>


✈️**Autonomous Intelligence Networks for UAVs**  
# 🛰️ Autonomous UAV Networks Simulator (Aerospace-Accurate Edition)

> **High-Fidelity Simulation of Autonomous UAV Mesh Networks**  
> Built for aerospace-grade research, education, and AI-driven network resilience modeling.  
> Inspired by *Sarkar & Gul (2023), “Artificial Intelligence-Based Autonomous UAV Networks: A Survey.”*

---

## 🌐 Overview

This Streamlit-based simulator models **autonomous UAV mesh networks** with full **RF propagation**, **energy**, and **threat environment** realism.  
It combines aerospace physics, dynamic networking, and visual analytics into one interactive, browser-based lab.

### ✳️ Key Features

- 📡 **Physics-Accurate Channel Modeling** — Free-space pathloss (FSPL), MHz–km scaling, and Gaussian shadowing  
- ⚙️ **Dynamic Graph Construction** — Source, relay, and sink topology updates per timestep  
- 🔋 **Propulsion Energy Model** — Speed-dependent drag power (∝ v³) and per-UAV battery tracking  
- 🚨 **Threat Simulation** — Configurable *jammer* and *eavesdropper* entities  
- 🧠 **Adaptive MAC Schemes** — TDMA (orthogonal), NOMA (non-orthogonal), and RSMA (rate-splitting)  
- 🌍 **3D Orbit View** — Realistic LEO/MEO/GEO visual context with fully interactive Plotly visualization  
- 🎛️ **Neon or B&W Themes** — Choose between futuristic “digital-green” or minimalist monochrome  

---

## 🧭 Quick Start

```bash
# Clone this repository
git clone https://github.com/<your-username>/Autonomous-UAV-Networks-Simulator.git
cd Autonomous-UAV-Networks-Simulator

# Install dependencies
pip install -r requirements.txt

# Run the simulator
streamlit run app.py
