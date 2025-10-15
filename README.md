<p align="center">
  <img src="banner3.PNG" alt="Autonomous UAV Networks Banner" width="100%">
</p>
MVP

**Autonomous Intelligence Networks for UAVs**  
Interactive Streamlit simulation mapping the ideas from *Sarkar & Gul (2023)* —  
“Artificial Intelligence-Based Autonomous UAV Networks: A Survey” (*Drones 7(5):322*).  

https://www.mdpi.com/2504-446X/7/5/322


---

### 🎯 Overview
This app transforms academic concepts on **AI-based UAV networks** into an interactive environment for studying:
- autonomous waypointing and connectivity  
- MAC and routing schemes (TDMA / NOMA / RSMA)  
- energy vs throughput trade-offs  
- jammer / eavesdropper security zones  
- cinematic 3D orbit visualization (LEO / MEO / GEO rings)

Built for rapid experimentation, analysis, and education on autonomous aerial mesh systems.

---

### ⚙️ Features
- **Autonomy & Routing:** UAVs navigate and re-route dynamically to maintain network capacity.  
- **Power & Energy:** Combines transmission + motion energy for each UAV over time.  
- **Security Simulation:** Models jamming and eavesdropping risk fields.  
- **Analytics Dashboard:** Live throughput, battery, and risk plots per step.  
- **3D Orbit View:** Optional cinematic layer showing orbits and UAV placement.  

---

### 🚀 Quick Start
```bash
pip install streamlit plotly numpy pandas networkx scipy
streamlit run app.py


