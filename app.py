import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import time
import networkx as nx

# --- 1. FRONTEND CONFIGURATION & STATE ---
st.set_page_config(page_title="Bangalore Traffic Assist", layout="wide", initial_sidebar_state="collapsed")

if 'active_events' not in st.session_state:
    st.session_state.active_events = []
if 'intro_played' not in st.session_state:
    st.session_state.intro_played = False

# --- 2. THE SPLASH SCREEN ANIMATION ---
if not st.session_state.intro_played:
    splash = st.empty()
    with splash.container():
        st.markdown(f"""
            <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 80vh;'>
                <h1 style='font-size: 4rem; font-weight: bold;'>Bangalore Traffic Assist</h1>
                <h3 style='color: gray;'>System Boot: {datetime.datetime.now().strftime('%d %b %Y | %H:%M:%S')}</h3>
            </div>
        """, unsafe_allow_html=True)
    time.sleep(2) 
    splash.empty() 
    st.session_state.intro_played = True

# --- 3. BACKEND AI ARCHITECTURE ---
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        return torch.matmul(A, F.relu(self.linear(x)))

class STGNN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, horizon):
        super(STGNN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.gcn = GraphConvolution(in_features, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2 * horizon)

    def forward(self, x, A):
        batch_size, num_nodes, time_steps, features = x.shape
        x_space = x.transpose(1, 2).reshape(batch_size * time_steps, num_nodes, features)
        gcn_out = self.gcn(x_space, A)
        gcn_out = gcn_out.reshape(batch_size, time_steps, num_nodes, self.hidden_dim)
        gru_in = gcn_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, self.hidden_dim)
        _, h_n = self.gru(gru_in)
        predictions = self.fc(h_n.squeeze(0))
        return predictions.reshape(batch_size, num_nodes, 2, self.horizon)

@st.cache_resource
def load_system():
    df_nodes = pd.read_csv('node_coordinates.csv')
    A = np.load('adjacency_matrix.npy')
    A_hat = A + np.eye(A.shape[0])
    A_tensor = torch.tensor(A_hat, dtype=torch.float32)
    
    model = STGNN(num_nodes=753, in_features=2, hidden_dim=64, horizon=3)
    try:
        model.load_state_dict(torch.load('stgnn_weights_smart.pth', map_location=torch.device('cpu')))
    except: pass 
    model.eval()
    
    G = nx.from_numpy_array(A)
    return model, A_tensor, df_nodes, G

model, A_tensor, df_nodes, G_base = load_system()

# --- DISPATCH & DIVERSION LOGIC ---
def recommend_resources(predicted_risk, df_nodes, max_off, max_bar, active_events, G, is_future=False):
    risk_scores = np.max(predicted_risk[:, 0, :], axis=1)
    df_temp = df_nodes.copy()
    df_temp['risk'] = risk_scores
    danger_zones = df_temp[df_temp['risk'] > 0.10].sort_values('risk', ascending=False)
    
    plan, explainers = [], []
    off_deployed, bar_deployed, nodes_at_risk = 0, 0, len(danger_zones)
    
    gz_node = active_events[0]['node_id'] if active_events else None
    
    for idx, row in danger_zones.iterrows():
        if off_deployed >= max_off and bar_deployed >= max_bar: break 
            
        a_off, a_bar = 0, 0
        if row['risk'] > 0.25:
            a_off = min(2, max_off - off_deployed)
            a_bar = min(1, max_bar - bar_deployed)
        else:
            a_off = min(1, max_off - off_deployed)
            a_bar = min(0, max_bar - bar_deployed)
            
        if a_off > 0 or a_bar > 0:
            ps_name = row['police_station']
            address_full = str(row['address'])
            
            # --- Dynamic Graph Diversion Generation ---
            neighbors = list(G.neighbors(idx))
            div_roads = []
            for n in neighbors:
                if n != gz_node:
                    clean_street = str(df_nodes.iloc[n]['address']).split(',')[0].strip()
                    div_roads.append(clean_street)
            
            unique_divs = list(set(div_roads))[:2]
            div_text = ", ".join(unique_divs) if unique_divs else "Nearest clear cross-street"
            
            # --- Formatted Action Protocol (Future vs Present) ---
            action_verb = "pre-deploy" if is_future else "send"
            action_text = f"**Inform {ps_name} Traffic PS** to {action_verb} **{a_off} Officers**"
            if a_bar > 0: action_text += f" and **{a_bar} Barricades**"
            action_text += f" here."
            
            block_verb = "Preemptive Blockade Point" if is_future else "Road Blocked"
            routing_text = f"🚧 **{block_verb}:** {address_full}\n\n↪️ **Suggested Diversion:** {div_text}"
            
            explainers.append({
                "station": ps_name, 
                "action": action_text, 
                "routing": routing_text, 
                "risk": row['risk']
            })
            
            plan.append({
                "Location": f"{ps_name} - {address_full[:30]}...",
                "Cascade Risk": row['risk'],
                "Officers": a_off, 
                "Barricades": a_bar
            })
            off_deployed += a_off
            bar_deployed += a_bar
            
    return plan, explainers, off_deployed, bar_deployed, nodes_at_risk

# --- 4. TOP APP BAR (Title + Dialogs) ---
current_time = datetime.datetime.now()
col_title, col_log, col_settings = st.columns([8, 1, 1])

with col_title:
    st.markdown(f"<h2>Bangalore Tactical Command <span style='font-size: 1rem; color: gray;'>| {current_time.strftime('%I:%M %p - %A, %b %d')}</span></h2>", unsafe_allow_html=True)

with col_settings:
    with st.popover("⚙️ Settings"):
        st.markdown("**Armory Limits**")
        off_input = st.text_input("Officers Available", value="25")
        bar_input = st.text_input("Barricades Available", value="15")
        try:
            max_officers = int(off_input)
            max_barricades = int(bar_input)
        except ValueError:
            max_officers, max_barricades = 25, 15
            
        st.divider()
        
        # --- NEW: EXPORT EVENT LOG FEATURE ---
        st.markdown("**Data Management**")
        if st.session_state.active_events:
            export_df = pd.DataFrame(st.session_state.active_events)
            # Clean up the output CSV
            export_df = export_df.rename(columns={
                "id": "Event_ID", "node_id": "Node_ID", "location_name": "Location", 
                "type": "Event_Type", "desc": "Description", "magnitude": "STGNN_Magnitude", 
                "closure": "Road_Closure", "scheduled_time": "Timestamp"
            })
            export_csv = export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Export Daily Event Log (CSV)",
                data=export_csv,
                file_name=f"Bangalore_Traffic_Log_{current_time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No events logged to export today.")

with col_log:
    with st.popover("🚨 Log Event"):
        ev_type = st.selectbox("Event Type", ["Sudden Breakdown", "Planned Event"])
        ev_severity = st.selectbox("Severity", ["Low", "Medium", "High"])
        
        location_options = [f"{str(row['address']).split(',')[0][:35]} (ID: {idx})" for idx, row in df_nodes.iterrows()]
        loc_choice = st.selectbox("Location (Ground Zero)", options=location_options, index=45)
        ev_node = location_options.index(loc_choice)
        
        ev_closure = st.checkbox("Requires Road Closure", value=True)
        
        if ev_type == "Planned Event":
            ev_date = st.date_input("Date")
            ev_time = st.time_input("Time")
        else:
            ev_date = current_time.date()
            ev_time = current_time.time()
            
        ev_desc = st.text_input("Notes", f"{ev_severity} severity incident")
        
        if st.button("Broadcast Incident", type="primary"):
            scheduled_dt = datetime.datetime.combine(ev_date, ev_time)
            
            mag = 0.8 if ev_severity == "High" else 0.55 if ev_severity == "Medium" else 0.25
            if ev_closure: mag = min(mag * 1.2, 1.0)
            
            st.session_state.active_events.append({
                "id": len(st.session_state.active_events),
                "node_id": ev_node, "location_name": loc_choice.split(' (ID:')[0], "type": ev_type, "desc": ev_desc,
                "magnitude": mag, "closure": 1.0 if ev_closure else 0.0,
                "scheduled_time": scheduled_dt,
                "is_active_now": scheduled_dt <= datetime.datetime.now()
            })
            st.rerun()

st.divider()

# --- 5. DATA PREPARATION ---
for ev in st.session_state.active_events:
    ev['is_active_now'] = ev['scheduled_time'] <= datetime.datetime.now()

live_events = [e for e in st.session_state.active_events if e['is_active_now']]
future_events = [e for e in st.session_state.active_events if not e['is_active_now']]
sudden_count = len([e for e in live_events if e['type'] == "Sudden Breakdown"])

# --- 6. KPIs ROW ---
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("City Nodes Overseen", "753")
with kpi2:
    st.metric("Total Active Events", f"{len(live_events)}")
with kpi3:
    st.metric("Future Planned Events", f"{len(future_events)}")
st.divider()

# --- 7. MAIN ENGINE (NOWCAST vs FORECAST LEDGERS) ---
col_now, col_fore = st.columns(2)

# ----- NOWCAST (Immediate Operations) -----
with col_now:
    st.subheader("🔴 NOWCAST: Immediate Operations")
    
    x_live = torch.zeros((1, 753, 12, 2), dtype=torch.float32)
    for ev in live_events:
        x_live[0, ev['node_id'], -1, 0] = ev['magnitude']  
        x_live[0, ev['node_id'], -1, 1] = ev['closure']  
            
    with torch.no_grad():
        preds_live = torch.sigmoid(model(x_live, A_tensor)).cpu().numpy()[0]
        
    plan_live, explainers_live, off_live, bar_live, _ = recommend_resources(preds_live, df_nodes, max_officers, max_barricades, live_events, G_base, is_future=False)
    
    with st.container(border=True):
        st.write(f"**Total Forces Deployed:** {off_live} / {max_officers} Officers | {bar_live} / {max_barricades} Barricades")
        st.progress(off_live / max_officers if max_officers > 0 else 0)
        
    st.markdown("### Active Dispatch Orders")
    if not explainers_live:
        st.success("No assets currently required. Network is clear.")
    else:
        for explainer in explainers_live[:5]:
            with st.expander(f"🚨 Dispatch: {explainer['station']} PS (Risk: {explainer['risk']:.1%})", expanded=True):
                st.markdown(explainer['action'])
                st.markdown(explainer['routing'])
                
    st.markdown("### Deployment Grid")
    if plan_live:
        st.dataframe(pd.DataFrame(plan_live).style.format({"Cascade Risk": "{:.1%}"}).background_gradient(subset=['Cascade Risk'], cmap='Reds', vmin=0.1, vmax=0.5), use_container_width=True, hide_index=True)

# ----- FORECAST (Predictive Radar) -----
with col_fore:
    st.subheader("🔮 FORECAST: Predictive Radar")
    
    if len(future_events) == 0:
        st.success("No future events logged. Advance radar is clear.")
    else:
        for ev in future_events:
            st.warning(f"**Predicted Impact Trigger:** {ev['scheduled_time'].strftime('%b %d, %I:%M %p')}")
            st.caption(f"Event: {ev['type']} @ {ev['location_name']} ({ev['desc']})")
            
            x_future = torch.zeros((1, 753, 12, 2), dtype=torch.float32)
            x_future[0, ev['node_id'], -1, 0] = ev['magnitude'] 
            x_future[0, ev['node_id'], -1, 1] = ev['closure']
                
            with torch.no_grad():
                preds_fut = torch.sigmoid(model(x_future, A_tensor)).cpu().numpy()[0]
                
            plan_fut, explainers_fut, _, _, _ = recommend_resources(preds_fut, df_nodes, max_officers, max_barricades, [ev], G_base, is_future=True)
            
            st.markdown("### Pre-Deployment Orders")
            if explainers_fut:
                for explainer in explainers_fut[:3]:
                    with st.expander(f"🚧 Pre-Plan: {explainer['station']} PS", expanded=False):
                        st.markdown(explainer['action'])
                        st.markdown(explainer['routing'])
            
            st.markdown("### Projected Deployment Grid")
            if plan_fut:
                st.dataframe(pd.DataFrame(plan_fut).style.format({"Cascade Risk": "{:.1%}"}).background_gradient(subset=['Cascade Risk'], cmap='Oranges', vmin=0.1, vmax=0.5), use_container_width=True, hide_index=True)
