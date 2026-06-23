import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import time
import networkx as nx
import pydeck as pdk

# --- 1. FRONTEND CONFIGURATION & STATE ---
st.set_page_config(page_title="Bangalore Tactical Command", layout="wide", initial_sidebar_state="collapsed")

if 'active_events' not in st.session_state:
    st.session_state.active_events = []
if 'intro_played' not in st.session_state:
    st.session_state.intro_played = False

if not st.session_state.intro_played:
    splash = st.empty()
    with splash.container():
        st.markdown(f"""
            <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 80vh;'>
                <h1 style='font-size: 4rem; font-weight: bold;'>Bangalore Tactical Command</h1>
                <h3 style='color: gray;'>System Boot: {datetime.datetime.now().strftime('%d %b %Y | %H:%M:%S')}</h3>
            </div>
        """, unsafe_allow_html=True)
    time.sleep(2) 
    splash.empty() 
    st.session_state.intro_played = True

# --- 2. BACKEND AI ARCHITECTURE ---
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
    
    A = np.load('adjacency_matrix.npy', allow_pickle=True)
    A_hat = A.astype(float) + np.eye(A.shape[0])
    A_tensor = torch.tensor(A_hat, dtype=torch.float32)
    
    try:
        df_hist = pd.read_csv('historical_blackspots.csv')
        df_hist = df_hist.sort_values('spatial_node_id').reset_index(drop=True)
        h_vector = df_hist['H_Score'].values
    except:
        df_hist = pd.DataFrame()
        h_vector = np.zeros(753)
    
    model = STGNN(num_nodes=753, in_features=2, hidden_dim=64, horizon=3)
    try:
        model.load_state_dict(torch.load('stgnn_weights_smart.pth', map_location=torch.device('cpu')))
    except: pass 
    model.eval()
    
    G = nx.from_numpy_array(A)
    return model, A_tensor, df_nodes, G, df_hist, h_vector

model, A_tensor, df_nodes, G_base, df_hist, h_vector = load_system()
location_options = [f"{str(row['address']).split(',')[0][:35]} (ID: {idx})" for idx, row in df_nodes.iterrows()]

# --- 3. DISPATCH & DIVERSION LOGIC ---
def calculate_ambient_patrols(df_hist, max_amb_off, max_amb_bar):
    plan = []
    off_dep, bar_dep = 0, 0
    if df_hist.empty or 'H_Score' not in df_hist.columns: return plan, off_dep, bar_dep
        
    top_blackspots = df_hist.sort_values('H_Score', ascending=False)
    for idx, row in top_blackspots.iterrows():
        if off_dep >= max_amb_off and bar_dep >= max_amb_bar: break
        if row['H_Score'] < 0.05: continue
            
        a_off = min(1, max_amb_off - off_dep)
        a_bar = min(0, max_amb_bar - bar_dep) 
        if a_off > 0 or a_bar > 0:
            plan.append({"Location": f"{row['police_station']} - {str(row['address'])[:30]}", "Historical Risk": row['H_Score'], "Officers": a_off, "Barricades": a_bar})
            off_dep += a_off
            bar_dep += a_bar
    return plan, off_dep, bar_dep

def recommend_resources(predicted_risk, df_nodes, max_off, max_bar, active_events, G, is_future=False):
    risk_scores = np.max(predicted_risk[:, 0, :], axis=1)
    df_temp = df_nodes.copy()
    df_temp['risk'] = risk_scores
    danger_zones = df_temp[df_temp['risk'] > 0.10].sort_values('risk', ascending=False)
    
    plan, explainers = [], []
    off_deployed, bar_deployed = 0, 0
    gz_nodes = [ev['node_id'] for ev in active_events] if active_events else []
    
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
            
            neighbors = list(G.neighbors(idx))
            div_roads = [str(df_nodes.iloc[n]['address']).split(',')[0].strip() for n in neighbors if n not in gz_nodes]
            unique_divs = list(set(div_roads))[:2]
            div_text = ", ".join(unique_divs) if unique_divs else "Nearest clear cross-street"
            
            action_verb = "pre-deploy" if is_future else "send"
            action_text = f"**Inform {ps_name} Traffic PS** to {action_verb} **{a_off} Officers**"
            if a_bar > 0: action_text += f" and **{a_bar} Barricades**"
            action_text += f" here."
            
            block_verb = "Preemptive Blockade Point" if is_future else "Road Blocked"
            routing_text = f"🚧 **{block_verb}:** {address_full}\n\n↪️ **Suggested Diversion:** {div_text}"
            
            explainers.append({"station": ps_name, "action": action_text, "routing": routing_text, "risk": row['risk']})
            plan.append({"Location": f"{ps_name} - {address_full[:30]}...", "Cascade Risk": row['risk'], "Officers": a_off, "Barricades": a_bar})
            off_deployed += a_off
            bar_deployed += a_bar
            
    return plan, explainers, off_deployed, bar_deployed

# --- 4. MODALS (DIALOGS) ---
@st.dialog("🚨 Log Tactical Incident")
def log_event_dialog():
    ev_type = st.selectbox("Event Type", ["Sudden Breakdown", "Planned Event"])
    ev_severity = st.selectbox("Severity", ["Low", "Medium", "High"])
    loc_choice = st.selectbox("Location (Type to search)", options=location_options, index=45)
    ev_node = location_options.index(loc_choice)
    ev_closure = st.checkbox("Requires Road Closure", value=True)
    
    current_time = datetime.datetime.now()
    if ev_type == "Planned Event":
        ev_date = st.date_input("Date") 
        ev_time = st.time_input("Time") 
        scheduled_dt = datetime.datetime.combine(ev_date, ev_time)
    else:
        scheduled_dt = current_time
        
    ev_desc = st.text_input("Notes", f"{ev_severity} severity incident")
    
    if st.button("Broadcast Incident", type="primary", use_container_width=True):
        if ev_type == "Planned Event" and scheduled_dt < (current_time - datetime.timedelta(minutes=1)):
            st.error("⏳ Invalid Time: Cannot schedule an event in the past.")
        else:
            mag = 0.8 if ev_severity == "High" else 0.55 if ev_severity == "Medium" else 0.25
            if ev_closure: mag = min(mag * 1.2, 1.0)
            st.session_state.active_events.append({
                "id": len(st.session_state.active_events) + int(time.time()), 
                "node_id": ev_node, "location_name": loc_choice.split(' (ID:')[0], 
                "type": ev_type, "desc": ev_desc, "magnitude": mag, 
                "closure": 1.0 if ev_closure else 0.0, "scheduled_time": scheduled_dt,
                "is_active_now": scheduled_dt <= datetime.datetime.now()
            })
            st.toast(f"✅ {ev_type} logged successfully! AI cascading risk...", icon="✅")
            time.sleep(0.6)
            st.rerun()

@st.dialog("🚑 Deploy Emergency Services")
def emergency_ops_dialog():
    ops_type = st.radio("Dispatch Unit", ["🚑 Ambulance", "🚒 Fire Engine"], horizontal=True)
    ops_loc = st.selectbox("Search Drop Zone (Type to search)", options=location_options)
    st.text_input("Situation Notes", placeholder="E.g., 2 casualties, trapped.")
    if st.button("Dispatch Unit", type="primary", use_container_width=True):
        agency = "Nearest Hospital" if "Ambulance" in ops_type else "Nearest Fire Station"
        st.toast(f"🚨 {agency} broadcasted to {ops_loc.split(' (ID:')[0]} successfully!", icon="✅")
        time.sleep(0.6)
        st.rerun()

# --- 5. TOP APP BAR ---
current_time = datetime.datetime.now()
col_title, col_controls = st.columns([7.5, 2.5])

with col_title:
    st.markdown(f"<h2>Bangalore Tactical Command <br><span style='font-size: 1.2rem; color: gray;'>{current_time.strftime('%I:%M %p - %A, %b %d')}</span></h2>", unsafe_allow_html=True)

with col_controls:
    if st.button("🚨 Log Event", type="primary", use_container_width=True):
        log_event_dialog()

    sub_col_ops, sub_col_set = st.columns(2)
    
    with sub_col_ops:
        if st.button("🚑 Ops", use_container_width=True):
            emergency_ops_dialog()
                
    with sub_col_set:
        with st.popover("⚙️ Settings", use_container_width=True):
            st.markdown("**City Context**")
            city_context = st.selectbox("Select Current Condition", ["Normal Operations", "Heavy Rain / Waterlogging", "Major Festival / Holiday"], label_visibility="collapsed")
            baseline_risk = 0.15 if "Rain" in city_context else 0.10 if "Festival" in city_context else 0.0
            
            st.markdown("**Armory Limits**")
            max_officers = int(st.text_input("Officers Available", value="30"))
            max_barricades = int(st.text_input("Barricades Available", value="15"))
            
            st.markdown("**Data Management**")
            if st.session_state.active_events:
                export_df = pd.DataFrame(st.session_state.active_events)
                st.download_button("📥 Export Log (CSV)", data=export_df.to_csv(index=False).encode('utf-8'), file_name=f"Traffic_Log_{current_time.strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
            else:
                st.info("No events to export.")

st.divider()

# --- 6. DATA PREPARATION & CALCULATIONS ---
for ev in st.session_state.active_events:
    ev['is_active_now'] = ev['scheduled_time'] <= datetime.datetime.now()

live_events = [e for e in st.session_state.active_events if e['is_active_now']]
future_events = [e for e in st.session_state.active_events if not e['is_active_now']]

amb_max_off = int(max_officers * 0.40) 
tac_max_off = max_officers - amb_max_off 

if live_events:
    x_live = torch.zeros((1, 753, 12, 2), dtype=torch.float32)
    for ev in live_events:
        x_live[0, ev['node_id'], -1, 0] = ev['magnitude']  
        x_live[0, ev['node_id'], -1, 1] = ev['closure']  
            
    with torch.no_grad():
        preds_live_raw = torch.sigmoid(model(x_live, A_tensor)).cpu().numpy()[0]
        preds_live = np.clip(preds_live_raw + (h_vector[:, np.newaxis, np.newaxis] * 0.15) + baseline_risk, 0, 1)
        
    plan_live, explainers_live, off_live, bar_live = recommend_resources(preds_live, df_nodes, tac_max_off, max_barricades, live_events, G_base, is_future=False)
    map_risk = np.max(preds_live[:, 0, :], axis=1)
else:
    amb_plan, amb_off, amb_bar = calculate_ambient_patrols(df_hist, amb_max_off, max_barricades)
    map_risk = (h_vector * 0.15) + baseline_risk

# --- 7. KPIs ROW ---
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("City Nodes Overseen", "753", delta=f"{city_context}", delta_color="off" if baseline_risk == 0 else "inverse")
with kpi2:
    st.metric("Total Active Events", f"{len(live_events)}")
with kpi3:
    st.metric("Future Planned Events", f"{len(future_events)}")
st.divider()

# --- 8. TACTICAL OVERVIEW (MAP & INCIDENT COMMAND) ---
col_map, col_resolve = st.columns([7, 3])

with col_map:
    df_map = df_nodes.copy()
    df_map['risk'] = map_risk
    df_active = df_map[df_map['risk'] > 0.05]
    
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_active,
        opacity=0.8,
        get_position=["longitude", "latitude"],
        get_weight="risk",
        threshold=0.05,
        radiusPixels=50,
        color_range=[[255, 255, 178], [254, 204, 92], [253, 141, 60], [240, 59, 32], [189, 0, 38]]
    )
    
    view_state = pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=10.5, pitch=45, bearing=0)
    
    st.markdown("### 🗺️ Live Tactical Overview")
    st.pydeck_chart(pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{police_station}\nRisk Level: {risk}"}
    ))

with col_resolve:
    with st.expander("🚦 Active Incidents Command", expanded=True):
        if live_events:
            for ev in live_events:
                with st.container(border=True):
                    st.write(f"**{ev['location_name']}** | {ev['type']}")
                    st.caption(ev['desc'])
                    if st.button("✅ Resolve & Unblock", key=f"res_{ev['id']}", use_container_width=True):
                        st.session_state.active_events = [e for e in st.session_state.active_events if e['id'] != ev['id']]
                        st.toast("✅ Incident resolved. Grid resetting.", icon="✅")
                        time.sleep(0.5)
                        st.rerun()
        else:
            st.success("Network is clear. Displaying baseline historical traffic.")

st.divider()

# --- 9. MAIN ENGINE (NOWCAST vs FORECAST LEDGERS) ---
col_now, col_fore = st.columns(2)

with col_now:
    st.subheader("🔴 NOWCAST: Immediate Operations")
    
    if not live_events:
        with st.container(border=True):
            st.write(f"**Standard Baseline Forces Deployed:** {amb_off} / {max_officers} Officers")
            
        st.markdown("### Historical Baseline Postings (H-Vector)")
        if amb_plan:
            st.dataframe(pd.DataFrame(amb_plan).style.format({"Historical Risk": "{:.1%}"}).background_gradient(subset=['Historical Risk'], cmap='Blues', vmin=0.0, vmax=1.0), width='stretch', hide_index=True)
    else:
        with st.container(border=True):
            st.write(f"**Tactical Reserve Deployed:** {off_live} / {tac_max_off} Officers | {bar_live} / {max_barricades} Barricades")
            st.progress(off_live / tac_max_off if tac_max_off > 0 else 0)
            st.caption(f"Note: {amb_max_off} officers retained for baseline traffic duty.")
            
        st.markdown("### Active Dispatch Orders")
        if explainers_live:
            top_order = explainers_live[0]
            with st.expander(f"🚨 Priority Dispatch: {top_order['station']} PS (Risk: {top_order['risk']:.1%})", expanded=True):
                st.markdown(top_order['action'])
                st.markdown(top_order['routing'])
                if st.button("📞 Call PS", key="call_live_top"): st.toast(f"Calling {top_order['station']} PS...", icon="📞")
                
            if len(explainers_live) > 1:
                with st.expander(f"📋 View All {len(explainers_live) - 1} Additional Dispatch Orders", expanded=False):
                    for i, explainer in enumerate(explainers_live[1:]):
                        st.markdown(f"**{explainer['station']} PS** (Risk: {explainer['risk']:.1%})")
                        st.markdown(explainer['action'])
                        st.markdown(explainer['routing'])
                        if st.button("📞 Call PS", key=f"call_live_{i}"): st.toast(f"Calling {explainer['station']} PS...", icon="📞")
                        st.divider()

        if plan_live:
            st.markdown("### STGNN Deployment Grid")
            st.dataframe(pd.DataFrame(plan_live).style.format({"Cascade Risk": "{:.1%}"}).background_gradient(subset=['Cascade Risk'], cmap='Reds', vmin=0.1, vmax=0.5), width='stretch', hide_index=True)

with col_fore:
    st.subheader("🔮 FORECAST: Predictive Radar")
    
    if len(future_events) == 0:
        st.info("No future events logged. Advance radar is clear.")
    else:
        for ev in future_events:
            st.warning(f"**Predicted Impact Trigger:** {ev['scheduled_time'].strftime('%b %d, %I:%M %p')}")
            st.caption(f"Event: {ev['type']} @ {ev['location_name']} ({ev['desc']})")
            
            x_future = torch.zeros((1, 753, 12, 2), dtype=torch.float32)
            x_future[0, ev['node_id'], -1, 0] = ev['magnitude'] 
            x_future[0, ev['node_id'], -1, 1] = ev['closure']
                
            with torch.no_grad():
                preds_fut_raw = torch.sigmoid(model(x_future, A_tensor)).cpu().numpy()[0]
                preds_fut = np.clip(preds_fut_raw + (h_vector[:, np.newaxis, np.newaxis] * 0.15) + baseline_risk, 0, 1)
                
            plan_fut, explainers_fut, _, _ = recommend_resources(preds_fut, df_nodes, tac_max_off, max_barricades, [ev], G_base, is_future=True)
            
            st.markdown("### Pre-Deployment Orders")
            if explainers_fut:
                top_fut_order = explainers_fut[0]
                with st.expander(f"🚧 Priority Pre-Plan: {top_fut_order['station']} PS", expanded=True):
                    st.markdown(top_fut_order['action'])
                    st.markdown(top_fut_order['routing'])
                    if st.button("📞 Notify PS", key=f"call_fut_top_{ev['id']}"): st.toast(f"Alerted {top_fut_order['station']} PS.", icon="📞")
                    
                if len(explainers_fut) > 1:
                    with st.expander(f"📋 View All {len(explainers_fut) - 1} Future Pre-Deployments", expanded=False):
                        for i, explainer in enumerate(explainers_fut[1:]):
                            st.markdown(f"**{explainer['station']} PS**")
                            st.markdown(explainer['action'])
                            st.markdown(explainer['routing'])
                            if st.button("📞 Notify PS", key=f"call_fut_{ev['id']}_{i}"): st.toast(f"Alerted {explainer['station']} PS.", icon="📞")
                            st.divider()
            
            if plan_fut:
                st.markdown("### Projected Deployment Grid")
                st.dataframe(pd.DataFrame(plan_fut).style.format({"Cascade Risk": "{:.1%}"}).background_gradient(subset=['Cascade Risk'], cmap='Oranges', vmin=0.1, vmax=0.5), width='stretch', hide_index=True)
