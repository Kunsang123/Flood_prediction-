"""Streamlit Dashboard for Flood Prediction System"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import json
import plotly.figure_factory as ff

st.set_page_config(page_title="Flood Prediction System", page_icon="🌊", layout="wide")
st.markdown("""<style>
.main-header {font-size:2.5rem;font-weight:bold;color:#1E88E5;text-align:center;margin-bottom:2rem}
</style>""", unsafe_allow_html=True)

_default_api = os.environ.get("API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("API URL", value=_default_api)
if 'predictions' not in st.session_state:
    st.session_state.predictions = []


def trigger_airflow_dag(dag_id):
    # Try localhost first (for local run), then service name (for docker)
    base_urls = [
        os.environ.get("AIRFLOW_URL", "http://localhost:8080"),
        "http://airflow-webserver:8080"
    ]
    
    auth = ("admin", "admin") # Default Airflow creds
    
    errors = []
    
    for base_url in base_urls:
        api_url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns"
        try:
            # st.info(f"Trying to trigger {dag_id} at {base_url}...") # verbose debug
            response = requests.post(
                api_url,
                json={"conf": {}},
                auth=auth,
                timeout=5
            )
            if response.status_code == 200:
                st.success(f"✅ Triggered DAG: {dag_id}")
                st.caption(f"Run ID: {response.json().get('dag_run_id')}")
                return True
            elif response.status_code == 409:
                 st.warning(f"⚠️ DAG {dag_id} is already running.")
                 return True
            else:
                 errors.append(f"{base_url}: {response.status_code} - {response.text}")
        except Exception as e:
            errors.append(f"{base_url}: {str(e)}")
            continue
            
    st.error(f"❌ Failed to reach Airflow.\nDetails:\n" + "\n".join(errors))
    return False


def make_prediction(params):
    try:
        # Include API Key for authentication
        headers = {"X-API-Key": "secret-token"}
        r = requests.post(f"{API_URL}/predict", json=params, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error ({r.status_code}): {r.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None


def get_risk_color(level):
    return {"LOW": "#4CAF50", "MODERATE": "#FF9800", "HIGH": "#f44336", "CRITICAL": "#9C27B0"}.get(level, "#1E88E5")


def main():
    st.markdown('<h1 class="main-header">🌊 Flood Prediction System</h1>', unsafe_allow_html=True)
    st.sidebar.header("📊 System Status")
    
    # Airflow Controls in Sidebar
    with st.sidebar.expander("⚙️ MLOps Pipelines", expanded=True):
        if st.button("🚀 Run Training Pipeline"):
            trigger_airflow_dag("flood_prediction_pipeline")
        
        if st.button("🔄 Trigger ALL Pipelines"):
             st.info("Triggering Training and Monitoring pipelines...")
             t1 = trigger_airflow_dag("flood_prediction_pipeline")
             t2 = trigger_airflow_dag("flood_prediction_monitoring")
             if t1 and t2:
                 st.success("✅ All pipelines triggered!")
            
    sidebar_status = st.sidebar.empty()
    try:
        h_resp = requests.get(f"{API_URL}/health", timeout=15)
        if h_resp.status_code == 200:
            h = h_resp.json()
            sidebar_status.success(f"✅ API: {h['status']}")
            model_loaded = h.get('model_loaded', False)
            model_name = h.get('model_name', 'N/A')
            
            # Model Selector (Aligned with CMP6230: RF, XGBoost, MLP)
            st.sidebar.markdown("---")
            available_models = ["xgboost", "rf", "mlp"]
            current_idx = available_models.index(model_name) if model_name in available_models else 0
            
            selected_model = st.sidebar.selectbox("Select Active Model", available_models, index=current_idx)
            
            if selected_model != model_name:
                if st.sidebar.button("Switch Model"):
                    with st.sidebar.status("🔄 Switching model...", expanded=True) as status:
                        try:
                            res = requests.post(f"{API_URL}/model/load", params={"model_type": selected_model}, 
                                             headers={"X-API-Key": "secret-token"}, timeout=30)
                            if res.status_code == 200:
                                status.update(label=f"✅ Switched to {selected_model}", state="complete")
                                time.sleep(2) # Give API a moment to stabilize
                                st.rerun()
                            else:
                                status.update(label="❌ Switch failed", state="error")
                                st.error(f"Switch failed: {res.text}")
                        except Exception as e:
                            status.update(label="❌ Connection Error", state="error")
                            st.error(f"Error: {e}")

            if model_loaded:
                st.sidebar.info(f"🤖 Active: **{model_name}**")
            else:
                st.sidebar.warning("⚠️ No model loaded")
        else:
            sidebar_status.error(f"❌ API Error: {h_resp.status_code}")
    except Exception as e:
        sidebar_status.error(f"❌ Connection Lost")
        st.sidebar.caption(f"Details: {str(e)[:50]}...")
        
        try:
            mi = requests.get(f"{API_URL}/model/info", timeout=5).json()
            st.sidebar.caption(f"Type: {mi.get('model_type','?')} | Features: {mi.get('n_features','?')}")
        except:
            pass
    except:
        st.sidebar.error("❌ API Unavailable")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Predict", "📈 History", "🧠 Explainability", "🛡️ Monitoring", "ℹ️ About"])
    
    with tab1:
        st.header("Enter Flood Risk Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader("Climate")
            monsoon = st.slider("Monsoon Intensity", 0, 20, 10)
            climate = st.slider("Climate Change", 0, 20, 10)
            coastal = st.slider("Coastal Vulnerability", 0, 20, 10)
            landslides = st.slider("Landslides", 0, 20, 10)
            watersheds = st.slider("Watersheds", 0, 20, 10)
        with c2:
            st.subheader("Infrastructure")
            topography = st.slider("Topography Drainage", 0, 20, 10)
            dams = st.slider("Dams Quality", 0, 20, 10)
            drainage = st.slider("Drainage Systems", 0, 20, 10)
            infra = st.slider("Infrastructure Deterioration", 0, 20, 10)
            river = st.slider("River Management", 0, 20, 10)
        with c3:
            st.subheader("Human Factors")
            urban = st.slider("Urbanization", 0, 20, 10)
            deforest = st.slider("Deforestation", 0, 20, 10)
            agri = st.slider("Agricultural Practices", 0, 20, 10)
            encroach = st.slider("Encroachments", 0, 20, 10)
            population = st.slider("Population Score", 0, 20, 10)
        with c4:
            st.subheader("Planning & Policy")
            disaster = st.slider("Disaster Preparedness", 0, 20, 10)
            siltation = st.slider("Siltation", 0, 20, 10)
            wetland = st.slider("Wetland Loss", 0, 20, 10)
            planning = st.slider("Planning Inadequacy", 0, 20, 10)
            political = st.slider("Political Factors", 0, 20, 10)

        if st.button("🔮 Predict Flood Probability", type="primary", use_container_width=True):
            params = {"MonsoonIntensity": float(monsoon), "TopographyDrainage": float(topography),
                      "RiverManagement": float(river), "Deforestation": float(deforest),
                      "Urbanization": float(urban), "ClimateChange": float(climate),
                      "DamsQuality": float(dams), "Siltation": float(siltation),
                      "AgriculturalPractices": float(agri), "Encroachments": float(encroach),
                      "IneffectiveDisasterPreparedness": float(disaster), "DrainageSystems": float(drainage),
                      "CoastalVulnerability": float(coastal), "Landslides": float(landslides),
                      "Watersheds": float(watersheds), "DeterioratingInfrastructure": float(infra),
                      "PopulationScore": float(population), "WetlandLoss": float(wetland),
                      "InadequatePlanning": float(planning), "PoliticalFactors": float(political)}
            with st.spinner("Predicting..."):
                result = make_prediction(params)
            if result:
                st.session_state.predictions.append({"timestamp": result['timestamp'],
                    "probability": result['flood_probability'], "risk_level": result['risk_level']})
                st.markdown("---")
                st.header("🎯 Result")
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=result['flood_probability'],
                        domain={'x':[0,1],'y':[0,1]}, title={'text':"Flood Probability"},
                        gauge={'axis':{'range':[0,100]}, 'bar':{'color':get_risk_color(result['risk_level'])},
                               'steps':[{'range':[0,30],'color':"#E8F5E9"},{'range':[30,60],'color':"#FFF3E0"},
                                        {'range':[60,80],'color':"#FFEBEE"},{'range':[80,100],'color':"#F3E5F5"}]}))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.metric("Flood Probability", f"{result['flood_probability']:.1f}%")
                    st.metric("Risk Level", result['risk_level'])
                    levels = {"LOW": ("success","✅ Low risk."), "MODERATE": ("warning","⚠️ Moderate risk."),
                              "HIGH": ("error","🚨 High risk!"), "CRITICAL": ("error","🆘 CRITICAL!")}
                    t, m = levels.get(result['risk_level'], ("info",""))
                    getattr(st, t)(m)

    with tab2:
        st.header("📈 History")
        if st.session_state.predictions:
            df = pd.DataFrame(st.session_state.predictions)
            st.plotly_chart(px.line(df, x='timestamp', y='probability', title='Predictions Over Time', markers=True), use_container_width=True)
            st.dataframe(df, use_container_width=True)
            if st.button("Clear History"):
                st.session_state.predictions = []
                st.rerun()
        else:
            st.info("No predictions yet.")

    with tab3:
        st.header("🧠 Model Explainability (SHAP)")
        st.markdown("""
        To comply with **GDPR Article 22**, this system uses SHAP (SHapley Additive exPlanations) to explain 
        which environmental factors are driving the current flood risk.
        """)
        
        # Look for the SHAP artifact from the latest training
        ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", "artifacts")
        # Use the name of the loaded model
        try:
            h_resp = requests.get(f"{API_URL}/health", timeout=5)
            model_name = h_resp.json().get('model_name', 'xgboost')
        except:
            model_name = "xgboost"
            
        shap_file = os.path.join(ARTIFACTS_PATH, f"{model_name}_shap.png")
        
        if os.path.exists(shap_file):
            st.image(shap_file, caption=f"Global Feature Importance for {model_name} (SHAP Summary Plot)", use_container_width=True)
            st.info("💡 **Insight**: Factors with longer bars to the right indicate higher contribution to flood probability.")
        else:
            st.warning(f"⚠️ SHAP plot for **{model_name}** not found.")
            st.caption(f"Checked path: {shap_file}")
            if st.button("Note on Explainability"):
                st.info("Explainability artifacts are generated during the automated 'Monitoring Dag' retraining cycle.")

    with tab4:
        st.header("🛡️ Drift Monitoring")
        st.markdown("Automated drift detection using **KS Test** (Data Drift) and **PSI** (Concept Drift).")
        
        # 1. Find latest JSON report
        REPORTS_PATH = os.environ.get("REPORTS_PATH", "artifacts/reports")
        PROCESSED_PATH = os.environ.get("PROCESSED_PATH", "data/processed")
        
        try:
            # Create directory if it doesn't exist (handled by monitor usually, but safe to check)
            if not os.path.exists(REPORTS_PATH):
                os.makedirs(REPORTS_PATH, exist_ok=True)
                
            # Find latest file starting with drift_report_
            report_files = [f for f in os.listdir(REPORTS_PATH) if f.startswith("drift_report_") and f.endswith(".json")]
            if report_files:
                latest_report = max(report_files, key=lambda f: os.path.getmtime(os.path.join(REPORTS_PATH, f)))
                report_path = os.path.join(REPORTS_PATH, latest_report)
                
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                # 2. Display Top-Level Metrics
                st.caption(f"Last Scan: {report.get('timestamp', 'Unknown')}")
                
                c1, c2, c3 = st.columns(3)
                drift_detected = report.get('drift_detected', False)
                retrain = report.get('retrain_recommended', False)
                avg_psi = report.get('average_psi', 0.0)
                
                c1.metric("Features Drifted", f"{len(report.get('drifted_features', []))}/{len(report.get('features', {}))}", 
                          delta="Drift Detected" if drift_detected else "Stable", delta_color="inverse")
                c2.metric("Concept Drift (Avg PSI)", f"{avg_psi:.4f}", 
                          delta="High Shift" if avg_psi > 0.2 else "Stable", delta_color="inverse")
                c3.metric("Retraining Status", "Required" if retrain else "Not Required", 
                          delta="Triggered" if retrain else "Monitoring", delta_color="inverse")
                
                if drift_detected:
                    st.error("🚨 Data Drift Detected! The production data distribution strictly differs from training data.")
                else:
                    st.success("✅ Data Distribution is stable.")

                # 3. Detailed Feature Analysis
                st.subheader("Feature Drift Details")
                features_data = []
                for feat, metrics in report.get('features', {}).items():
                    features_data.append({
                        "Feature": feat,
                        "Drift": "🔴 Yes" if metrics['ks_drift'] else "🟢 No",
                        "KS Statistic": f"{metrics['ks_statistic']:.4f}",
                        "P-Value": f"{metrics['ks_p_value']:.4f}",
                        "PSI": f"{metrics['psi_value']:.4f}"
                    })
                
                df_metrics = pd.DataFrame(features_data)
                st.dataframe(df_metrics, use_container_width=True)

                # 4. Visual Distribution Comparison
                st.subheader("Distribution Comparison")
                selected_feat = st.selectbox("Select Feature to Visualize", [f['Feature'] for f in features_data])
                
                if selected_feat:
                    # Load data for visualization (Heavy operation, cached)
                    @st.cache_data
                    def load_comparison_data():
                        try:
                            # Assuming standard paths
                            ref = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train.csv"))
                            curr = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test.csv")) # Using test as proxy for 'current' in demo
                            return ref, curr
                        except:
                            return None, None
                            
                    ref_df, curr_df = load_comparison_data()
                    
                    if ref_df is not None and curr_df is not None:
                        try:
                            fig_dist = ff.create_distplot([ref_df[selected_feat].dropna(), curr_df[selected_feat].dropna()], 
                                                        ['Reference', 'Current'], show_hist=False, show_rug=False)
                            fig_dist.update_layout(title=f"Shift: {selected_feat}")
                            st.plotly_chart(fig_dist, use_container_width=True)
                        except Exception as plotting_error:
                            st.warning(f"Plotting error: {plotting_error}")
                            st.bar_chart(ref_df[selected_feat].head(100))
                    else:
                        st.warning("⚠️ Could not load source datasets for visualization.")

            else:
                st.info("ℹ️ No drift reports found.")
                
        except Exception as e:
            st.error(f"Error loading monitoring dashboard: {e}")
            
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Manual Check")
            if st.button("Run Python Script (Fast)"):
                 with st.spinner("Running local analysis..."):
                    try:
                        import subprocess
                        import sys
                        REPORTS_PATH = os.environ.get("REPORTS_PATH", "artifacts/reports")
                        subprocess.run([sys.executable, "-m", "src.monitor"], check=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Job failed: {e}")
        
        with c2:
            st.subheader("Airflow Task")
            if st.button("Trigger Monitoring DAG"):
                trigger_airflow_dag("flood_prediction_monitoring")

    with tab5:
        st.header("ℹ️ About")
        st.markdown("""### Flood Prediction MLOps System
        **Models**: Random Forest, XGBoost, MLP | **Metrics**: R², RMSE, MAE
        **Stack**: Airflow, MLflow, FastAPI, Streamlit, Prometheus, Grafana, Docker
        **Risk**: 🟢 LOW (0-30%) | 🟡 MODERATE (30-60%) | 🔴 HIGH (60-80%) | 🟣 CRITICAL (80-100%)""")


if __name__ == "__main__":
    main()
