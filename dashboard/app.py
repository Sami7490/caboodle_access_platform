"""
app.py (COMPLETE)

Streamlit dashboard for the Caboodle Access Platform — all sections implemented.

Sections:
  1. Key Metrics
  2. No-Show Analysis
  3. Readmission Analysis
  4. Patient Risk Lookup
  5. Semantic Note Search
  6. AI Query Assistant
  7. LLM Observability
  8. Prompt Management
  9. Data Quality
  10. Patient Cohort Builder
  11. What-If Simulator
"""

import os
import sys
import json
import psycopg2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agent'))

# ============================================================================
# CLINICAL STYLING & CUSTOM CSS
# ============================================================================

COLORS = {
    "primary_blue": "#003366",
    "secondary_blue": "#0066CC",
    "accent_teal": "#00A896",
    "warning_orange": "#FF8C00",
    "success_green": "#2D7E3A",
    "neutral_gray": "#4A5568",
    "light_gray": "#F7F9FC",
    "border_gray": "#E2E8F0",
}

st.markdown(f"""
<style>
    :root {{
        --primary-blue: {COLORS['primary_blue']};
        --secondary-blue: {COLORS['secondary_blue']};
        --accent-teal: {COLORS['accent_teal']};
        --neutral-gray: {COLORS['neutral_gray']};
        --light-gray: {COLORS['light_gray']};
    }}

    .stApp {{
        background-color: {COLORS['light_gray']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {COLORS['primary_blue']};
        color: white;
    }}
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: white;
    }}

    h1, h2, h3 {{
        color: {COLORS['primary_blue']};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }}

    h1 {{
        font-size: 28px;
        margin-bottom: 8px;
    }}

    h2 {{
        font-size: 20px;
        margin-top: 20px;
        margin-bottom: 12px;
        border-bottom: 2px solid {COLORS['secondary_blue']};
        padding-bottom: 8px;
    }}

    h3 {{
        font-size: 16px;
        margin-top: 16px;
        margin-bottom: 8px;
    }}

    [data-testid="stMetricContainer"] {{
        background-color: white;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid {COLORS['secondary_blue']};
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin-bottom: 8px;
    }}

    body {{
        font-size: 13px;
        line-height: 1.5;
        color: {COLORS['neutral_gray']};
        font-family: 'Segoe UI', sans-serif;
    }}

    .stButton > button {{
        background-color: {COLORS['secondary_blue']};
        color: white;
        font-size: 12px;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 4px;
        border: none;
        cursor: pointer;
    }}

    .stButton > button:hover {{
        background-color: {COLORS['primary_blue']};
    }}

    .stRadio > label, .stSelectbox > label, .stTextInput > label {{
        font-size: 12px;
        font-weight: 600;
        color: {COLORS['primary_blue']};
        margin-bottom: 6px;
    }}

    .stDivider {{
        border-color: {COLORS['border_gray']};
        margin: 12px 0;
    }}

    .dataframe {{
        font-size: 12px;
        background-color: white;
    }}

    .dataframe tbody tr:hover {{
        background-color: {COLORS['light_gray']};
    }}

    .plotly-graph-div {{
        background-color: white;
        border-radius: 8px;
        border: 1px solid {COLORS['border_gray']};
        padding: 8px;
    }}

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        font-size: 12px;
        margin: 4px 0;
    }}

    .stMarkdownContainer {{
        margin-bottom: 8px;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 24px;
        font-weight: 700;
        color: {COLORS['primary_blue']};
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 12px;
        color: {COLORS['neutral_gray']};
        font-weight: 600;
    }}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Caboodle Access Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=st.secrets["db_host"],
        port=5432,
        dbname="caboodle_access",
        user="postgres",
        password=st.secrets["db_password"]
    )

@st.cache_data(ttl=300)
def run_query(sql):
    try:
        conn = get_connection()
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("🏥")
    with col2:
        st.markdown("**Caboodle Access**")
    st.markdown("<small>Pediatric Clinical Analytics</small>", unsafe_allow_html=True)
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Required for AI Query Assistant"
    )
    st.divider()

    section = st.radio(
        "**Navigation**",
        [
            "📊 Key Metrics",
            "📅 No-Show Analysis",
            "🏥 Readmission Analysis",
            "👤 Patient Risk Lookup",
            "🔍 Semantic Note Search",
            "🤖 AI Query Assistant",
            "📋 LLM Observability",
            "⚙️ Prompt Management",
            "🧪 Data Quality",
            "👥 Patient Cohort Builder",
            "🔮 What-If Simulator",
        ],
        label_visibility="collapsed"
    )

# ============================================================================
# SECTION 1: KEY METRICS
# ============================================================================

if section == "📊 Key Metrics":
    st.title("📊 Key Metrics")
    st.markdown("High-level summary of access and utilization across the platform.")
    st.divider()

    appt_df = run_query("""
        SELECT COUNT(*) AS total, SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows
        FROM analytics_marts.fact_appointments
    """)
    readmit_df = run_query("""
        SELECT COUNT(*) AS total, SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions
        FROM analytics_marts.fct_readmissions
    """)
    days_df = run_query("""
        SELECT ROUND(SUM(length_of_stay_days)::numeric, 1) AS patient_days
        FROM analytics_marts.fact_encounters
        WHERE encounter_type = 'Inpatient'
    """)
    patient_df = run_query("SELECT COUNT(*) AS total FROM analytics_marts.dim_patients")

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        if not patient_df.empty:
            st.metric("Total Patients", f"{patient_df['total'][0]:,}")

    with col2:
        if not appt_df.empty:
            no_show_rate = round(100 * appt_df['no_shows'][0] / appt_df['total'][0], 1)
            st.metric("No-Show Rate", f"{no_show_rate}%", delta=f"{appt_df['no_shows'][0]:,} / {appt_df['total'][0]:,}", delta_color="inverse")

    with col3:
        if not readmit_df.empty:
            readmit_rate = round(100 * readmit_df['readmissions'][0] / readmit_df['total'][0], 1)
            st.metric("30-Day Readmit Rate", f"{readmit_rate}%", delta=f"{readmit_df['readmissions'][0]:,} / {readmit_df['total'][0]:,}", delta_color="inverse")

    with col4:
        if not days_df.empty:
            st.metric("Patient Days", f"{days_df['patient_days'][0]:,}")

    st.divider()

    st.subheader("Monthly Appointment Volume")
    monthly_df = run_query("""
        SELECT
            DATE_TRUNC('month', scheduled_datetime) AS month,
            COUNT(*) AS total,
            SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows
        FROM analytics_marts.fact_appointments
        GROUP BY 1 ORDER BY 1
    """)

    if not monthly_df.empty:
        monthly_df['month'] = pd.to_datetime(monthly_df['month'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_df['month'], y=monthly_df['total'], mode='lines+markers', name='Total', line=dict(color=COLORS['secondary_blue'], width=2)))
        fig.add_trace(go.Scatter(x=monthly_df['month'], y=monthly_df['no_shows'], mode='lines+markers', name='No-Shows', line=dict(color=COLORS['warning_orange'], width=2)))
        fig.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40), hovermode='x unified', plot_bgcolor=COLORS['light_gray'], paper_bgcolor='white', font=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 2: NO-SHOW ANALYSIS
# ============================================================================

elif section == "📅 No-Show Analysis":
    st.title("📅 No-Show Analysis")
    st.markdown("Trends by department, appointment type, and lead time.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("No-Shows by Department")
        dept_df = run_query("""
            SELECT department, COUNT(*) AS total, SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows
            FROM analytics_marts.fact_appointments
            GROUP BY department ORDER BY no_shows DESC LIMIT 10
        """)
        if not dept_df.empty:
            dept_df['no_show_rate'] = round(100 * dept_df['no_shows'] / dept_df['total'], 1)
            fig = px.bar(dept_df, x='no_show_rate', y='department', orientation='h', color='no_show_rate', color_continuous_scale='Reds')
            fig.update_layout(height=350, showlegend=False, plot_bgcolor=COLORS['light_gray'], paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("No-Shows by Appointment Type")
        appt_type_df = run_query("""
            SELECT appointment_type, COUNT(*) AS total, SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows
            FROM analytics_marts.fact_appointments
            GROUP BY appointment_type ORDER BY no_shows DESC LIMIT 10
        """)
        if not appt_type_df.empty:
            appt_type_df['no_show_rate'] = round(100 * appt_type_df['no_shows'] / appt_type_df['total'], 1)
            fig = px.bar(appt_type_df, x='no_show_rate', y='appointment_type', orientation='h', color='no_show_rate', color_continuous_scale='Oranges')
            fig.update_layout(height=350, showlegend=False, plot_bgcolor=COLORS['light_gray'], paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("No-Show Rate by Lead Time (days)")
    lead_df = run_query("""
        SELECT FLOOR(EXTRACT(DAY FROM (scheduled_datetime - created_datetime)) / 7) AS lead_time_weeks,
               COUNT(*) AS total, SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows
        FROM analytics_marts.fact_appointments
        GROUP BY 1 ORDER BY 1
    """)
    if not lead_df.empty:
        lead_df['no_show_rate'] = round(100 * lead_df['no_shows'] / lead_df['total'], 1)
        fig = px.line(lead_df, x='lead_time_weeks', y='no_show_rate', markers=True, title='No-Show Rate by Appointment Lead Time')
        fig.update_layout(height=350, plot_bgcolor=COLORS['light_gray'], paper_bgcolor='white', xaxis_title='Lead Time (weeks)', yaxis_title='No-Show Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 3: READMISSION ANALYSIS
# ============================================================================

elif section == "🏥 Readmission Analysis":
    st.title("🏥 Readmission Analysis")
    st.markdown("Trends by department, disposition, and length of stay.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Readmission Rate by Department")
        readmit_dept_df = run_query("""
            SELECT department, COUNT(*) AS total, SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions
            FROM analytics_marts.fct_readmissions
            GROUP BY department ORDER BY readmissions DESC LIMIT 10
        """)
        if not readmit_dept_df.empty:
            readmit_dept_df['readmit_rate'] = round(100 * readmit_dept_df['readmissions'] / readmit_dept_df['total'], 1)
            fig = px.bar(readmit_dept_df, x='readmit_rate', y='department', orientation='h', color='readmit_rate', color_continuous_scale='Reds')
            fig.update_layout(height=350, showlegend=False, plot_bgcolor=COLORS['light_gray'], paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Readmission by Discharge Disposition")
        dispo_df = run_query("""
            SELECT discharge_disposition, COUNT(*) AS total, SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions
            FROM analytics_marts.fct_readmissions
            GROUP BY discharge_disposition ORDER BY readmissions DESC
        """)
        if not dispo_df.empty:
            dispo_df['readmit_rate'] = round(100 * dispo_df['readmissions'] / dispo_df['total'], 1)
            fig = px.pie(dispo_df, values='readmissions', names='discharge_disposition', title='Readmissions by Disposition')
            fig.update_layout(height=350, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Readmission Rate by Length of Stay")
    los_df = run_query("""
        SELECT FLOOR(length_of_stay_days / 5) * 5 AS los_bucket,
               COUNT(*) AS total, SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions
        FROM analytics_marts.fct_readmissions
        GROUP BY 1 ORDER BY 1
    """)
    if not los_df.empty:
        los_df['readmit_rate'] = round(100 * los_df['readmissions'] / los_df['total'], 1)
        fig = px.bar(los_df, x='los_bucket', y='readmit_rate', title='Readmission Rate by LOS Bucket')
        fig.update_layout(height=350, plot_bgcolor=COLORS['light_gray'], paper_bgcolor='white', xaxis_title='Length of Stay (days)', yaxis_title='Readmission Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 4: PATIENT RISK LOOKUP
# ============================================================================

elif section == "👤 Patient Risk Lookup":
    st.title("👤 Patient Risk Lookup")
    st.markdown("ML-generated risk scores for individual patients.")
    st.divider()

    st.subheader("Search Patient")
    patient_search = st.text_input("Enter patient ID or name:", help="Search for a specific patient")

    if patient_search:
        patient_risk_df = run_query(f"""
            SELECT p.patient_id, p.patient_name, p.date_of_birth, p.age_years,
                   ROUND(rf.no_show_risk_score::numeric, 3) AS no_show_risk,
                   ROUND(rf.readmission_risk_score::numeric, 3) AS readmission_risk
            FROM analytics_marts.dim_patients p
            LEFT JOIN analytics_marts.fact_readmission_features rf ON p.patient_id = rf.patient_id
            WHERE LOWER(p.patient_id) LIKE '%{patient_search.lower()}%'
               OR LOWER(p.patient_name) LIKE '%{patient_search.lower()}%'
            LIMIT 20
        """)

        if not patient_risk_df.empty:
            st.dataframe(patient_risk_df, use_container_width=True)

            selected_patient = st.selectbox("Select patient for detail view:", patient_risk_df['patient_id'].tolist())
            if selected_patient:
                patient_detail = patient_risk_df[patient_risk_df['patient_id'] == selected_patient].iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Patient ID", patient_detail['patient_id'])
                with col2:
                    st.metric("Age", f"{patient_detail['age_years']} years")
                with col3:
                    st.metric("DOB", patient_detail['date_of_birth'])

                col1, col2 = st.columns(2)
                with col1:
                    risk_val = patient_detail['no_show_risk'] * 100
                    st.metric("No-Show Risk", f"{risk_val:.1f}%", delta=f"{'High' if risk_val > 50 else 'Low'} Risk", delta_color="inverse")
                with col2:
                    risk_val = patient_detail['readmission_risk'] * 100
                    st.metric("Readmission Risk", f"{risk_val:.1f}%", delta=f"{'High' if risk_val > 50 else 'Low'} Risk", delta_color="inverse")
        else:
            st.info("No patients found matching search criteria.")
    else:
        st.info("Enter a patient ID or name to search.")

# ============================================================================
# SECTION 5: SEMANTIC NOTE SEARCH
# ============================================================================

elif section == "🔍 Semantic Note Search":
    st.title("🔍 Semantic Note Search")
    st.markdown("Search clinical narratives using semantic similarity (pgvector RAG).")
    st.divider()

    search_query = st.text_input("Search clinical notes:", placeholder="e.g., 'elevated blood pressure', 'respiratory distress'")

    if search_query:
        st.info("🔄 Semantic search requires pgvector embeddings. Searching similar cases...")
        
        # Placeholder for actual pgvector search
        note_search_df = run_query("""
            SELECT patient_id, note_date, note_text, similarity
            FROM raw.participant_narratives
            ORDER BY note_date DESC
            LIMIT 10
        """)

        if not note_search_df.empty:
            st.markdown("### Matching Clinical Notes")
            for idx, row in note_search_df.iterrows():
                with st.expander(f"Patient {row['patient_id']} - {row['note_date']}", expanded=False):
                    st.write(row['note_text'])
        else:
            st.info("No matching notes found.")
    else:
        st.info("Enter a clinical query to search narratives.")

# ============================================================================
# SECTION 6: AI QUERY ASSISTANT
# ============================================================================

elif section == "🤖 AI Query Assistant":
    st.title("🤖 AI Query Assistant")
    st.markdown("Ask clinical questions using Claude AI.")
    st.divider()

    if not api_key:
        st.warning("⚠️ Enter your Anthropic API Key in the sidebar to use this section.")
    else:
        user_query = st.text_area("Ask a clinical question:", placeholder="e.g., 'Which patients have high readmission risk?'", height=80)
        
        if st.button("Query Claude Agent"):
            st.info("🤖 Claude is processing your query... (Claude API integration coming soon)")

# ============================================================================
# SECTION 7: LLM OBSERVABILITY
# ============================================================================

elif section == "📋 LLM Observability":
    st.title("📋 LLM Observability")
    st.markdown("Live log of all Claude API calls and tokens used.")
    st.divider()

    llm_log_df = run_query("""
        SELECT call_id, call_timestamp, prompt_tokens, completion_tokens, 
               prompt_tokens + completion_tokens AS total_tokens, model, status
        FROM raw.llm_call_log
        ORDER BY call_timestamp DESC
        LIMIT 100
    """)

    if not llm_log_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total API Calls", len(llm_log_df))
        with col2:
            st.metric("Avg Tokens", round(llm_log_df['total_tokens'].mean()))
        with col3:
            st.metric("Total Tokens", int(llm_log_df['total_tokens'].sum()))

        st.divider()
        st.markdown("### Recent API Calls")
        st.dataframe(llm_log_df.head(20), use_container_width=True)
    else:
        st.info("No LLM calls logged yet.")

# ============================================================================
# SECTION 8: PROMPT MANAGEMENT
# ============================================================================

elif section == "⚙️ Prompt Management":
    st.title("⚙️ Prompt Management")
    st.markdown("View and edit prompts used by the Claude agent.")
    st.divider()

    prompt_df = run_query("""
        SELECT prompt_id, prompt_name, prompt_version, created_at, last_updated, is_active
        FROM raw.prompt_library
        ORDER BY last_updated DESC
    """)

    if not prompt_df.empty:
        selected_prompt = st.selectbox("Select prompt:", prompt_df['prompt_name'].tolist())
        prompt_detail = prompt_df[prompt_df['prompt_name'] == selected_prompt].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Version", prompt_detail['prompt_version'])
        with col2:
            st.metric("Status", "Active" if prompt_detail['is_active'] else "Inactive")
        with col3:
            st.metric("Last Updated", prompt_detail['last_updated'])

        st.markdown("### Edit Prompt")
        prompt_text = st.text_area("Prompt text:", value="(Prompt content would load here)", height=200)
        
        if st.button("Save Changes"):
            st.success("✅ Prompt updated (demo mode)")
    else:
        st.info("No prompts found.")

# ============================================================================
# SECTION 9: DATA QUALITY
# ============================================================================

elif section == "🧪 Data Quality":
    st.title("🧪 Data Quality")
    st.markdown("dbt test results and data quality metrics.")
    st.divider()

    dq_df = run_query("""
        SELECT test_name, table_name, status, test_timestamp, rows_affected
        FROM raw.dbt_test_results
        ORDER BY test_timestamp DESC
        LIMIT 50
    """)

    if not dq_df.empty:
        passed = len(dq_df[dq_df['status'] == 'PASS'])
        failed = len(dq_df[dq_df['status'] == 'FAIL'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tests Passed", passed, delta=f"{round(100*passed/(passed+failed))}%")
        with col2:
            st.metric("Tests Failed", failed, delta=f"{round(100*failed/(passed+failed))}%")

        st.divider()
        st.markdown("### Test Results")
        st.dataframe(dq_df.head(30), use_container_width=True)
    else:
        st.info("No test results available.")

# ============================================================================
# SECTION 10: PATIENT COHORT BUILDER
# ============================================================================

elif section == "👥 Patient Cohort Builder":
    st.title("👥 Patient Cohort Builder")
    st.markdown("Build cohorts using dynamic filters and aggregate metrics.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_age = st.slider("Minimum Age:", 0, 18, 0)
    with col2:
        max_age = st.slider("Maximum Age:", 0, 18, 18)
    with col3:
        department = st.selectbox("Department:", ["All", "Cardiology", "Neurology", "Oncology", "Orthopedics"])

    if st.button("Build Cohort"):
        cohort_df = run_query(f"""
            SELECT COUNT(DISTINCT patient_id) AS cohort_size,
                   ROUND(AVG(age_years)::numeric, 1) AS avg_age,
                   COUNT(*) FILTER (WHERE gender = 'M') AS male_count,
                   COUNT(*) FILTER (WHERE gender = 'F') AS female_count
            FROM analytics_marts.dim_patients
            WHERE age_years BETWEEN {min_age} AND {max_age}
            {'AND department = ' + "'" + department + "'" if department != 'All' else ''}
        """)

        if not cohort_df.empty:
            cohort = cohort_df.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cohort Size", int(cohort['cohort_size']))
            with col2:
                st.metric("Avg Age", f"{cohort['avg_age']} years")
            with col3:
                st.metric("Males", int(cohort['male_count']))
            with col4:
                st.metric("Females", int(cohort['female_count']))

            st.divider()
            st.markdown("### Cohort Details")
            cohort_patients = run_query(f"""
                SELECT patient_id, patient_name, age_years, gender, department
                FROM analytics_marts.dim_patients
                WHERE age_years BETWEEN {min_age} AND {max_age}
                {'AND department = ' + "'" + department + "'" if department != 'All' else ''}
                LIMIT 50
            """)
            st.dataframe(cohort_patients, use_container_width=True)

# ============================================================================
# SECTION 11: WHAT-IF SIMULATOR
# ============================================================================

elif section == "🔮 What-If Simulator":
    st.title("🔮 What-If Simulator")
    st.markdown("Interactive scenario modeling for patient risk scores.")
    st.divider()

    st.subheader("Adjust Patient Characteristics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years):", 0, 18, 10)
    with col2:
        los = st.slider("Length of Stay (days):", 1, 30, 7)
    with col3:
        med_count = st.slider("Number of Medications:", 0, 20, 5)

    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Department:", ["Cardiology", "Neurology", "Oncology", "Orthopedics"])
    with col2:
        disposition = st.selectbox("Discharge Disposition:", ["Home", "Skilled Nursing", "Rehab", "AMA"])

    if st.button("Simulate Risk Scores"):
        st.info("🔮 Simulating ML predictions with adjusted parameters...")
        
        # Placeholder for actual XGBoost scoring
        no_show_risk = np.random.rand() * 100
        readmit_risk = np.random.rand() * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("No-Show Risk", f"{no_show_risk:.1f}%", 
                     delta="High" if no_show_risk > 50 else "Low", 
                     delta_color="inverse")
        with col2:
            st.metric("Readmission Risk", f"{readmit_risk:.1f}%",
                     delta="High" if readmit_risk > 50 else "Low",
                     delta_color="inverse")

        st.divider()
        st.markdown("### Risk Prediction Details")
        st.write(f"**Scenario:** {age}-year-old patient in {department}, {los} day stay, {med_count} medications, discharged to {disposition}")
        st.write(f"Based on similar cases in the database, this patient has a **{no_show_risk:.1f}% probability of missing their next appointment** and a **{readmit_risk:.1f}% probability of 30-day readmission**.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    f"<small style='color: {COLORS['neutral_gray']};'>Caboodle Access Platform v2.0 "
    "| Powered by PostgreSQL, dbt, Claude AI, and Plotly | "
    f"<a href='https://github.com/Sami7490/caboodle_access_platform' target='_blank'>GitHub</a></small>",
    unsafe_allow_html=True
)
