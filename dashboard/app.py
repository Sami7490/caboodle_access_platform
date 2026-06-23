"""
app.py

Streamlit dashboard (our local front-end display tool -- renders
charts, tables, and interactive UI components as a web app running
entirely on your Mac, nothing hosted externally) for the Caboodle
Access Platform.

Sections:
  1. Key Metrics    -- no-show rate, readmission rate, patient days
  2. No-Show Analysis -- trends by department, appointment type, lead time
  3. Readmission Analysis -- trends by department, disposition, LOS
  4. Patient Risk Lookup -- ML-generated risk scores per patient
  5. Semantic Search -- pgvector RAG search over clinical notes
  6. AI Query Assistant -- natural language queries via the Claude agent
  7. LLM Observability -- live log of all Claude API calls
"""

import os
import sys
import json
import psycopg2
import pandas as pd
import plotly.express as px
import streamlit as st

# Add agent/ folder to path so we can import our agent modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agent'))

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Caboodle Access Platform",
    page_icon="🏥",
    layout="wide",
)

# ----------------------------------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------------------------------

@st.cache_resource  # cache_resource keeps the connection alive across
                    # reruns rather than reconnecting on every interaction
def get_connection():
    return psycopg2.connect(
        host="localhost", port=5432,
        dbname="caboodle_access", user="postgres"
    )

@st.cache_data(ttl=300)  # cache_data caches query results for 5 minutes
                          # (ttl=300 seconds) so the dashboard doesn't
                          # re-query Postgres on every single interaction
def run_query(sql):
    conn = get_connection()
    return pd.read_sql(sql, conn)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=60)
st.sidebar.title("Caboodle Access Platform")
st.sidebar.markdown("*Pediatric Clinical Analytics*")
st.sidebar.divider()

api_key = st.sidebar.text_input(
    "Anthropic API Key",
    type="password",
    help="Required for the AI Query Assistant section"
)

st.sidebar.divider()
section = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 Key Metrics",
        "📅 No-Show Analysis",
        "🏥 Readmission Analysis",
        "👤 Patient Risk Lookup",
        "🔍 Semantic Note Search",
        "🤖 AI Query Assistant",
        "📋 LLM Observability",
        "⚙️ Prompt Management",
    ]
)

# ----------------------------------------------------------------------------
# SECTION 1: KEY METRICS
# ----------------------------------------------------------------------------

if section == "📊 Key Metrics":
    st.title("📊 Key Metrics")
    st.markdown("High-level summary of access and utilization across the platform.")

    # Pull summary stats
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

    # Display as metric cards across the top row.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", f"{patient_df['total'][0]:,}")
    with col2:
        no_show_rate = round(100 * appt_df['no_shows'][0] / appt_df['total'][0], 1)
        st.metric("No-Show Rate", f"{no_show_rate}%",
                  delta=f"{appt_df['no_shows'][0]:,} of {appt_df['total'][0]:,} appointments",
                  delta_color="inverse")
    with col3:
        readmit_rate = round(100 * readmit_df['readmissions'][0] / readmit_df['total'][0], 1)
        st.metric("30-Day Readmission Rate", f"{readmit_rate}%",
                  delta=f"{readmit_df['readmissions'][0]:,} of {readmit_df['total'][0]:,} inpatient",
                  delta_color="inverse")
    with col4:
        st.metric("Total Patient Days", f"{days_df['patient_days'][0]:,}")

    st.divider()

    # Monthly appointment volume trend
    st.subheader("Monthly Appointment Volume")
    monthly_df = run_query("""
        SELECT
            DATE_TRUNC('month', scheduled_datetime) AS month,
            COUNT(*) AS total,
            SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows
        FROM analytics_marts.fact_appointments
        GROUP BY 1 ORDER BY 1
    """)
    monthly_df['month'] = pd.to_datetime(monthly_df['month'])
    fig = px.line(monthly_df, x='month', y=['total', 'no_shows'],
                  labels={'value': 'Appointments', 'month': 'Month'},
                  title="Monthly Appointments vs No-Shows")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# SECTION 2: NO-SHOW ANALYSIS
# ----------------------------------------------------------------------------

elif section == "📅 No-Show Analysis":
    st.title("📅 No-Show Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("No-Show Rate by Department")
        dept_df = run_query("""
            SELECT d.department_name,
                   COUNT(*) AS total,
                   SUM(CASE WHEN fa.is_no_show THEN 1 ELSE 0 END) AS no_shows,
                   ROUND(100.0 * SUM(CASE WHEN fa.is_no_show THEN 1 ELSE 0 END) / COUNT(*), 1)
                       AS no_show_rate
            FROM analytics_marts.fact_appointments fa
            JOIN analytics_staging.stg_departments d ON d.department_key = fa.department_key
            GROUP BY d.department_name ORDER BY no_show_rate DESC
        """)
        fig = px.bar(dept_df, x='no_show_rate', y='department_name',
                     orientation='h', color='no_show_rate',
                     color_continuous_scale='Reds',
                     labels={'no_show_rate': 'No-Show Rate (%)', 'department_name': ''})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("No-Show Rate by Appointment Type")
        type_df = run_query("""
            SELECT appointment_type,
                   COUNT(*) AS total,
                   ROUND(100.0 * SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) / COUNT(*), 1)
                       AS no_show_rate
            FROM analytics_marts.fact_appointments
            GROUP BY appointment_type ORDER BY no_show_rate DESC
        """)
        fig = px.bar(type_df, x='appointment_type', y='no_show_rate',
                     color='no_show_rate', color_continuous_scale='Oranges',
                     labels={'no_show_rate': 'No-Show Rate (%)', 'appointment_type': 'Appointment Type'})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("No-Show Rate by Lead Time (Days Before Appointment)")
    lead_df = run_query("""
        SELECT
            WIDTH_BUCKET(lead_time_days, 0, 100, 10) AS bucket,
            ROUND(AVG(lead_time_days)) AS avg_lead_days,
            ROUND(100.0 * SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) / COUNT(*), 1)
                AS no_show_rate
        FROM analytics_marts.fact_appointments
        GROUP BY bucket ORDER BY bucket
    """)
    fig = px.line(lead_df, x='avg_lead_days', y='no_show_rate',
                  markers=True,
                  labels={'avg_lead_days': 'Avg Lead Time (Days)', 'no_show_rate': 'No-Show Rate (%)'},
                  title="No-Show Rate vs Appointment Lead Time")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# SECTION 3: READMISSION ANALYSIS
# ----------------------------------------------------------------------------

elif section == "🏥 Readmission Analysis":
    st.title("🏥 Readmission Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Readmission Rate by Discharge Disposition")
        disp_df = run_query("""
            SELECT discharge_disposition,
                   COUNT(*) AS total,
                   SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions,
                   ROUND(100.0 * SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) / COUNT(*), 1)
                       AS readmit_rate
            FROM analytics_marts.fct_readmissions
            GROUP BY discharge_disposition ORDER BY readmit_rate DESC
        """)
        fig = px.bar(disp_df, x='discharge_disposition', y='readmit_rate',
                     color='readmit_rate', color_continuous_scale='Reds',
                     labels={'readmit_rate': 'Readmission Rate (%)', 'discharge_disposition': 'Disposition'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Length of Stay Distribution")
        los_df = run_query("""
            SELECT length_of_stay_days,
                   is_30_day_readmission
            FROM analytics_marts.fct_readmissions
            WHERE length_of_stay_days < 30
        """)
        fig = px.histogram(los_df, x='length_of_stay_days',
                           color='is_30_day_readmission',
                           barmode='overlay',
                           labels={'length_of_stay_days': 'Length of Stay (Days)',
                                   'is_30_day_readmission': 'Readmitted'},
                           title="LOS Distribution: Readmitted vs Not")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Readmission Trend")
    trend_df = run_query("""
        SELECT
            DATE_TRUNC('month', admission_datetime) AS month,
            COUNT(*) AS total_inpatient,
            SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions
        FROM analytics_marts.fct_readmissions
        GROUP BY 1 ORDER BY 1
    """)
    trend_df['month'] = pd.to_datetime(trend_df['month'])
    trend_df['readmit_rate'] = round(100 * trend_df['readmissions'] / trend_df['total_inpatient'], 1)
    fig = px.line(trend_df, x='month', y='readmit_rate', markers=True,
                  labels={'readmit_rate': 'Readmission Rate (%)', 'month': 'Month'},
                  title="Monthly 30-Day Readmission Rate")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# SECTION 4: PATIENT RISK LOOKUP
# ----------------------------------------------------------------------------

elif section == "👤 Patient Risk Lookup":
    st.title("👤 Patient Risk Lookup")
    st.markdown("Enter a patient key to retrieve ML-generated risk scores.")

    patient_key = st.number_input("Patient Key", min_value=1, max_value=500, value=1, step=1)

    if st.button("Get Risk Scores"):
        from sql_agent import get_patient_risk
        with st.spinner("Scoring patient risk..."):
            result = json.loads(get_patient_risk(patient_key))

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if "noshow_probability" in result:
                    prob = result["noshow_probability"]
                    risk = result["noshow_risk"]
                    color = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
                    st.metric("No-Show Probability", f"{prob:.1%}")
                    st.markdown(f"**Risk Level:** {color} {risk}")
                else:
                    st.info("No appointment history for this patient.")

            with col2:
                if "readmission_probability" in result:
                    prob = result["readmission_probability"]
                    risk = result["readmission_risk"]
                    color = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
                    st.metric("Readmission Probability", f"{prob:.1%}")
                    st.markdown(f"**Risk Level:** {color} {risk}")
                else:
                    st.info("No inpatient history for this patient.")

        # Show patient details
        patient_df = run_query(f"""
            SELECT first_name, last_name, date_of_birth, age_years, sex, race, primary_language
            FROM analytics_marts.dim_patients WHERE patient_key = {patient_key}
        """)
        if not patient_df.empty:
            st.subheader("Patient Demographics")
            st.dataframe(patient_df, use_container_width=True)

# ----------------------------------------------------------------------------
# SECTION 5: SEMANTIC NOTE SEARCH
# ----------------------------------------------------------------------------

elif section == "🔍 Semantic Note Search":
    st.title("🔍 Semantic Clinical Note Search")
    st.markdown(
        "Search clinical notes by meaning rather than exact keywords. "
        "Powered by pgvector (our Postgres vector extension) and a local "
        "embedding model running on your Mac."
    )

    query = st.text_input(
        "Search query",
        placeholder="e.g. respiratory distress, fever and infection, post-surgical complications"
    )
    limit = st.slider("Number of results", min_value=3, max_value=20, value=5)

    if st.button("Search") and query:
        from sql_agent import search_similar_notes
        with st.spinner("Searching notes..."):
            results = json.loads(search_similar_notes(query, limit))

        if "error" in results:
            st.error(results["error"])
        else:
            st.success(f"Found {len(results['results'])} similar notes")
            for r in results["results"]:
                with st.expander(
                    f"Note {r['note_id']} — Patient {r['patient_key']} "
                    f"({r['encounter_type']}) — Similarity: {r['similarity_score']:.3f}"
                ):
                    st.write(r["note_text"])

# ----------------------------------------------------------------------------
# SECTION 6: AI QUERY ASSISTANT
# ----------------------------------------------------------------------------

elif section == "🤖 AI Query Assistant":
    st.title("🤖 AI Query Assistant")
    st.markdown(
        "Ask any question about the data in plain English. The Claude agent "
        "will choose the right tools (database queries, risk scores, semantic "
        "search) to answer your question."
    )

    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to use this feature.")
    else:
        # Keep conversation history in session state so the chat scrolls
        # as new messages are added.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display existing chat messages.
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input box at the bottom.
        if prompt := st.chat_input("Ask a question about patients, appointments, or readmissions..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    from sql_agent import run_agent
                    answer = run_agent(prompt, api_key)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# ----------------------------------------------------------------------------
# SECTION 7: PROMPT MANAGEMENT
# ----------------------------------------------------------------------------

elif section == "⚙️ Prompt Management":
    st.title("⚙️ Prompt Management")
    st.markdown(
        "Edit and version the Claude agent's system prompt directly from the "
        "dashboard -- no code changes or restarts needed. Every change is versioned "
        "in raw.prompt_library (our prompt management table in Postgres)."
    )

    # Show current active prompt
    current_df = run_query("""
        SELECT prompt_name, version, prompt_text, updated_at, notes
        FROM raw.prompt_library
        WHERE is_active = TRUE
        ORDER BY prompt_name
    """)

    st.subheader("Active Prompts")
    for _, row in current_df.iterrows():
        with st.expander(f"{row['prompt_name']} (v{row['version']}) — last updated {str(row['updated_at'])[:16]}"):
            st.text_area("Current prompt text", value=row['prompt_text'], height=150, disabled=True,
                        key=f"current_{row['prompt_name']}")
            if row['notes']:
                st.caption(f"Notes: {row['notes']}")

    st.divider()
    st.subheader("Edit a Prompt")

    prompt_names = current_df['prompt_name'].tolist()
    selected = st.selectbox("Select prompt to edit", prompt_names)

    if selected:
        current_row = current_df[current_df['prompt_name'] == selected].iloc[0]
        new_text = st.text_area(
            "New prompt text",
            value=current_row['prompt_text'],
            height=200,
            key="edit_prompt"
        )
        edit_notes = st.text_input("Notes (why are you changing this?)", placeholder="e.g. Added instruction to always cite table names")

        if st.button("Save new version"):
            conn = get_connection()
            cur = conn.cursor()
            # Deactivate current version
            cur.execute(
                "UPDATE raw.prompt_library SET is_active = FALSE WHERE prompt_name = %s",
                (selected,)
            )
            # Insert new version
            cur.execute(
                """INSERT INTO raw.prompt_library
                   (prompt_name, prompt_text, version, is_active, notes)
                   VALUES (%s, %s, %s, TRUE, %s)""",
                (selected, new_text, int(current_row['version']) + 1, edit_notes)
            )
            conn.commit()
            cur.close()
            st.success(f"Saved v{int(current_row['version']) + 1} of '{selected}'. Agent will use new prompt on next query.")
            st.cache_data.clear()

    st.divider()
    st.subheader("Version History")
    history_df = run_query("""
        SELECT prompt_name, version, is_active, updated_at, notes,
               LEFT(prompt_text, 80) AS prompt_preview
        FROM raw.prompt_library
        ORDER BY prompt_name, version DESC
    """)
    st.dataframe(history_df, use_container_width=True)


# ----------------------------------------------------------------------------
# SECTION 8: LLM OBSERVABILITY
# ----------------------------------------------------------------------------

elif section == "📋 LLM Observability":
    st.title("📋 LLM Observability")
    st.markdown(
        "Live log of every Claude API call made by the agent. "
        "Tracks prompt, response, token usage, latency, and success rate."
    )

    log_df = run_query("""
        SELECT log_id, called_at, tool_name, model,
               input_tokens, output_tokens, latency_ms, success,
               LEFT(prompt_text, 100) AS prompt_preview
        FROM raw.llm_call_log
        ORDER BY called_at DESC
        LIMIT 50
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total API Calls", len(log_df))
    with col2:
        avg_latency = round(log_df['latency_ms'].mean(), 0) if not log_df.empty else 0
        st.metric("Avg Latency", f"{avg_latency}ms")
    with col3:
        success_rate = round(100 * log_df['success'].sum() / len(log_df), 1) if not log_df.empty else 0
        st.metric("Success Rate", f"{success_rate}%")

    st.subheader("Recent API Calls")
    st.dataframe(log_df, use_container_width=True)

    if not log_df.empty:
        st.subheader("Token Usage Over Time")
        fig = px.bar(log_df.sort_values('called_at'), x='called_at',
                     y=['input_tokens', 'output_tokens'],
                     labels={'value': 'Tokens', 'called_at': 'Time'},
                     title="Input vs Output Tokens Per Call")
        st.plotly_chart(fig, use_container_width=True)
