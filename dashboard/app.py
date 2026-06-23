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
        "🧪 Data Quality",
        "👥 Patient Cohort Builder",
        "🔮 What-If Simulator",
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
# SECTION 7: WHAT-IF SIMULATOR
# ----------------------------------------------------------------------------

elif section == "🔮 What-If Simulator":
    st.title("🔮 What-If Simulator")
    st.markdown(
        "Adjust hypothetical patient and appointment parameters and see how "
        "the ML model's predicted risk scores change in real time. Built on "
        "the same scikit-learn models trained in Phase 2."
    )

    tab1, tab2 = st.tabs(["No-Show Risk Simulator", "Readmission Risk Simulator"])

    # ----------------------------------------------------------------
    # TAB 1: NO-SHOW RISK SIMULATOR
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("No-Show Risk Simulator")
        st.markdown("Adjust appointment parameters to see how predicted no-show probability changes.")

        col1, col2 = st.columns(2)

        with col1:
            lead_time = st.slider(
                "Lead time (days between booking and appointment)",
                min_value=1, max_value=90, value=14,
                help="Longer lead times are associated with higher no-show risk"
            )
            appt_type = st.selectbox(
                "Appointment type",
                ["New Patient", "Follow-up", "Annual Wellness", "Procedure", "Therapy Session"]
            )
            day_of_week = st.selectbox(
                "Day of week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )

        with col2:
            is_weekend = day_of_week in ["Saturday", "Sunday"]
            prior_noshow = st.slider(
                "Patient's prior no-show count",
                min_value=0, max_value=10, value=0,
                help="How many times has this patient previously no-showed?"
            )
            prior_appts = st.slider(
                "Patient's total prior appointments",
                min_value=0, max_value=20, value=5
            )

        # Score the hypothetical appointment using our trained no-show model.
        try:
            import joblib, pandas as pd
            noshow_model   = joblib.load("agent/noshow_model.pkl")
            noshow_scaler  = joblib.load("agent/noshow_scaler.pkl")
            noshow_features = joblib.load("agent/noshow_features.pkl")

            # Build a single-row feature dataframe matching the training schema.
            input_df = pd.DataFrame([{
                "lead_time_days": float(lead_time),
                "is_weekend": is_weekend,
                "prior_noshow_count": float(prior_noshow),
                "prior_appt_count": float(prior_appts),
                "appointment_type": appt_type,
                "day_of_week": day_of_week,
            }])
            input_df = pd.get_dummies(input_df, columns=["appointment_type", "day_of_week"], drop_first=True)
            for col in noshow_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[noshow_features].fillna(0)
            scaled = noshow_scaler.transform(input_df)
            prob = float(noshow_model.predict_proba(scaled)[0][1])

            # Display the result prominently.
            st.divider()
            risk_level = "🔴 High" if prob > 0.4 else "🟡 Medium" if prob > 0.2 else "🟢 Low"
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted No-Show Probability", f"{prob:.1%}")
            with col2:
                st.metric("Risk Level", risk_level)

            # Show a gauge-style progress bar.
            st.progress(min(prob, 1.0), text=f"No-show probability: {prob:.1%}")

            # Key drivers narrative
            st.markdown("**Key factors in this prediction:**")
            drivers = []
            if lead_time > 30:
                drivers.append(f"• Long lead time ({lead_time} days) increases no-show risk")
            if prior_noshow > 0:
                drivers.append(f"• Prior no-show history ({prior_noshow} previous no-shows) is a strong predictor")
            if is_weekend:
                drivers.append("• Weekend appointments have higher no-show rates")
            if appt_type == "New Patient":
                drivers.append("• New patient appointments tend to have higher no-show rates")
            if not drivers:
                drivers.append("• No strong risk factors identified for this combination")
            for d in drivers:
                st.markdown(d)

        except Exception as e:
            st.error(f"Model scoring error: {e}")

    # ----------------------------------------------------------------
    # TAB 2: READMISSION RISK SIMULATOR
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("Readmission Risk Simulator")
        st.markdown("Adjust inpatient encounter parameters to see predicted 30-day readmission probability.")

        col1, col2 = st.columns(2)

        with col1:
            los = st.slider(
                "Length of stay (days)",
                min_value=1, max_value=30, value=3,
                help="Longer stays often indicate higher acuity and readmission risk"
            )
            disposition = st.selectbox(
                "Discharge disposition",
                ["Home", "Home Health", "SNF", "AMA", "Expired"]
            )

        with col2:
            age = st.slider("Patient age (years)", min_value=0, max_value=18, value=8)
            prior_admissions = st.slider(
                "Prior inpatient admissions",
                min_value=0, max_value=10, value=0,
                help="Previous admissions are a strong predictor of readmission"
            )
            dept_key = st.selectbox(
                "Department",
                [6, 7],
                format_func=lambda x: "Pediatric Inpatient Unit" if x == 6 else "Pediatric ICU"
            )

        # Score the hypothetical encounter using our trained readmission model.
        try:
            readmit_model    = joblib.load("agent/readmission_model.pkl")
            readmit_scaler   = joblib.load("agent/readmission_scaler.pkl")
            readmit_features = joblib.load("agent/readmission_features.pkl")

            input_df = pd.DataFrame([{
                "length_of_stay_days": float(los),
                "age_years": float(age),
                "prior_admission_count": float(prior_admissions),
                f"department_key_{dept_key}": 1,
                "discharge_disposition": disposition,
            }])
            input_df = pd.get_dummies(input_df, columns=["discharge_disposition"], drop_first=True)
            for col in readmit_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[readmit_features].fillna(0)
            scaled = readmit_scaler.transform(input_df)
            prob = float(readmit_model.predict_proba(scaled)[0][1])

            st.divider()
            risk_level = "🔴 High" if prob > 0.4 else "🟡 Medium" if prob > 0.2 else "🟢 Low"
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Readmission Probability", f"{prob:.1%}")
            with col2:
                st.metric("Risk Level", risk_level)

            st.progress(min(prob, 1.0), text=f"Readmission probability: {prob:.1%}")

            st.markdown("**Key factors in this prediction:**")
            drivers = []
            if los > 5:
                drivers.append(f"• Long length of stay ({los} days) suggests higher acuity")
            if disposition in ["AMA", "SNF"]:
                drivers.append(f"• Discharge to {disposition} is associated with higher readmission risk")
            if prior_admissions > 1:
                drivers.append(f"• {prior_admissions} prior admissions indicates a high-utilization patient")
            if dept_key == 7:
                drivers.append("• ICU discharge carries elevated readmission risk")
            if not drivers:
                drivers.append("• No strong risk factors identified for this combination")
            for d in drivers:
                st.markdown(d)

        except Exception as e:
            st.error(f"Model scoring error: {e}")


# ----------------------------------------------------------------------------
# SECTION 8: PATIENT COHORT BUILDER
# ----------------------------------------------------------------------------

elif section == "👥 Patient Cohort Builder":
    st.title("👥 Patient Cohort Builder")
    st.markdown(
        "Filter patients by demographics and clinical characteristics to see "
        "aggregate metrics for that cohort. Useful for identifying high-risk "
        "subpopulations and targeting interventions."
    )

    # --- FILTERS ---
    st.subheader("Define Cohort")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_range = st.slider("Age range (years)", 0, 18, (0, 18))
        sex_filter = st.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])

    with col2:
        language_filter = st.multiselect(
            "Primary language",
            ["English", "Spanish", "Other"],
            default=["English", "Spanish", "Other"]
        )
        race_options = run_query("SELECT DISTINCT race FROM analytics_marts.dim_patients ORDER BY race")
        race_filter = st.multiselect(
            "Race",
            race_options['race'].tolist(),
            default=race_options['race'].tolist()
        )

    with col3:
        dept_options = run_query("SELECT DISTINCT department_name FROM analytics_staging.stg_departments ORDER BY department_name")
        dept_filter = st.multiselect(
            "Department (appointment history)",
            dept_options['department_name'].tolist(),
            default=dept_options['department_name'].tolist()
        )

    # Build the cohort query dynamically from the selected filters.
    # We use Python to construct the WHERE clause based on what the
    # user selected, then pass it to Postgres via run_query().
    sex_list = "', '".join(sex_filter) if sex_filter else "''"
    lang_list = "', '".join(language_filter) if language_filter else "''"
    race_list = "', '".join(race_filter) if race_filter else "''"
    dept_list = "', '".join(dept_filter) if dept_filter else "''"

    cohort_query = f"""
        SELECT DISTINCT p.patient_key
        FROM analytics_marts.dim_patients p
        JOIN analytics_marts.fact_appointments fa ON fa.patient_key = p.patient_key
        JOIN analytics_staging.stg_departments d ON d.department_key = fa.department_key
        WHERE p.age_years BETWEEN {age_range[0]} AND {age_range[1]}
          AND p.sex IN ('{sex_list}')
          AND p.primary_language IN ('{lang_list}')
          AND p.race IN ('{race_list}')
          AND d.department_name IN ('{dept_list}')
    """

    cohort_df = run_query(cohort_query)
    cohort_size = len(cohort_df)

    st.divider()
    st.subheader(f"Cohort Results — {cohort_size:,} patients matched")

    if cohort_size == 0:
        st.warning("No patients match the selected filters. Try broadening your criteria.")
    else:
        cohort_keys = tuple(cohort_df['patient_key'].tolist())
        # Handle single-item tuple formatting for SQL IN clause.
        keys_sql = f"({cohort_keys[0]})" if len(cohort_keys) == 1 else str(cohort_keys)

        col1, col2, col3, col4 = st.columns(4)

        # No-show rate for this cohort
        noshow_df = run_query(f"""
            SELECT
                COUNT(*) AS total_appts,
                SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows,
                ROUND(100.0 * SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) / COUNT(*), 1)
                    AS no_show_rate
            FROM analytics_marts.fact_appointments
            WHERE patient_key IN {keys_sql}
        """)

        # Readmission rate for this cohort
        readmit_df = run_query(f"""
            SELECT
                COUNT(*) AS total_inpatient,
                SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions,
                ROUND(100.0 * SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1)
                    AS readmit_rate
            FROM analytics_marts.fct_readmissions
            WHERE patient_key IN {keys_sql}
        """)

        # Patient days for this cohort
        days_df = run_query(f"""
            SELECT ROUND(SUM(length_of_stay_days)::numeric, 1) AS patient_days
            FROM analytics_marts.fact_encounters
            WHERE encounter_type = 'Inpatient' AND patient_key IN {keys_sql}
        """)

        with col1:
            st.metric("Cohort Size", f"{cohort_size:,} patients")
        with col2:
            rate = noshow_df['no_show_rate'][0] if not noshow_df.empty else 0
            st.metric("No-Show Rate", f"{rate}%")
        with col3:
            rate = readmit_df['readmit_rate'][0] if not readmit_df.empty else 0
            st.metric("Readmission Rate", f"{rate}%")
        with col4:
            days = days_df['patient_days'][0] if not days_df.empty else 0
            st.metric("Patient Days", f"{days:,}")

        st.divider()

        # Department breakdown for this cohort
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("No-Show Rate by Department")
            dept_df = run_query(f"""
                SELECT d.department_name,
                       COUNT(*) AS total,
                       ROUND(100.0 * SUM(CASE WHEN fa.is_no_show THEN 1 ELSE 0 END) / COUNT(*), 1)
                           AS no_show_rate
                FROM analytics_marts.fact_appointments fa
                JOIN analytics_staging.stg_departments d ON d.department_key = fa.department_key
                WHERE fa.patient_key IN {keys_sql}
                GROUP BY d.department_name
                ORDER BY no_show_rate DESC
            """)
            fig = px.bar(dept_df, x='no_show_rate', y='department_name',
                         orientation='h', color='no_show_rate',
                         color_continuous_scale='Reds',
                         labels={'no_show_rate': 'No-Show Rate (%)', 'department_name': ''})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Age Distribution")
            age_df = run_query(f"""
                SELECT age_years, COUNT(*) AS count
                FROM analytics_marts.dim_patients
                WHERE patient_key IN {keys_sql}
                GROUP BY age_years ORDER BY age_years
            """)
            fig = px.bar(age_df, x='age_years', y='count',
                         labels={'age_years': 'Age (years)', 'count': 'Patients'},
                         title="Age Distribution of Cohort")
            st.plotly_chart(fig, use_container_width=True)

        # Patient list for this cohort
        st.subheader("Cohort Patient List")
        patient_list_df = run_query(f"""
            SELECT p.patient_key, p.first_name, p.last_name, p.age_years,
                   p.sex, p.race, p.primary_language,
                   COUNT(DISTINCT fa.appointment_key) AS total_appts,
                   SUM(CASE WHEN fa.is_no_show THEN 1 ELSE 0 END) AS no_shows
            FROM analytics_marts.dim_patients p
            LEFT JOIN analytics_marts.fact_appointments fa ON fa.patient_key = p.patient_key
            WHERE p.patient_key IN {keys_sql}
            GROUP BY p.patient_key, p.first_name, p.last_name, p.age_years,
                     p.sex, p.race, p.primary_language
            ORDER BY no_shows DESC
            LIMIT 50
        """)
        st.dataframe(patient_list_df, use_container_width=True)


# ----------------------------------------------------------------------------
# SECTION 8: DATA QUALITY MONITORING
# ----------------------------------------------------------------------------

elif section == "🧪 Data Quality":
    st.title("🧪 Data Quality Monitoring")
    st.markdown(
        "Tracks dbt test results over time — every time `run_dbt_tests.py` "
        "runs, results are logged to `raw.dbt_test_results`. This gives a "
        "trend view of data quality rather than a single point-in-time snapshot."
    )

    # Summary metrics from latest run
    latest_df = run_query("""
        SELECT status, COUNT(*) AS count
        FROM raw.dbt_test_results
        WHERE run_at = (SELECT MAX(run_at) FROM raw.dbt_test_results)
        GROUP BY status
    """)

    col1, col2, col3 = st.columns(3)
    passed = int(latest_df[latest_df['status'] == 'pass']['count'].sum()) if not latest_df.empty else 0
    failed = int(latest_df[latest_df['status'] == 'fail']['count'].sum()) if not latest_df.empty else 0
    warned = int(latest_df[latest_df['status'] == 'warn']['count'].sum()) if not latest_df.empty else 0

    with col1:
        st.metric("✅ Passing", passed)
    with col2:
        st.metric("❌ Failing", failed, delta=None if failed == 0 else f"{failed} failures", delta_color="inverse")
    with col3:
        st.metric("⚠️ Warnings", warned)

    st.divider()

    # Latest run results
    st.subheader("Latest Run Results")
    results_df = run_query("""
        SELECT test_name, status, failures, run_at
        FROM raw.dbt_test_results
        WHERE run_at = (SELECT MAX(run_at) FROM raw.dbt_test_results)
        ORDER BY status DESC, test_name
    """)
    st.dataframe(results_df, use_container_width=True)

    st.divider()

    # Pass rate trend over time
    st.subheader("Pass Rate Trend Over Time")
    trend_df = run_query("""
        SELECT
            run_at,
            COUNT(*) AS total_tests,
            SUM(CASE WHEN status = 'pass' THEN 1 ELSE 0 END) AS passed,
            ROUND(100.0 * SUM(CASE WHEN status = 'pass' THEN 1 ELSE 0 END) / COUNT(*), 1)
                AS pass_rate_pct
        FROM raw.dbt_test_results
        GROUP BY run_at
        ORDER BY run_at
    """)
    if len(trend_df) > 1:
        fig = px.line(trend_df, x='run_at', y='pass_rate_pct',
                      markers=True,
                      labels={'pass_rate_pct': 'Pass Rate (%)', 'run_at': 'Run Time'},
                      title="dbt Test Pass Rate Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python3 airflow/run_dbt_tests.py` a few more times to see a trend chart here.")

    st.divider()

    # Full history
    st.subheader("Full Test History")
    history_df = run_query("""
        SELECT test_name, status, failures, run_at
        FROM raw.dbt_test_results
        ORDER BY run_at DESC
        LIMIT 100
    """)
    st.dataframe(history_df, use_container_width=True)


# ----------------------------------------------------------------------------
# SECTION 8: PROMPT MANAGEMENT
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
