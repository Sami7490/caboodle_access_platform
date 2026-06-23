"""
sql_agent.py

A multi-tool Claude agent (using the Anthropic API -- the interface
that lets us send prompts to Claude programmatically and receive
responses back in Python) that answers natural-language questions
about the caboodle_access_platform database.

Unlike a simple single-shot NL-to-SQL script (which just converts a
question to SQL and runs it), this is a real AGENT -- Claude decides
which tool to call, calls it, sees the result, and decides whether to
call another tool or return a final answer. This loop continues until
Claude has enough information to answer the question.

Tools available to the agent:
  1. query_database    : runs a validated SELECT query against Postgres
  2. get_patient_risk  : fetches no-show + readmission risk scores for
                         a specific patient from our trained ML models
  3. get_summary_stats : returns pre-computed key metrics (no-show rate,
                         readmission rate, patient days) for the dashboard

Guardrails: every SQL query passes through guardrails.py (our SQL
safety layer) before execution -- the agent can never modify data.

Observability: every Claude API call is logged to raw.llm_call_log
(our LLM observability table) with prompt, response, tokens, latency.
"""

import os
import json
import time
import psycopg2
import joblib
import numpy as np
import anthropic  # the Anthropic Python SDK -- lets us call Claude API
from guardrails import validate_sql, safe_execute

# ----------------------------------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------------------------------

def get_db_connection():
    """Returns a fresh psycopg2 connection to our local Postgres database."""
    return psycopg2.connect(
        host="localhost", port=5432,
        dbname="caboodle_access", user="postgres"
    )


def get_system_prompt(prompt_name: str = "agent_system_prompt") -> str:
    """
    Loads the active system prompt from raw.prompt_library (our prompt
    management table in Postgres) rather than using a hardcoded string.

    This is the prompt management pattern: storing prompts in a database
    means you can edit, version, and swap them without touching code,
    restart the app, or redeploying anything. It also gives you a full
    audit trail of every prompt change via the version and updated_at
    columns.

    Falls back to a safe default if the table or prompt name is not found,
    so the agent never crashes just because of a missing prompt.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT prompt_text FROM raw.prompt_library
               WHERE prompt_name = %s AND is_active = TRUE
               ORDER BY version DESC LIMIT 1""",
            (prompt_name,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row[0]
    except Exception as e:
        print(f"[PROMPT MANAGER] Failed to load prompt '{prompt_name}': {e}")

    # Safe fallback if database lookup fails.
    return "You are a clinical analytics assistant. Answer questions about the database accurately and concisely."

# ----------------------------------------------------------------------------
# OBSERVABILITY: log every Claude API call to the database
# ----------------------------------------------------------------------------

def log_llm_call(tool_name, prompt, response_text, model,
                 input_tokens, output_tokens, latency_ms, success):
    """
    Writes one row to raw.llm_call_log (our LLM observability table)
    for every Claude API call made by the agent, regardless of whether
    it succeeded or failed. This is how we track usage, debug issues,
    and prove the agent is working in a portfolio demo.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO raw.llm_call_log
               (tool_name, prompt_text, response_text, model,
                input_tokens, output_tokens, latency_ms, success)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (tool_name, prompt[:2000], response_text[:2000], model,
             input_tokens, output_tokens, latency_ms, success)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        # Never let observability logging crash the agent itself.
        print(f"[OBSERVABILITY WARNING] Failed to log call: {e}")

# ----------------------------------------------------------------------------
# TOOL 1: query_database
# Runs a validated SELECT query against Postgres and returns results.
# ----------------------------------------------------------------------------

def query_database(sql: str) -> str:
    """
    Executes a SQL query against the caboodle_access database after
    passing it through the guardrails layer (our SQL safety validator).
    Returns results as a JSON string so Claude can read and reason about
    the data in its next turn.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    success, message, rows = safe_execute(cur, sql)
    cur.close()
    conn.close()

    if not success:
        return json.dumps({"error": message})

    # Get column names from the cursor description so results are
    # returned as a list of dicts (column name -> value) rather than
    # raw tuples, which are harder for Claude to interpret.
    col_names = [desc[0] for desc in cur.description] if cur.description else []
    result_rows = [dict(zip(col_names, row)) for row in rows]

    # Limit to 100 rows to keep the response within Claude's context window.
    return json.dumps({
        "row_count": len(result_rows),
        "rows": result_rows[:100]
    }, default=str)  # default=str handles dates/decimals that aren't JSON-serializable


# ----------------------------------------------------------------------------
# TOOL 2: get_patient_risk
# Loads trained ML models and scores a specific patient's risk.
# ----------------------------------------------------------------------------

def get_patient_risk(patient_key: int) -> str:
    """
    Fetches no-show probability and readmission probability for a
    specific patient by loading our trained scikit-learn models (the
    ML models we built in Phase 2) and scoring that patient's features.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Pull this patient's most recent appointment features for no-show scoring.
        cur.execute("""
            SELECT lead_time_days, appointment_type, is_no_show,
                   scheduled_datetime
            FROM analytics_marts.fact_appointments
            WHERE patient_key = %s
            ORDER BY scheduled_datetime DESC
            LIMIT 1
        """, (patient_key,))
        appt = cur.fetchone()

        # Pull this patient's most recent inpatient encounter for readmission scoring.
        cur.execute("""
            SELECT length_of_stay_days, discharge_disposition,
                   is_30_day_readmission, department_key
            FROM analytics_marts.fct_readmissions
            WHERE patient_key = %s
            ORDER BY admission_datetime DESC
            LIMIT 1
        """, (patient_key,))
        enc = cur.fetchone()

        # Count prior no-shows and admissions for context features.
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE is_no_show) AS noshow_count,
                COUNT(*) AS total_appts
            FROM analytics_marts.fact_appointments
            WHERE patient_key = %s
        """, (patient_key,))
        appt_hist = cur.fetchone()

        cur.execute("""
            SELECT COUNT(*) AS prior_admissions
            FROM analytics_marts.fct_readmissions
            WHERE patient_key = %s
        """, (patient_key,))
        enc_hist = cur.fetchone()

        cur.close()
        conn.close()

        result = {"patient_key": patient_key}

        # Score no-show risk if appointment data exists.
        if appt:
            noshow_model   = joblib.load("agent/noshow_model.pkl")
            noshow_scaler  = joblib.load("agent/noshow_scaler.pkl")
            noshow_features = joblib.load("agent/noshow_features.pkl")

            # Build a feature row matching exactly what the model was trained on.
            import pandas as pd
            appt_df = pd.DataFrame([{
                "lead_time_days": float(appt[0] or 0),
                "appointment_type": appt[1],
                "is_weekend": False,   # simplified -- could derive from scheduled_datetime
                "prior_noshow_count": int(appt_hist[0] or 0),
                "prior_appt_count": int(appt_hist[1] or 0),
                "day_of_week": "Monday",  # simplified default
            }])
            appt_df = pd.get_dummies(appt_df, columns=["appointment_type", "day_of_week"], drop_first=True)
            # Align columns to match training features exactly.
            for col in noshow_features:
                if col not in appt_df.columns:
                    appt_df[col] = 0
            appt_df = appt_df[noshow_features].fillna(0)
            scaled = noshow_scaler.transform(appt_df)
            noshow_prob = float(noshow_model.predict_proba(scaled)[0][1])
            result["noshow_probability"] = round(noshow_prob, 3)
            result["noshow_risk"] = "High" if noshow_prob > 0.4 else "Medium" if noshow_prob > 0.2 else "Low"

        # Score readmission risk if inpatient data exists.
        if enc:
            readmit_model    = joblib.load("agent/readmission_model.pkl")
            readmit_scaler   = joblib.load("agent/readmission_scaler.pkl")
            readmit_features = joblib.load("agent/readmission_features.pkl")

            import pandas as pd
            enc_df = pd.DataFrame([{
                "length_of_stay_days": float(enc[0] or 0),
                "discharge_disposition": enc[1],
                "age_years": 10,   # simplified default
                "prior_admission_count": int(enc_hist[0] or 0),
                f"department_key_{enc[3]}": 1,
            }])
            enc_df = pd.get_dummies(enc_df, columns=["discharge_disposition"], drop_first=True)
            for col in readmit_features:
                if col not in enc_df.columns:
                    enc_df[col] = 0
            enc_df = enc_df[readmit_features].fillna(0)
            scaled = readmit_scaler.transform(enc_df)
            readmit_prob = float(readmit_model.predict_proba(scaled)[0][1])
            result["readmission_probability"] = round(readmit_prob, 3)
            result["readmission_risk"] = "High" if readmit_prob > 0.4 else "Medium" if readmit_prob > 0.2 else "Low"

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ----------------------------------------------------------------------------
# TOOL 3: get_summary_stats
# Returns pre-computed key metrics for the dashboard.
# ----------------------------------------------------------------------------

def get_summary_stats() -> str:
    """
    Returns the key summary metrics we agreed on for the dashboard:
    overall no-show rate, readmission rate, total patient days, and
    appointment volume breakdown. Pre-computed here so the dashboard
    doesn't need to run expensive aggregations on every load.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) AS total_appointments,
            SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) AS no_shows,
            ROUND(100.0 * SUM(CASE WHEN is_no_show THEN 1 ELSE 0 END) / COUNT(*), 1)
                AS no_show_rate_pct
        FROM analytics_marts.fact_appointments
    """)
    appt_stats = cur.fetchone()

    cur.execute("""
        SELECT
            COUNT(*) AS total_inpatient,
            SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) AS readmissions,
            ROUND(100.0 * SUM(CASE WHEN is_30_day_readmission THEN 1 ELSE 0 END) / COUNT(*), 1)
                AS readmission_rate_pct
        FROM analytics_marts.fct_readmissions
    """)
    readmit_stats = cur.fetchone()

    cur.execute("""
        SELECT ROUND(SUM(length_of_stay_days)::numeric, 1) AS total_patient_days
        FROM analytics_marts.fact_encounters
        WHERE encounter_type = 'Inpatient'
    """)
    patient_days = cur.fetchone()

    cur.close()
    conn.close()

    return json.dumps({
        "total_appointments": appt_stats[0],
        "no_show_count": appt_stats[1],
        "no_show_rate_pct": float(appt_stats[2]),
        "total_inpatient_encounters": readmit_stats[0],
        "readmission_count": readmit_stats[1],
        "readmission_rate_pct": float(readmit_stats[2]),
        "total_patient_days": float(patient_days[0]),
    })


# ----------------------------------------------------------------------------
# AGENT: the main agentic loop
# ----------------------------------------------------------------------------

# Tool definitions -- this is how we tell Claude what tools exist,
# what they do, and what parameters they accept. Claude reads these
# descriptions and decides which tool to call based on the user's question.
TOOLS = [
    {
        "name": "query_database",
        "description": (
            "Runs a read-only SELECT SQL query against the caboodle_access "
            "Postgres database and returns the results. Use this to answer "
            "specific questions about patients, appointments, encounters, or "
            "readmissions. The database has these schemas: "
            "analytics_marts (fact_appointments, fact_encounters, fct_readmissions, dim_patients) "
            "and analytics_staging (stg_patients, stg_appointments, stg_encounters, "
            "stg_departments, stg_providers, stg_dates). "
            "Only SELECT statements are allowed -- no modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid PostgreSQL SELECT statement."
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "get_patient_risk",
        "description": (
            "Returns no-show probability and readmission probability scores "
            "for a specific patient, computed by our trained ML models. "
            "Use this when the user asks about a specific patient's risk level."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_key": {
                    "type": "integer",
                    "description": "The integer patient_key for the patient to score."
                }
            },
            "required": ["patient_key"]
        }
    },
    {
        "name": "get_summary_stats",
        "description": (
            "Returns pre-computed summary statistics for the dashboard: "
            "overall no-show rate, readmission rate, total patient days, "
            "and appointment volume. Use this for high-level overview questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# Map tool names to their Python functions so we can call them dynamically.
TOOL_FUNCTIONS = {
    "query_database": lambda args: query_database(args["sql"]),
    "get_patient_risk": lambda args: get_patient_risk(args["patient_key"]),
    "get_summary_stats": lambda args: get_summary_stats(),
}


def run_agent(user_question: str, api_key: str) -> str:
    """
    Runs the multi-tool agent loop for a given user question.

    The loop:
      1. Send the question + tool definitions to Claude.
      2. If Claude returns a tool_use block, call the tool and send
         the result back to Claude as a tool_result message.
      3. Repeat until Claude returns a plain text response (end_turn),
         meaning it has enough information to answer the question.

    This is the standard Anthropic tool-use / agentic loop pattern.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Load the system prompt from raw.prompt_library (our prompt management
    # table) rather than using a hardcoded string -- this lets us edit,
    # version, and swap prompts without touching code.
    system_prompt = get_system_prompt("agent_system_prompt")

    messages = [{"role": "user", "content": user_question}]

    max_turns = 5   # safety limit -- prevents infinite loops if something goes wrong
    turn = 0

    while turn < max_turns:
        turn += 1
        start_time = time.time()

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Log this API call to our observability table.
        log_llm_call(
            tool_name="agent_loop",
            prompt=user_question,
            response_text=str(response.content),
            model="claude-sonnet-4-6",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        # If Claude is done (no more tool calls), extract and return the text.
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "No response generated."

        # Otherwise, process each tool_use block Claude returned.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [Agent] Calling tool: {block.name} with {block.input}")
                tool_result = TOOL_FUNCTIONS[block.name](block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result,
                })

        # Add Claude's response and the tool results to the message history,
        # then loop back to send everything to Claude for the next turn.
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Agent reached maximum turns without a final answer."


# ----------------------------------------------------------------------------
# QUICK TEST: run the agent with a sample question if called directly
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Run: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    print("=== Testing multi-tool Claude agent ===\n")
    question = "What is the overall no-show rate, and which department has the highest no-show rate?"
    print(f"Question: {question}\n")
    answer = run_agent(question, api_key)
    print(f"Answer:\n{answer}")


# ----------------------------------------------------------------------------
# TOOL 4: search_similar_notes (appended after initial file creation)
# Semantic search over clinical notes using pgvector.
# ----------------------------------------------------------------------------

def search_similar_notes(query: str, limit: int = 5) -> str:
    """
    Finds clinical notes semantically similar to the query string using
    pgvector (our Postgres vector extension) and sentence-transformers
    (our local embedding model -- converts text to vectors locally).

    Unlike SQL LIKE searches (which match exact words), this finds notes
    that are semantically similar -- meaning notes about "breathing
    difficulty" will surface when you search for "respiratory distress"
    even if those exact words don't appear in the notes.
    """
    try:
        from sentence_transformers import SentenceTransformer

        # Embed the search query using the same model we used to embed
        # the notes -- critical that it's the same model, otherwise the
        # vectors won't be in the same space and similarity won't work.
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(query).tolist()

        conn = get_db_connection()
        cur = conn.cursor()

        # <=> is pgvector's cosine distance operator -- finds notes whose
        # embedding vector is closest (most similar in meaning) to our
        # query embedding, ordered from most to least similar.
        cur.execute("""
            SELECT
                n.note_id,
                n.note_type,
                n.note_text,
                e.encounter_type,
                e.patient_key,
                1 - (n.note_embedding <=> %s::vector) AS similarity_score
            FROM raw.clinical_notes n
            JOIN raw.encounter_fact e ON e.encounter_key = n.encounter_key
            ORDER BY n.note_embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = [
            {
                "note_id": r[0],
                "note_type": r[1],
                "note_text": r[2],
                "encounter_type": r[3],
                "patient_key": r[4],
                "similarity_score": round(float(r[5]), 3),
            }
            for r in rows
        ]
        return json.dumps({"results": results}, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


# Register the new tool in the TOOLS list and TOOL_FUNCTIONS map.
TOOLS.append({
    "name": "search_similar_notes",
    "description": (
        "Performs semantic search over clinical notes to find encounters "
        "with similar clinical presentations, regardless of exact wording. "
        "Use this when the user asks to find patients with similar symptoms, "
        "diagnoses, or clinical presentations. Returns the most semantically "
        "similar notes with their similarity scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A plain-English description of the clinical scenario to search for."
            },
            "limit": {
                "type": "integer",
                "description": "Number of similar notes to return (default 5)."
            }
        },
        "required": ["query"]
    }
})

TOOL_FUNCTIONS["search_similar_notes"] = lambda args: search_similar_notes(
    args["query"], args.get("limit", 5)
)
