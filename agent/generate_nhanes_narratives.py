"""
generate_nhanes_narratives.py

Uses Claude (Anthropic API -- our LLM for generating text) to create
a natural-language health narrative for each NHANES participant based
on their decoded survey responses.

This is the document generation step for our RAG pipeline:
  1. Pull decoded participant data from stg_participants (our dbt view)
  2. For each participant, send their health profile to Claude and ask
     it to write a concise clinical narrative summary
  3. Store the narrative + a pgvector embedding in a new table
     nhanes_platform.raw.participant_narratives

The narratives become our RAG corpus -- real population health data
expressed as searchable natural-language documents.
"""

import os
import sys
import psycopg2
import anthropic
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "nhanes_platform",
    "user": "postgres",
}


def create_narratives_table(cur, conn):
    """Creates the table that stores generated narratives and embeddings."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw.participant_narratives (
            participant_id    INT PRIMARY KEY,
            narrative_text    TEXT NOT NULL,
            narrative_embedding vector(384),
            created_at        TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_narrative_embedding
        ON raw.participant_narratives
        USING ivfflat (narrative_embedding vector_cosine_ops)
        WITH (lists = 50)
    """)
    conn.commit()
    print("Created participant_narratives table.")


def build_prompt(row):
    """
    Builds a structured prompt for Claude to generate a health narrative.
    Uses the decoded human-readable values from stg_participants.
    """
    return f"""Write a concise 2-3 sentence clinical health summary for this survey participant. 
Be factual and objective. Use plain language. Do not make up information beyond what is provided.

Participant profile:
- Age: {row['age_years']} years old
- Gender: {row['gender']}
- Race/Ethnicity: {row['race_ethnicity']}
- Education: {row['education_level']}
- Household income: {row['income_category']}
- Health insurance: {row['has_health_insurance']}
- Self-reported health: {row['self_reported_health']}
- Doctor visits (past year): {row['doctor_visits_past_year']}
- Hospitalized (past year): {row['hospitalized_past_year']}
- Hypertension diagnosis: {row['diagnosed_hypertension']}
- Taking BP medication: {row['taking_bp_medication']}
- Diabetes diagnosis: {row['diagnosed_diabetes']}
- Smoking history: {row['ever_smoked']}
- Current smoking: {row['current_smoking_status']}
- Depression severity (PHQ-9): {row['depression_severity']}

Write the summary now:"""


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    create_narratives_table(cur, conn)

    # Pull participants who don't yet have a narrative generated.
    cur.execute("""
        SELECT participant_id, age_years, gender, race_ethnicity,
               education_level, income_category, has_health_insurance,
               self_reported_health, doctor_visits_past_year,
               hospitalized_past_year, diagnosed_hypertension,
               taking_bp_medication, diagnosed_diabetes,
               ever_smoked, current_smoking_status, depression_severity
        FROM analytics_staging.stg_participants
        WHERE participant_id NOT IN (
            SELECT participant_id FROM raw.participant_narratives
        )
        ORDER BY participant_id
        LIMIT 200
    """)
    participants = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    print(f"Generating narratives for {len(participants)} participants...")

    # Load embedding model (sentence-transformers -- runs locally, no API cost)
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Claude client (Anthropic API -- for generating the narrative text)
    client = anthropic.Anthropic(api_key=api_key)

    for i, row_tuple in enumerate(participants):
        row = dict(zip(columns, row_tuple))
        participant_id = row['participant_id']

        # Generate narrative using Claude
        try:
            prompt = build_prompt(row)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            narrative = response.content[0].text.strip()
        except Exception as e:
            print(f"  Error generating narrative for {participant_id}: {e}")
            continue

        # Embed the narrative using sentence-transformers (local, no API cost)
        embedding = embed_model.encode(narrative).tolist()

        # Store both the narrative and its embedding
        cur.execute("""
            INSERT INTO raw.participant_narratives
                (participant_id, narrative_text, narrative_embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (participant_id) DO UPDATE
                SET narrative_text = EXCLUDED.narrative_text,
                    narrative_embedding = EXCLUDED.narrative_embedding
        """, (participant_id, narrative, embedding))

        if (i + 1) % 10 == 0:
            conn.commit()
            print(f"  [{i+1}/{len(participants)}] Generated and stored {i+1} narratives...")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. {len(participants)} participant narratives generated and stored.")


if __name__ == "__main__":
    main()
