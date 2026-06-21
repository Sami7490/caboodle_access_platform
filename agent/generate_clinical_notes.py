"""
generate_clinical_notes.py

Generates synthetic clinical notes for each encounter in the database
and stores them with vector embeddings in raw.clinical_notes.

Two steps:
  1. Generate fake but realistic-looking clinical note text using
     templates -- no API call needed, just structured fake text that
     varies based on encounter type and discharge disposition.
  2. Embed each note using sentence-transformers (a Python library
     that runs a small local embedding model -- converts text into
     a 384-dimensional vector representing its meaning) and store
     the embedding in pgvector (our Postgres vector extension).
"""

import psycopg2
import random
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# CLINICAL NOTE TEMPLATES
# These are realistic-sounding but entirely fake clinical notes.
# They vary by encounter type and discharge disposition so that
# semantic search will actually find meaningful patterns.
# ----------------------------------------------------------------------------

INPATIENT_TEMPLATES = [
    "Patient admitted with {complaint}. Vitals on admission: BP {bp}, HR {hr}, Temp {temp}F. "
    "Physical exam revealed {finding}. Labs notable for {lab}. Patient was treated with {treatment}. "
    "Condition improved over {los} day hospital stay. Discharged {disposition} with follow-up in {followup} weeks.",

    "{age}-year-old patient presented with {complaint}. History of {history}. "
    "Treated with {treatment} during admission. Length of stay {los} days. "
    "Patient discharged {disposition} in stable condition.",

    "Admission for {complaint}. Notable findings include {finding}. "
    "Patient responded well to {treatment}. Discharged {disposition} after {los} days.",
]

ED_TEMPLATES = [
    "Patient presented to ED with {complaint}. Vitals stable. "
    "Exam showed {finding}. Treated with {treatment} and observed for {los} hours. "
    "Discharged home in improved condition.",

    "Emergency visit for {complaint}. {age}-year-old with history of {history}. "
    "Labs and imaging {lab_result}. Managed with {treatment}. Disposition: home.",
]

OUTPATIENT_TEMPLATES = [
    "Follow-up visit for {complaint}. Patient reports {symptom_update}. "
    "Vitals within normal limits. Plan: continue {treatment}, return in {followup} weeks.",

    "{age}-year-old presenting for {visit_type}. No acute concerns. "
    "Growth and development appropriate. Counseled on {counseling}.",
]

COMPLAINTS = [
    "fever and cough", "abdominal pain", "respiratory distress", "asthma exacerbation",
    "pneumonia", "dehydration", "seizure", "head trauma", "urinary tract infection",
    "bronchiolitis", "appendicitis", "diabetic ketoacidosis", "cellulitis",
    "failure to thrive", "anemia"
]

FINDINGS = [
    "decreased breath sounds bilaterally", "diffuse abdominal tenderness",
    "wheezing on auscultation", "elevated white count", "consolidation on chest X-ray",
    "mild dehydration", "normal neurological exam", "erythema and swelling of affected area"
]

TREATMENTS = [
    "IV fluids and antibiotics", "nebulized albuterol", "oral rehydration therapy",
    "supplemental oxygen", "IV methylprednisolone", "surgical consultation",
    "pain management with ibuprofen", "antiepileptic medication"
]

HISTORIES = [
    "asthma", "type 1 diabetes", "congenital heart disease", "sickle cell disease",
    "no significant past medical history", "premature birth", "recurrent UTIs"
]

DISPOSITIONS_TEXT = {
    "Home": "home",
    "Home Health": "home with home health services",
    "SNF": "to skilled nursing facility",
    "AMA": "against medical advice",
    "Expired": "deceased -- family notified and supported",
}


def generate_note(encounter_type, discharge_disposition, length_of_stay_days):
    """Generates one fake clinical note for an encounter."""
    age = random.randint(1, 17)
    los = max(1, round(length_of_stay_days or 1))
    disposition_text = DISPOSITIONS_TEXT.get(discharge_disposition, "home")

    if encounter_type == "Inpatient":
        template = random.choice(INPATIENT_TEMPLATES)
        note = template.format(
            complaint=random.choice(COMPLAINTS),
            bp=f"{random.randint(90,120)}/{random.randint(60,80)}",
            hr=random.randint(70, 110),
            temp=round(random.uniform(97.5, 104.0), 1),
            finding=random.choice(FINDINGS),
            lab=f"WBC {random.randint(5,18)}k, CRP elevated",
            treatment=random.choice(TREATMENTS),
            los=los,
            disposition=disposition_text,
            followup=random.choice([1, 2, 4]),
            age=age,
            history=random.choice(HISTORIES),
        )
    elif encounter_type == "Emergency":
        template = random.choice(ED_TEMPLATES)
        note = template.format(
            complaint=random.choice(COMPLAINTS),
            finding=random.choice(FINDINGS),
            treatment=random.choice(TREATMENTS),
            los=random.randint(2, 8),
            age=age,
            history=random.choice(HISTORIES),
            lab_result=random.choice(["unremarkable", "notable for mild leukocytosis", "within normal limits"]),
        )
    else:
        template = random.choice(OUTPATIENT_TEMPLATES)
        note = template.format(
            complaint=random.choice(COMPLAINTS),
            symptom_update=random.choice(["improvement in symptoms", "no new concerns", "persistent cough"]),
            treatment=random.choice(TREATMENTS),
            followup=random.choice([1, 2, 4, 6]),
            age=age,
            visit_type=random.choice(["well child visit", "follow-up", "chronic disease management"]),
            counseling=random.choice(["nutrition and exercise", "medication adherence", "asthma action plan"]),
        )
    return note


def main():
    print("Connecting to Postgres...")
    conn = psycopg2.connect(
        host="localhost", port=5432,
        dbname="caboodle_access", user="postgres"
    )
    cur = conn.cursor()

    # Pull all encounters to generate notes for.
    cur.execute("""
        SELECT encounter_key, encounter_type, discharge_disposition, discharge_datetime - admission_datetime
        FROM raw.encounter_fact
        ORDER BY encounter_key
    """)
    encounters = cur.fetchall()
    print(f"  Found {len(encounters)} encounters to generate notes for.")

    # Generate one note per encounter.
    print("Generating clinical notes...")
    notes = []
    for enc_key, enc_type, disposition, los_interval in encounters:
        los_days = los_interval.days if los_interval else 1
        note_text = generate_note(enc_type, disposition, los_days)
        note_type = f"{enc_type} Note"
        notes.append((enc_key, note_text, note_type))

    # Load the local embedding model via sentence-transformers (a Python
    # library that runs a small transformer model locally to convert text
    # into 384-dimensional vectors representing semantic meaning).
    print("Loading local embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # all-MiniLM-L6-v2 is a small, fast, high-quality embedding model
    # that runs entirely locally -- no API calls, no cost per embedding.

    print("Generating embeddings for all notes...")
    note_texts = [n[1] for n in notes]
    embeddings = model.encode(
        note_texts,
        show_progress_bar=True,  # shows a progress bar while encoding
        batch_size=64,            # process 64 notes at a time for efficiency
    )
    print(f"  Generated {len(embeddings)} embeddings (each {len(embeddings[0])} dimensions).")

    # Insert notes + embeddings into raw.clinical_notes.
    print("Inserting notes and embeddings into Postgres...")
    for i, (enc_key, note_text, note_type) in enumerate(notes):
        embedding_list = embeddings[i].tolist()  # convert numpy array to plain Python list
        cur.execute(
            """INSERT INTO raw.clinical_notes
               (encounter_key, note_text, note_type, note_embedding)
               VALUES (%s, %s, %s, %s)""",
            (enc_key, note_text, note_type, embedding_list)
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. {len(notes)} clinical notes with embeddings stored in raw.clinical_notes.")


if __name__ == "__main__":
    main()
