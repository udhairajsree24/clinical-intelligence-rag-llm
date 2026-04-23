from transformers import pipeline

ner_pipeline = pipeline(
    "token-classification",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

KNOWN_DIAGNOSES = {
    "hypertension",
    "diabetes",
    "asthma",
    "anemia",
    "pneumonia",
    "depression",
    "migraine",
    "heart failure",
    "chronic kidney disease",
    "covid-19",
}

KNOWN_SYMPTOMS = {
    "fatigue",
    "fever",
    "cough",
    "headache",
    "dizziness",
    "chest pain",
    "shortness of breath",
    "nausea",
    "vomiting",
    "abdominal pain",
}

KNOWN_MEDICATIONS = {
    "metformin",
    "aspirin",
    "ibuprofen",
    "paracetamol",
    "acetaminophen",
    "lisinopril",
    "amlodipine",
    "atorvastatin",
    "insulin",
    "omeprazole",
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split()).strip(".,;:()[]{}\"'")


def unique_keep_order(items):
    seen = set()
    output = []

    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            output.append(item)

    return output


def expand_token_span(clinical_note: str, start: int, end: int) -> str:
    if start is None or end is None:
        return ""

    n = len(clinical_note)
    left = start
    right = end

    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/")

    while left > 0 and clinical_note[left - 1] in valid_chars:
        left -= 1

    while right < n and clinical_note[right] in valid_chars:
        right += 1

    return clinical_note[left:right].strip()


def clean_entity_text(clinical_note: str, item: dict) -> str:
    start = item.get("start")
    end = item.get("end")

    if start is not None and end is not None:
        expanded = expand_token_span(clinical_note, start, end)
        if expanded:
            return expanded

    return item.get("word", "").strip()


def apply_fallback_rules(clinical_note: str, medications: list, symptoms: list, diagnoses: list):
    note_lower = clinical_note.lower()

    med_norms = [normalize_text(x) for x in medications]
    sym_norms = [normalize_text(x) for x in symptoms]
    diag_norms = [normalize_text(x) for x in diagnoses]

    for med in KNOWN_MEDICATIONS:
        if med in note_lower and med not in med_norms:
            medications.append(med)
            med_norms.append(med)

    for sym in KNOWN_SYMPTOMS:
        if sym in note_lower and sym not in sym_norms:
            symptoms.append(sym)
            sym_norms.append(sym)

    for diag in KNOWN_DIAGNOSES:
        if diag in note_lower and diag not in diag_norms:
            diagnoses.append(diag)
            diag_norms.append(diag)

    return medications, symptoms, diagnoses


def remove_fragmented_terms(items, known_terms):
    """
    Remove junk fragments like 'di' or 'zziness' if the full known term
    (e.g. 'dizziness') is also present.
    """
    norm_items = [normalize_text(x) for x in items]
    cleaned = []

    for item in items:
        item_norm = normalize_text(item)

        # Keep if it is a known term
        if item_norm in known_terms:
            cleaned.append(item)
            continue

        # Drop very short fragments
        if len(item_norm) <= 3:
            continue

        # Drop if this fragment is part of a full known term already present
        is_fragment = False
        for term in known_terms:
            if item_norm != term and item_norm in term and term in norm_items:
                is_fragment = True
                break

        if not is_fragment:
            cleaned.append(item)

    return unique_keep_order(cleaned)


def extract_entities_with_bert(clinical_note: str):
    results = ner_pipeline(clinical_note)

    medications = []
    symptoms = []
    diagnoses = []
    processed_entities = []

    for item in results:
        entity_text = clean_entity_text(clinical_note, item)
        entity_group = item.get("entity_group", "").strip()

        if not entity_text:
            continue

        entity_text_norm = normalize_text(entity_text)
        entity_group_lower = entity_group.lower()

        processed_entities.append({
            "text": entity_text,
            "normalized_text": entity_text_norm,
            "entity_group": entity_group,
            "score": float(item.get("score")) if item.get("score") is not None else None,
            "start": item.get("start"),
            "end": item.get("end"),
        })

        if entity_group_lower == "medication":
            medications.append(entity_text)

        elif entity_group_lower in {"sign_symptom", "symptom"}:
            if entity_text_norm in KNOWN_DIAGNOSES:
                diagnoses.append(entity_text)
            else:
                symptoms.append(entity_text)

        elif entity_group_lower in {"disease_disorder", "disease", "diagnosis"}:
            diagnoses.append(entity_text)

    medications = unique_keep_order(medications)
    symptoms = unique_keep_order(symptoms)
    diagnoses = unique_keep_order(diagnoses)

    medications, symptoms, diagnoses = apply_fallback_rules(
        clinical_note,
        medications,
        symptoms,
        diagnoses
    )

    medications = remove_fragmented_terms(medications, KNOWN_MEDICATIONS)
    symptoms = remove_fragmented_terms(symptoms, KNOWN_SYMPTOMS)
    diagnoses = remove_fragmented_terms(diagnoses, KNOWN_DIAGNOSES)

    return {
        "clinical_note": clinical_note,
        "medications": medications,
        "symptoms": symptoms,
        "diagnoses": diagnoses,
        "raw_entities": processed_entities
    }