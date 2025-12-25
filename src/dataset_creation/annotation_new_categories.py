"""
This script labels Dutch clinical sentences with difficulty levels for
predefined ICF categories using a GPT model accessed via Azure OpenAI.
"""

import os
import csv
import json
import pathlib
import httpx
import re
import time
from openai import AzureOpenAI


os.environ["NO_PROXY"] = "YOUR_AZURE_OPENAI_ENDPOINT,PRIVATE_IP"

http=httpx.Client(verify=False)
dns="https://YOUR_AZURE_OPENAI_ENDPOINT"
ip="https://PRIVATE_IP"
client= AzureOpenAI(api_key="AZURE_OPENAI_API_KEY",
                    api_version="2024-02-01",
                    azure_endpoint=ip,
                    http_client=http)

FEWSHOT_PATH = pathlib.Path("fewshot_examples.json")
FEWSHOT = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))

CATEGORIES = ["B280 Sensations of pain", "B134 Sleep functions", "D760 Family relationships", "B164 Higher-level cognitive functions", 
              "D465 Moving around using equipment", "D410 Changing basic body position", "B230 Hearing functions", "D240 Handling stress and other psychological demands"]

DEFINITION = {"B280 Sensations of pain": "Sensation of unpleasant feeling indicating potential or actual damage to some body structure", 
              "B134 Sleep functions": "General mental functions of periodic, reversible and selective physical and mental disengagement from one's immediate environment accompanied by characteristic physiological changes", 
              "D760 Family relationships": "Creating and maintaining kinship relationships, such as with members of the nuclear family, extended family, foster and adopted family and step-relationships, more distant relationships such as second cousins, or legal guardians", 
              "B164 Higher-level cognitive functions": "Specific mental functions especially dependent on the frontal lobes of the brain, including complex goal-directed behaviours such as decision-making, abstract thinking, planning and carrying out plans, mental flexibility, and deciding which behaviours are appropriate under what circumstances; often called executive functions", 
              "D465 Moving around using equipment": "Moving the whole body from place to place, on any surface or space, by using specific devices designed to facilitate moving or create other ways of moving around, such as with skates, skis, scuba equipment, swim fins, or moving down the street in a wheelchair or a walker", 
              "D410 Changing basic body position": "Getting into and out of a body position and moving from one location to another, such as rolling from one side to the other, sitting, standing, getting up out of a chair to lie down on a bed, and getting into and out of positions of kneeling or squatting", 
              "B230 Hearing functions": "Sensory functions relating to sensing the presence of sounds and discriminating the location, pitch, loudness and quality of sounds", 
              "D240 Handling stress and other psychological demands": "Carrying out simple or complex and coordinated actions to manage and control the psychological demands required to carry out tasks demanding significant responsibilities and involving stress, distraction, or crises, such as taking exams, driving a vehicle during heavy traffic, putting on clothes when hurried by parents, finishing a task within a time-limit or taking care of a large group of children"}

def build_prompt(sentences_with_cats, detailed_defs=False):
    """
    sentences_with_cats: list of tuples (sentence, categories)
    """
    sys = {"role": "system",
           "content": ("You are an annotation assistant.\n"
                       "You will receive sentences from a Dutch clinical note, each with one or more ICF categories already assigned.\n"
                       "For each sentence, assign a difficulty level for each given category.\n"
                       "For categories 'D450 Walking' and 'B455 Exercise tolerance functions', use a scale from 0 to 5 "
                       "(0 = no ability at all, 5 = no problems/full ability).\n"
                       "For other ICF categories, use a scale from 0 to 4 "
                       "(0 = no ability at all, 4 = no problems/full ability).\n"
                       "Return a JSON array of objects.")}

    if detailed_defs:
        defs = "\n".join(f"- **{cat}**: {DEFINITION[cat]}" for cat in CATEGORIES)
    else:
        defs = "Categories: " + ", ".join(CATEGORIES)

    examples = "\n".join(
        f"- **Sentence**: {ex['sentence']}\n"
        f"  **Categories**: {', '.join(ex['categories'])}\n"
        f"  **Labels**: {', '.join(ex['labels'])}"
        for ex in FEWSHOT
    )

    user_sentences = []
    for i, (sent, cats) in enumerate(sentences_with_cats, start=1):
        cats_str = ", ".join(cats)
        user_sentences.append(f"{i}. {sent}\n   Categories: {cats_str}")

    user = {
        "role": "user",
        "content": (
            f"{defs}\n\n"
            "### Examples (already annotated):\n"
            f"{examples}\n\n"
            "### Sentences:\n"
            + "\n".join(user_sentences)
            + "\n\n"
            "### Output format:\n"
            "Return a JSON array of objects, each with:\n"
            "  {\n"
            "    \"sentence_index\": <1-based index>,\n"
            "    \"labels\": [<a label value (0–4 or 0–5) for each category in the same order>]\n"
            "  }\n"
        )
    }

    return [sys, user]

def main(
    input_json: str = "filtered_sentences_new_categories.json",
    output_json: str = "labeled_sentences_new_categories.json",
    model: str = "gpt-4o",
    detailed_defs: bool = False,
    batch_size: int = 50
):
    results = []
    buffer = []

    with open(input_json, encoding="utf-8") as fin:
        data = json.load(fin)

    for entry in data:
        note_id = entry["note_id"]
        sent_idx = entry["sentence_index"]
        sentence = entry["sentence"]
        cats = entry.get("categories", [])

        buffer.append((note_id, sent_idx, sentence, cats))
        if len(buffer) >= batch_size:
            _flush(buffer, results, model, detailed_defs)
            buffer.clear()

    if buffer:
        _flush(buffer, results, model, detailed_defs)

    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} labeled sentences to {output_json}")


FLUSH_COUNT = 0
PROCESSED = 0

def _flush(buffer, results, model, detailed_defs):
    """Send a batch of sentences with categories to GPT, parse the JSON output safely, and append labels to results."""
    global PROCESSED
    PROCESSED += len(buffer)
    print(f"[{time.strftime('%H:%M:%S')}] processed {PROCESSED:,} sentences ...")

    sentences_with_cats = [(b[2], b[3]) for b in buffer]
    msgs = build_prompt(sentences_with_cats, detailed_defs=detailed_defs)

    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0
    )
    raw = resp.choices[0].message.content or ""

    # Removing Markdown code fences if present
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```$", "", raw).strip()

    # Extracting the JSON array from GPT output
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        print("Could not find a JSON array in GPT output:")
        print(raw[:800])
        raise ValueError("No JSON array found")

    json_str = match.group(0)

    # Removing trailing commas in lists and objects
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)

    try:
        annos = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Failed to parse GPT output as JSON:")
        print(json_str[:800])
        raise

    for i, a in enumerate(annos):
        note_id, sent_idx, sent_text, cats = buffer[i]
        results.append({
            "note_id": note_id,
            "sentence_index": sent_idx,
            "sentence": sent_text,
            "categories": cats,
            "labels": a.get("labels", [])
        })

if __name__ == "__main__":
    main(detailed_defs=False)
