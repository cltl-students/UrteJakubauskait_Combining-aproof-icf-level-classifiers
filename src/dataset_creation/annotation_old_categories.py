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

FEWSHOT_PATH = pathlib.Path("FILE_WITH_EXAMPLES.json")
FEWSHOT = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))

CATEGORIES = ["B1300 Energy level", "B140 Attention functions", "B152 Emotional functions",
              "B440 Respiration functions", "B455 Exercise tolerance functions", "B530 Weight maintenance functions",
              "D450 Walking", "D550 Eating", "D840-D859 Work and employment", "None"]

DEFINITION = {"B1300 Energy level": "Mental functions that produce vigour and stamina",
              "B140 Attention functions": "Specific mental functions of focusing on an external stimulus or internal experience for the required period of time", 
              "B152 Emotional functions": "Specific mental functions related to the feeling and affective components of the processes of the mind", 
              "B440 Respiration functions": "Functions of inhaling air into the lungs, the exchange of gases between air and blood, and exhaling air", 
              "B455 Exercise tolerance functions": "Functions related to respiratory and cardiovascular capacity as required for enduring physical exertion",  
              "B530 Weight maintenance functions": "Functions of maintaining appropriate body weight, including weight gain during the developmental period", 
              "D450 Walking": "Moving along a surface on foot, step by step, so that one foot is always on the ground, such as when strolling, sauntering, walking forwards, backwards, or sideways. Include: walking short or long distances; walking on different surfaces; walking around obstacles", 
              "D550 Eating": "Carrying out the coordinated tasks and actions of eating food that has been served, bringing it to the mouth and consuming it in culturally acceptable ways, cutting or breaking food into pieces, opening bottles and cans, using eating implements, having meals, feasting or dining. Exclude: ingestion functions (chewing, swallowing, etc.), appetite", 
              "D840-D859 Work and employment": "apprenticeship (work preparation); acquiring, keeping and terminating a job; remunerative employment; non-remunerative employment",
              "None": "Does not belong to any of the ICF categories in the list"}

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
    input_json: str = "filtered_sentences.json",
    output_json: str = "labeled_sentences.json",
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
