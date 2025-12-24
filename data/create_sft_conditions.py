import json
import random
import pandas as pd

df = pd.read_csv('all_icd10cm_codes.csv')
code2conditions = {row["code"]: row["text"] for _, row in df.iterrows()}

with open("synthetic_medical_notes_gpt5_9k.json", "r") as f:
    data = json.load(f)

sft_data = []

for index, item in enumerate(data):
    note = item["generated_note"]
    codes = item["combo"]
    conditions = [code2conditions[code] for code in codes]
    random.shuffle(conditions)
    _id = ("0" * 12)[:12-len(str(index))] + str(index)
    conversations = [
        {
            "from": "human",
            "value": "What are the ICD-10CM generic conditions that can be extracted from '{}'?".format(note)
        },
        {
            "from": "gpt",
            "value": f"{' <CONDITION> '.join(conditions)}"
        }
    ]
    sft_data.append({
        "id": _id,
        "conversations": conversations
    })

with open('synthetic_medical_notes_gpt5_9k_sft.json', 'w') as f:
    json.dump(sft_data, f, indent=4)

