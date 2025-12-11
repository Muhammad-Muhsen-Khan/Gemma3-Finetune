import pandas as pd

import json

df = pd.read_csv('all_icd10cm_codes.csv')
sft_data = []

for index, row in df.iterrows():
    _id = ("0" * 12)[:12-len(str(index))] + str(index)
    code = row['code'][:3]
    text = row['text']
    conversations = [
        {
            "from": "human",
            "value": "What is the ICD-10CM generic code for '{}'?".format(text)
        },
        {
            "from": "gpt",
            "value": f"{code}"
        }
    ]
    sft_data.append({
        "id": _id,
        "conversations": conversations
    })

with open('train_set_official_desc_sft.json', 'w') as f:
    json.dump(sft_data, f, indent=4)
