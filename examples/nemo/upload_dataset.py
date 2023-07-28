import os

import nemollm
import pandas as pd

input_filename = "/home/mdemoret/Repos/morpheus/morpheus-dev3/examples/nemo/pubmedqa/data/pqal_fold0/dev_set.json"
output_filename = "/home/mdemoret/Repos/morpheus/morpheus-dev3/examples/nemo/datasets/pubmedqa-pqal_fold0-val.jsonl"

# Load the datapd.
train_df = pd.read_json(input_filename, orient="index")

# Convert prompt to the following format:
#   "Provided context:
#   {LABEL 0}: {CONTEXT 0},
#   ...
#   {LABEL k}: {CONTEXT k}
#   Question: {QUESTION}
#   Answer (yes / no / maybe):"


def apply_fn(x: pd.Series):
    labels = "\n".join([f"{context}: {label}" for context, label in zip(x.LABELS, x.CONTEXTS)])

    return f"Provided context:\n{labels}\nQuestion: {x.QUESTION}\nAnswer (yes / no / maybe):"


# Get the data into the correct format { "prompt": "question", "completion": "answer" }
train_df["prompt"] = train_df[["CONTEXTS", "LABELS", "QUESTION"]].apply(apply_fn, axis=1)
train_df["completion"] = train_df["final_decision"]

# Show one value:
print(train_df["prompt"].iloc[0])

# Save it to disk
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

with open(output_filename, "w") as f:
    train_df[["prompt", "completion"]].to_json(f, orient="records", lines=True)

# Upload to LLM
nemo = nemollm.NemoLLM(
    api_key=os.environ["NGC_API_KEY"],
    org_id="bwbg3fjn7she",
)

# nemo.upload(output_filename, True)

print("===SUCCESS!!!===")
