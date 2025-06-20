import os
import json
import pandas as pd

# Iterate through all the json files in the directory
directory = "traces"
global_id = 0
data = []
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            prompt = json.load(file)["prompt"]
            # Process each item in the JSON file
            for idx, item in enumerate(prompt):
                if item["role"] == "tool":
                    # Create a new data structure for each item
                    data.append({
                        "data_source": "rlla",
                        "prompt": prompt[:idx - 1],
                        "ability": "tool_use",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": prompt[idx - 1]["content"]
                        },
                        "extra_info": {
                            "split": "train",  # Use the filename without extension as split
                            "index": global_id,
                            "type": "tool_use"
                        }
                    })
                    global_id += 1
                elif idx == len(prompt) - 1:
                    # Create a new data structure for the last item
                    data.append({
                        "data_source": "rlla",
                        "prompt": prompt[:-1],
                        "ability": "response",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": item["content"]
                        },
                        "extra_info": {
                            "split": "train",  # Use the filename without extension as split
                            "index": global_id,
                            "type": "response"
                        }
                    })
                    global_id += 1

# Convert the data to parquet
df = pd.DataFrame(data)
output_file = "train.parquet"
df.to_parquet(output_file, index=False)
