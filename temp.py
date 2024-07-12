from glob import glob
import os
import json

def update_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json") and filename.startswith("performance_metrics_round_"):
            filepath = os.path.join(directory, filename)
            
            # Read the JSON file
            with open(filepath, 'r') as file:
                data = json.load(file)
            
            # Update the client values
            data["unlearn_accuracy"] = data["global_accuracy"]
            data["unlearn_loss"] = data["global_loss"]
            
            # Write the updated JSON back to the file
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)

# Example usage:
# update_json_files('/path/to/your/json/files')
for exp_dir in glob(os.path.join("experiments", "*_retrain")):
    update_json_files(exp_dir)

