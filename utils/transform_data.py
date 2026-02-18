import os
import re
import pandas as pd
import json

# Path to your data directory
data_dir = '/home/oso/code/deceptive_generalization/data/azaria'
output_dir = os.path.join(data_dir, 'jsonl_files')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of CSV files
list_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

for file in list_files:
    csv_path = os.path.join(data_dir, file)
    jsonl_path = os.path.join(output_dir, file.replace('.csv', '.jsonl'))

    df = pd.read_csv(csv_path)

    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for record in df.to_dict(orient='records'):
            # Add label based on filename
            record['label'] = record['label']

            # Get and clean the "Fact" field
            fact_text = str(record.get('final', '')).strip()

            # Remove numbering or symbols at the beginning, e.g. "1.", "4)", "12 -", "3:" etc.
            fact_text = re.sub(r'^\s*\d+[\.\)\-:]*\s*', '', fact_text)

            # Add cleaned text
            record['final'] = fact_text

            # Write record to JSONL
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ Created {jsonl_path}")

print("🎉 All CSV files converted to JSONL and cleaned!")