import json
import csv

json_file_path = "yourJSONfile.json"

csv_file_path = "whereYouWantToSaveYourCSVfile.csv"

with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

words = list(data.keys())
frequencies = list(data.values())

# Write to CSV
with open(csv_file_path, "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(words)
    
    writer.writerow(frequencies)

print(f"CSV file saved to {csv_file_path}")