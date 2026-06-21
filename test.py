from pathlib import Path

csv_dir = Path("data/raw/FLNET2023/DDoS/CSV")
files = list(csv_dir.glob("*.csv"))

print("CSV files:", len(files))
print(files[:5])