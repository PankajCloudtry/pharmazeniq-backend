# scripts/fetch_vendors_overpass.py
import requests, pandas as pd, pathlib, time, random

BBOX = "28.55,77.25,28.60,77.35"   # south,west,north,east  (Greater-Noida demo)
QUERY = f"""
[out:json][timeout:25];
node({BBOX})[amenity=pharmacy];
out center;
"""

print("⏳ Querying Overpass…")
r = requests.post("https://overpass-api.de/api/interpreter",
                  data={"data": QUERY}, timeout=60)
r.raise_for_status()
nodes = r.json()["elements"]
print("✅ Received", len(nodes), "pharmacies")

MED_ID = 1  # Augmentin 625 in medicines.csv
rows = [{
    "vendor_id": n["id"],
    "vendor_name": n.get("tags", {}).get("name", "Local Pharmacy"),
    "latitude": n["lat"],
    "longitude": n["lon"],
    "medicine_id": MED_ID,
    "price": round(random.uniform(140, 170), 2),
    "stock": random.randint(5, 25),
    "eta_minutes": random.randint(15, 30),
} for n in nodes[:20]]

out = pathlib.Path("data") / "vendor_prices.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("📄  Wrote", out.absolute())
