"""
Run this to check which image URLs are working and which are broken.
Usage:  python check_images.py
"""
import sys
sys.path.insert(0, '.')
from utils.recommender import FOOD_IMAGE_MAP
import requests, time

print("Checking all food image URLs...\n")
broken = []
working = []

for key, url in FOOD_IMAGE_MAP.items():
    try:
        r = requests.head(url, timeout=6, allow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            working.append(key)
            print(f"  ✅ {key}")
        else:
            broken.append((key, url, r.status_code))
            print(f"  ❌ {key}  (HTTP {r.status_code})")
    except Exception as e:
        broken.append((key, url, str(e)))
        print(f"  ❌ {key}  ({e})")
    time.sleep(0.1)   # be polite to Unsplash

print(f"\n✅ Working: {len(working)}")
print(f"❌ Broken:  {len(broken)}")

if broken:
    print("\nBroken URLs to fix:")
    for key, url, status in broken:
        print(f"  '{key}': {url}")
    print("\nReplace broken ones with any working Unsplash photo ID from:")
    print("  https://unsplash.com/s/photos/YOUR_FOOD_NAME")
    print("  Right-click the photo → Copy link → extract the photo-XXXXXXX part")
