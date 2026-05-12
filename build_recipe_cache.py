"""
Run this ONCE before starting the app to pre-build the fast recipe CSV.
After this runs, the app loads in ~2 seconds instead of 2 minutes.

Usage:
    cd nutrition_app
    python build_recipe_cache.py
"""
import os, time, pandas as pd

OUT = "datasets/recipes_healthy.csv"

if os.path.exists(OUT):
    print(f"✅ Already exists: {OUT}  ({os.path.getsize(OUT)//1024} KB)")
    print("Nothing to do. Run 'streamlit run app.py'")
    exit()

HEALTHY_WORDS = [
    "spinach","lentil","chicken","salmon","tuna","quinoa","broccoli","tofu","oat",
    "chickpea","carrot","kale","avocado","apple","banana","mango","berry","blueberry",
    "strawberry","orange","tomato","cucumber","cauliflower","mushroom","asparagus",
    "pumpkin","zucchini","beet","celery","cabbage","pepper","pea","bean",
    "walnut","almond","cashew","yogurt","egg","turkey","sardine","sweet potato",
    "brown rice","barley","collard","radish","leek","lettuce",
    "lemon","peach","pear","cherry","grape","guava","pomegranate","kiwi","papaya",
    "mozzarella","ricotta","cottage","tempeh","edamame","soup","stew","salad",
    "roast","baked","grilled","steamed","stir fry"
]

print("Building fast recipe cache from foodcom_recipes.xlsx...")
print("This runs ONCE only and takes ~15 seconds. Never again after this.")
t0 = time.time()

chunks  = []
scanned = 0
for chunk in pd.read_excel("datasets/foodcom_recipes.xlsx",
                           usecols=["Name","Images","RecipeIngredientParts","RecipeInstructions"],
                           chunksize=5000):
    scanned += len(chunk)
    mask = chunk["Name"].str.lower().apply(
        lambda x: any(w in str(x) for w in HEALTHY_WORDS)
    )
    chunks.append(chunk[mask])
    print(f"  Scanned {scanned:,} rows...", end="\r")

df = pd.concat(chunks, ignore_index=True)

# Keep top 50 per keyword
selected = []
for word in HEALTHY_WORDS:
    m = df[df["Name"].str.lower().str.contains(word, na=False)]
    selected.append(m.head(50))

result = pd.concat(selected).drop_duplicates(subset=["Name"])
result = result[result["RecipeInstructions"].notna()]
result.to_csv(OUT, index=False)

print(f"\n✅ Done in {time.time()-t0:.1f}s")
print(f"   {len(result)} healthy recipes saved to {OUT}")
print(f"   File size: {os.path.getsize(OUT)//1024} KB")
print(f"\nNow run: streamlit run app.py")
