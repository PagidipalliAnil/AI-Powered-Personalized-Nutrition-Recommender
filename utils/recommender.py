"""
recommender.py — AI Nutrition Recommender Engine v5

Images     : Food-specific SVG data-URIs (always work in Streamlit) +
             TheMealDB CDN as CSS overlay (loads when network allows)
Foods      : Dataset-driven (FOOD-DATA-GROUP1-5.csv) + curated medical seeds
Conditions : Diabetes, Pre-Diabetes, Hypertension, Heart Risk, Anemia,
             Obesity, Underweight, Borderline Cholesterol, Normal
Goals      : Weight Loss, Weight Gain, Gym & Muscle, Skin Health (+ Natural Skincare)
Personalise: gender / age / BMI / TDEE from diet_recommendations_dataset
"""

from __future__ import annotations
import base64, random, re, os, io
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE SYSTEM  — SVG primary (always renders) + TheMealDB overlay
# ══════════════════════════════════════════════════════════════════════════════

_MDB = "https://www.themealdb.com/images/ingredients/{}-Small.png"

_CAT_GRAD = {
    "fruits":     ("#f472b6", "#ec4899"),
    "vegetables": ("#34d399", "#10b981"),
    "legumes":    ("#fbbf24", "#f59e0b"),
    "grains":     ("#fb923c", "#f97316"),
    "proteins":   ("#818cf8", "#6366f1"),
    "dairy":      ("#38bdf8", "#0ea5e9"),
    "other":      ("#a78bfa", "#8b5cf6"),
}

_FOOD_EMOJI: dict[str, str] = {
    # Fruits
    "apple":"🍎","banana":"🍌","mango":"🥭","orange":"🍊","blueberr":"🫐","blueberry":"🫐",
    "strawberr":"🍓","strawberry":"🍓","raspberry":"🍓","raspberr":"🍓","blackberr":"🫐",
    "grape":"🍇","cherry":"🍒","pear":"🍐","lemon":"🍋","peach":"🍑","watermelon":"🍉",
    "pineapple":"🍍","kiwi":"🥝","avocado":"🥑","pomegranate":"🍎","mango":"🥭",
    "papaya":"🥭","fig":"🍈","plum":"🍑","apricot":"🍑","date":"🌴","melon":"🍈",
    # Vegetables
    "spinach":"🥬","broccoli":"🥦","carrot":"🥕","tomato":"🍅","cucumber":"🥒",
    "kale":"🥬","lettuce":"🥬","cauliflower":"🥦","mushroom":"🍄","asparagus":"🌿",
    "pumpkin":"🎃","zucchini":"🥒","beet":"🫀","celery":"🌿","cabbage":"🥬",
    "pepper":"🫑","eggplant":"🍆","sweet potato":"🍠","corn":"🌽","garlic":"🧄",
    "onion":"🧅","ginger":"🫚","radish":"🥕","leek":"🌿","chard":"🥬","collard":"🥬",
    # Legumes
    "lentil":"🫘","chickpea":"🫘","tofu":"⬜","black bean":"🫘","kidney bean":"🫘",
    "green bean":"🫘","pea":"🟢","edamame":"🫘","soybean":"🫘","mung bean":"🫘",
    "tempeh":"🟫","fenugreek":"🌿","bean":"🫘",
    # Grains
    "oat":"🌾","quinoa":"🌾","rice":"🍚","barley":"🌾","buckwheat":"🌾","millet":"🌾",
    "wheat":"🌾","bread":"🍞","pasta":"🍝","noodle":"🍜","amaranth":"🌾",
    # Proteins
    "egg":"🥚","salmon":"🐟","tuna":"🐟","sardine":"🐟","chicken":"🍗","turkey":"🦃",
    "cod":"🐟","mackerel":"🐟","herring":"🐟","tilapia":"🐟","trout":"🐟","shrimp":"🦐",
    "almond":"🌰","walnut":"🌰","cashew":"🌰","peanut":"🥜","pistachio":"🌰",
    "chia":"🌱","flaxseed":"🌱","sunflower":"🌻","pumpkin seed":"🌱","hemp":"🌱",
    "sesame":"🌱","peanut butter":"🥜",
    # Dairy
    "yogurt":"🥛","cottage":"🧀","mozzarella":"🧀","ricotta":"🧀","kefir":"🥛",
    "cheese":"🧀","milk":"🥛",
}

def _food_emoji(name: str, category: str) -> str:
    n = name.lower()
    for k, v in _FOOD_EMOJI.items():
        if k in n:
            return v
    return {"fruits":"🍎","vegetables":"🥦","legumes":"🫘","grains":"🌾",
            "proteins":"🥩","dairy":"🥛","other":"🥗"}.get(category, "🥗")

def _svg(food_name: str, category: str = "other") -> str:
    """Generate a beautiful food-specific SVG data URI — always renders in Streamlit."""
    c1, c2 = _CAT_GRAD.get(category, ("#818cf8", "#6366f1"))
    emoji  = _food_emoji(food_name, category)
    label  = food_name[:22]
    cat_u  = category.upper()
    # Unique gradient ID per food to prevent SVG ID collisions on the same page
    gid    = "g" + str(abs(hash(food_name + category)) % 999999)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200" viewBox="0 0 300 200">
  <defs>
    <linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{c1}" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="{c2}" stop-opacity="0.10"/>
    </linearGradient>
  </defs>
  <rect width="300" height="200" fill="#0f172a"/>
  <rect width="300" height="200" fill="url(#{gid})"/>
  <circle cx="150" cy="88" r="52" fill="{c1}" opacity="0.13"/>
  <text x="150" y="106" text-anchor="middle" font-size="56" font-family="Segoe UI Emoji,Apple Color Emoji,sans-serif">{emoji}</text>
  <text x="150" y="142" text-anchor="middle" font-size="14" font-weight="700" font-family="Inter,Arial,sans-serif" fill="#f1f5f9">{label}</text>
  <rect x="10" y="8" width="{len(cat_u)*7+14}" height="18" rx="9" fill="{c1}" opacity="0.22"/>
  <text x="17" y="21" font-size="9" font-weight="700" font-family="Inter,Arial,sans-serif" fill="{c1}">{cat_u}</text>
</svg>'''
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

def _mdb_url(food_name: str) -> str:
    """TheMealDB ingredient CDN URL — exact ingredient name required."""
    # Map common food names to TheMealDB ingredient names
    _MAP = {
        "spinach":"Spinach","broccoli":"Broccoli","kale":"Kale","carrot":"Carrots",
        "tomato":"Tomatoes","cucumber":"Cucumber","cauliflower":"Cauliflower",
        "mushroom":"Mushrooms","asparagus":"Asparagus","celery":"Celery",
        "garlic":"Garlic","onion":"Onion","ginger":"Ginger","beet":"Beetroot",
        "sweet potato":"Sweet Potatoes","corn":"Sweetcorn","eggplant":"Aubergine",
        "zucchini":"Courgettes","lettuce":"Romaine Lettuce","cabbage":"Green Cabbage",
        "pepper":"Red Pepper","sweet pepper":"Red Pepper","radish":"Radishes",
        "chard":"Swiss Chard","kale":"Kale","collard":"Collard Greens",
        "apple":"Apple","banana":"Bananas","mango":"Mango","orange":"Orange",
        "avocado":"Avocado","blueberry":"Blueberries","strawberry":"Strawberries",
        "blueberries":"Blueberries","strawberries":"Strawberries",
        "raspberry":"Raspberries","raspberries":"Raspberries","blackberry":"Blackberries",
        "grape":"Grapes","cherry":"Cherries","pear":"Pear","lemon":"Lemon",
        "peach":"Peaches","watermelon":"Watermelon","kiwi":"Kiwi","pineapple":"Pineapple",
        "pomegranate":"Pomegranate","date":"Dates","fig":"Figs","grapefruit":"Grapefruit",
        "lentil":"Lentils","lentils":"Lentils","chickpea":"Chickpeas","chickpeas":"Chickpeas",
        "tofu":"Tofu","edamame":"Edamame","soybean":"Soybeans","tempeh":"Tempeh",
        "black bean":"Black Beans","black beans":"Black Beans",
        "kidney bean":"Kidney Beans","kidney beans":"Kidney Beans",
        "green bean":"Green Beans","green beans":"Green Beans",
        "pea":"Green Peas","green peas":"Green Peas","mung bean":"Mung Beans",
        "fenugreek":"Fenugreek Seeds","fenugreek seeds":"Fenugreek Seeds",
        "oat":"Oats","oatmeal":"Oats","quinoa":"Quinoa","brown rice":"Brown Rice",
        "barley":"Barley","buckwheat":"Buckwheat","millet":"Millet",
        "whole wheat bread":"Whole Wheat Bread","whole wheat pasta":"Whole Wheat Pasta",
        "amaranth":"Amaranth","pasta":"Whole Wheat Pasta","bread":"Whole Wheat Bread",
        "salmon":"Salmon","tuna":"Tuna","sardine":"Sardines","sardines":"Sardines",
        "mackerel":"Mackerel","cod":"Cod","tilapia":"Tilapia","trout":"Trout",
        "chicken breast":"Chicken Breast","chicken":"Chicken Breast",
        "turkey breast":"Turkey","turkey":"Turkey",
        "egg":"Eggs","eggs":"Eggs","shrimp":"Prawns",
        "almond":"Almonds","almonds":"Almonds","walnut":"Walnuts","walnuts":"Walnuts",
        "cashew":"Cashew Nuts","cashews":"Cashew Nuts",
        "peanut":"Peanuts","peanuts":"Peanuts","pistachio":"Pistachios","pistachios":"Pistachios",
        "chia":"Chia Seeds","chia seeds":"Chia Seeds","flaxseed":"Flaxseed","flaxseeds":"Flaxseed",
        "sunflower seeds":"Sunflower Seeds","pumpkin seeds":"Pumpkin Seeds",
        "sesame seeds":"Sesame Seeds","hemp seeds":"Hemp Seeds",
        "peanut butter":"Peanut Butter",
        "greek yogurt":"Greek Yoghurt","yogurt":"Greek Yoghurt","kefir":"Kefir",
        "cottage cheese":"Cottage Cheese","mozzarella":"Mozzarella","ricotta":"Ricotta",
        "bitter gourd":"Bitter Melon","sweet potato":"Sweet Potatoes",
    }
    n = food_name.lower().strip()
    ingredient = _MAP.get(n, food_name.title())
    return _MDB.format(ingredient.replace(" ", "%20"))

def get_food_image(food_name: str, category: str = "other") -> dict:
    """Return {svg: '...', mdb: '...'} — svg is always reliable."""
    return {
        "svg": _svg(food_name, category),
        "mdb": _mdb_url(food_name),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

_FOOD_DF: pd.DataFrame | None = None
_DIET_DF: pd.DataFrame | None = None
_REC_DF:  pd.DataFrame | None = None

_EXCLUDE = {
    # Fast food / chains
    "cream","mayonnaise","ketchup","soda","chips","candy","cake","cookie","donut",
    "fries","pizza","burger","hot dog","bacon","sausage","salami","pepperoni",
    # Alcohol
    "alcohol","beer","wine","whiskey","vodka","rum","liquor","spirits",
    # Sweets
    "ice cream","brownie","fudge","toffee","caramel","nougat","chocolate syrup",
    "sugar coated","honey roasted","sweetened beverage",
    # Ultra-processed
    "crisped","puff","snack cake","energy drink","instant noodle",
    # Condiments / seasonings alone
    "bouillon","stock","broth","consomme","gravy","sauce packet",
    "lard","shortening","margarine","hydrogenated",
    # Non-food parts
    " peel"," rind","extract","supplement","protein shake",
    # Toxic plants
    "pokeberry","pokeweed",
    # Processed/deli meats
    "luncheon meat","spam","bologna","liverwurst","hot dog","sausage","salami",
    # Organ meats / offal (not commonly recommended by nutritionists)
    "goose liver","veal liver","beef liver","pork liver","duck liver","chicken liver",
    "veal lung","pork lung","beef lung","lamb lung","veal lungs","lamb kidneys",
    "chicken heart","beef heart","pork heart","lamb heart",
    "tripe","intestine","gizzard","spleen","pancreas","thymus",
    "sweetbread","blood sausage"," giblet"," offal",
    "tongue raw","brain ","brain,",
    " kidneys"," kidney cooked"," kidney raw",
    # Obscure fungi / plants
    "cloud ear","wood ear","jelly ear","tremella","jew's ear",
    "elderberr","prickly pear cactus","lotus seed",
    "arrowroot","acorn",
    # Fast food chain items
    "mcdonald","mcdonalds","burger king","kentucky fried","taco bell",
    "english muffin mc","egg mc","big mac","whopper",
}

def _is_junk(name: str) -> bool:
    n = name.lower()
    if len(n) < 3: return True
    return any(ex in n for ex in _EXCLUDE)

def _get_cat(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["apple","banana","mango","orange","grape","berry","watermelon",
        "papaya","guava","pomegranate","kiwi","pear","peach","plum","cherry","apricot",
        "fig","lychee","melon","pineapple","avocado","lemon","lime","grapefruit",
        "date","raisin","apricot","prune"]): return "fruits"
    if any(k in n for k in ["spinach","broccoli","carrot","tomato","cucumber","kale",
        "lettuce","cauliflower","beet","sweet potato","pumpkin","zucchini","celery",
        "asparagus","pepper","eggplant","cabbage","leek","mushroom","okra","radish",
        "artichoke","fennel","watercress","chard","collard","corn","garlic","onion",
        "ginger","turmeric","bitter","gourd","yam","turnip"]): return "vegetables"
    if any(k in n for k in ["lentil","chickpea","black bean","kidney bean","pea ",
        "soybean","tofu","mung","edamame","tempeh","cowpea","bean","legume",
        "fenugreek"]): return "legumes"
    if any(k in n for k in ["oat","quinoa","rice","barley","buckwheat","millet",
        "wheat","bread","pasta","noodle","grain","bran","rye","cereal",
        "amaranth","spelt","teff"]): return "grains"
    if any(k in n for k in ["salmon","tuna","sardine","chicken","turkey","egg",
        "almond","walnut","cashew","peanut","pistachio","chia","flaxseed",
        "sunflower seed","pumpkin seed","hemp","sesame","mackerel","cod",
        "tilapia","trout","shrimp","beef ","lamb ","pork ","venison",
        "herring","anchovy","oyster","clam"]): return "proteins"
    if any(k in n for k in ["yogurt","milk","cheese","ricotta","mozzarella","kefir",
        "cottage","butter","cream","dairy","whey"]): return "dairy"
    return "other"

def load_food_dataset() -> pd.DataFrame:
    global _FOOD_DF
    if _FOOD_DF is not None: return _FOOD_DF
    groups = []
    base = "datasets"
    for i in range(1, 6):
        for pth in [f"{base}/FOOD-DATA-GROUP{i}.csv",
                    f"../uploads/FOOD-DATA-GROUP{i}.csv",
                    f"/mnt/user-data/uploads/FOOD-DATA-GROUP{i}.csv"]:
            if os.path.exists(pth):
                try: groups.append(pd.read_csv(pth)); break
                except Exception: pass
    if not groups:
        _FOOD_DF = pd.DataFrame()
        return _FOOD_DF
    df = pd.concat(groups, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Caloric Value": "calories", "Fat": "fat", "Protein": "protein",
        "Carbohydrates": "carbs", "Sugars": "sugars", "Sodium": "sodium_g",
        "Dietary Fiber": "fiber", "Cholesterol": "cholesterol",
        "Vitamin C": "vit_c", "Iron": "iron", "Potassium": "potassium",
        "Calcium": "calcium", "Vitamin A": "vit_a", "Vitamin E": "vit_e",
        "Vitamin D": "vit_d", "Vitamin B12": "vit_b12", "Magnesium": "magnesium",
        "Nutrition Density": "nutrition_density", "Zinc": "zinc",
        "Saturated Fats": "sat_fat",
    })
    for col in ["calories","fat","protein","carbs","fiber","sugars",
                "sodium_g","cholesterol","vit_c","iron","potassium",
                "calcium","vit_a","vit_e","magnesium","nutrition_density","zinc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # sodium_g is in g/100g → convert to mg for display
    df["sodium"] = (df.get("sodium_g", 0) * 1000).round(1)
    df = df[df["food"].notna()].copy()
    df = df[~df["food"].apply(_is_junk)]
    df["category"] = df["food"].apply(_get_cat)
    # Normalise calories: some rows seem to be per-serving, cap at 900 kcal/100g
    df = df[df["calories"].between(5, 900)].copy()
    df = df[df["protein"].between(0, 100)].copy()
    _FOOD_DF = df.reset_index(drop=True)
    return _FOOD_DF

def load_recipe_dataset() -> pd.DataFrame:
    global _REC_DF
    if _REC_DF is not None: return _REC_DF
    for pth in ["datasets/recipes_healthy.csv",
                "/mnt/user-data/uploads/recipes_healthy.csv"]:
        if os.path.exists(pth):
            try:
                _REC_DF = pd.read_csv(pth)
                return _REC_DF
            except Exception: pass
    _REC_DF = pd.DataFrame()
    return _REC_DF

def load_diet_dataset() -> pd.DataFrame:
    global _DIET_DF
    if _DIET_DF is not None: return _DIET_DF
    for pth in ["datasets/diet_recommendations_dataset.csv",
                "/mnt/user-data/uploads/diet_recommendations_dataset.csv"]:
        if os.path.exists(pth):
            try:
                _DIET_DF = pd.read_csv(pth)
                return _DIET_DF
            except Exception: pass
    _DIET_DF = pd.DataFrame()
    return _DIET_DF


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITION NUTRITIONAL CRITERIA  (dataset filter rules per condition)
# ══════════════════════════════════════════════════════════════════════════════

# Rules applied to the FOOD dataset columns
CONDITION_CRITERIA: dict = {
    "Diabetes": {
        "max_calories": 250, "max_carbs": 30, "max_sugars": 5,
        "min_fiber": 2.0,  "max_fat": 15, "max_sodium": 400,
        "prefer_cats": ["vegetables","legumes","proteins"],
        "sort_by": "fiber", "ascending": False,
        "description": "Low-GI, high-fiber, lean protein foods (ADA guidelines)",
    },
    "Pre-Diabetes": {
        "max_calories": 300, "max_carbs": 40, "max_sugars": 8,
        "min_fiber": 1.5,  "max_fat": 18, "max_sodium": 500,
        "prefer_cats": ["vegetables","legumes","grains","proteins"],
        "sort_by": "fiber", "ascending": False,
        "description": "Moderate carb, high-fiber foods to prevent progression",
    },
    "Hypertension": {
        "max_calories": 300, "max_sodium": 200, "max_fat": 15,
        "min_fiber": 1.0, "min_potassium": 0,
        "prefer_cats": ["fruits","vegetables","legumes","grains"],
        "sort_by": "potassium", "ascending": False,
        "description": "DASH diet: high potassium, very low sodium foods",
    },
    "Pre-Hypertension": {
        "max_calories": 320, "max_sodium": 350, "max_fat": 16,
        "prefer_cats": ["fruits","vegetables","legumes","grains"],
        "sort_by": "potassium", "ascending": False,
        "description": "Moderately sodium-restricted, potassium-rich foods",
    },
    "Heart Risk": {
        "max_calories": 280, "max_fat": 12, "max_cholesterol": 50,
        "min_fiber": 2.0, "max_sodium": 400,
        "prefer_cats": ["fruits","vegetables","legumes","grains"],
        "sort_by": "fiber", "ascending": False,
        "description": "Omega-3 rich, low saturated fat, high soluble fiber (AHA)",
    },
    "Borderline Cholesterol": {
        "max_calories": 300, "max_fat": 14, "max_cholesterol": 60,
        "min_fiber": 2.0, "max_sodium": 400,
        "prefer_cats": ["fruits","vegetables","grains","legumes"],
        "sort_by": "fiber", "ascending": False,
        "description": "High soluble fiber, plant sterols to lower LDL",
    },
    "Obesity": {
        "max_calories": 180, "max_fat": 8, "min_fiber": 1.5,
        "max_carbs": 25, "max_sugars": 6, "max_sodium": 400,
        "prefer_cats": ["vegetables","fruits","proteins","legumes"],
        "sort_by": "calories", "ascending": True,
        "description": "Low calorie-density, high satiety, high protein-to-calorie ratio",
    },
    "Underweight": {
        "min_calories": 150, "max_calories": 700, "min_protein": 8,
        "prefer_cats": ["proteins","grains","legumes","dairy","fruits"],
        "sort_by": "calories", "ascending": False,
        "description": "Calorie-dense, protein-rich, nutrient-dense whole foods",
    },
    "Anemia": {
        "max_calories": 350, "min_protein": 5,
        "prefer_cats": ["proteins","legumes","vegetables","fruits"],
        "sort_by": "iron", "ascending": False,
        "description": "Iron-rich foods + Vitamin C for enhanced absorption",
    },
    "Normal": {
        "max_calories": 400, "max_fat": 22, "min_fiber": 1.0, "max_sodium": 600,
        "prefer_cats": ["vegetables","fruits","legumes","grains","proteins","dairy"],
        "sort_by": "nutrition_density", "ascending": False,
        "description": "Balanced Mediterranean-style whole foods",
    },
}

GOAL_CRITERIA: dict = {
    "Weight Loss": {
        "max_calories": 150, "min_fiber": 1.5, "max_fat": 8,
        "max_carbs": 20, "max_sugars": 6,
        "prefer_cats": ["vegetables","fruits","proteins","legumes"],
        "sort_by": "calories", "ascending": True,
        "description": "Thermogenic, high-satiety, low calorie-density foods",
        "macro_split": "Protein 35% · Carbs 35% · Fat 30%",
        "cal_formula": lambda tdee: tdee - 500,
        "tips": [
            "Eat 30g+ protein per meal — protein has the highest thermic effect of any macronutrient.",
            "Prioritise volume eating: fill half your plate with non-starchy vegetables first.",
            "Drink 500ml water before each meal — reduces hunger and total calorie intake.",
            "Sleep 7–9 hours: poor sleep elevates ghrelin (hunger hormone) by up to 24%.",
            "Strength train 3×/week: muscle tissue burns 3× more calories than fat at rest.",
        ],
    },
    "Weight Gain": {
        "min_calories": 150, "max_calories": 700, "min_protein": 6,
        "prefer_cats": ["grains","proteins","legumes","dairy","fruits"],
        "sort_by": "calories", "ascending": False,
        "description": "Calorie-dense, protein-rich, nutrient-dense whole foods",
        "macro_split": "Carbs 50% · Protein 25% · Fat 25%",
        "cal_formula": lambda tdee: tdee + 500,
        "tips": [
            "Eat every 3–4 hours to maintain a consistent calorie surplus.",
            "Liquid calories (smoothies, whole milk, yogurt drinks) are easier to consume in volume.",
            "Add calorie boosters: nut butter, olive oil, avocado, seeds to meals.",
            "Strength training converts surplus calories into muscle rather than fat.",
            "Track food intake for the first 2 weeks to ensure you're hitting your calorie target.",
        ],
    },
    "Gym & Muscle": {
        "min_protein": 15, "max_fat": 20, "max_sodium": 500,
        "prefer_cats": ["proteins","dairy","legumes","grains"],
        "sort_by": "protein", "ascending": False,
        "description": "Complete protein, amino-acid-rich, muscle-synthesis foods",
        "macro_split": "Protein 35% · Carbs 45% · Fat 20%",
        "cal_formula": lambda tdee: tdee + 250,
        "tips": [
            "Target 1.6–2.2g protein per kg bodyweight per day for optimal muscle growth.",
            "Consume fast-digesting protein (eggs, Greek yogurt) within 30 min post-workout.",
            "Complex carbs pre-workout (oats, banana, brown rice) fuel maximum performance.",
            "Cottage cheese or Greek yogurt before bed provides slow-release casein for overnight repair.",
            "Creatine from fish and red meat (and supplementation) improves strength output by 5–15%.",
        ],
    },
    "Skin Health": {
        "max_calories": 400, "max_fat": 25,
        "prefer_cats": ["fruits","vegetables","proteins","legumes"],
        "sort_by": "vit_c", "ascending": False,
        "description": "Antioxidant, collagen-building, anti-inflammatory foods",
        "macro_split": "Balanced · Emphasise Omega-3 and Vitamin C/E",
        "cal_formula": lambda tdee: tdee,
        "tips": [
            "Vitamin C (berries, citrus, peppers) is essential for collagen synthesis.",
            "Omega-3s (salmon, walnuts, flaxseeds) reduce skin inflammation and dryness.",
            "Beta-carotene (carrots, sweet potato) converts to Vitamin A for skin cell renewal.",
            "Lycopene (tomatoes, watermelon) protects against UV-induced skin damage.",
            "Hydration is everything — even mild dehydration dulls complexion and worsens wrinkles.",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  CURATED SEED LISTS  (medically verified priority foods per condition)
#  These appear first; dataset items are appended to increase variety
# ══════════════════════════════════════════════════════════════════════════════

def _c(food, cal, prot, fat, carbs, fiber, sodium, sugars, chol, cat,
       iron=0, potassium=0, vit_c=0, calcium=0):
    return {"food":food,"calories":cal,"protein":prot,"fat":fat,"carbs":carbs,
            "fiber":fiber,"sodium":sodium,"sugars":sugars,"cholesterol":chol,
            "category":cat,"iron":iron,"potassium":potassium,
            "vit_c":vit_c,"calcium":calcium,"from_dataset":False}

CURATED_SEEDS: dict[str, list] = {
    "Diabetes": [
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Cauliflower",       25, 1.9,0.3,  5.0,2.0,  30,1.9,  0,"vegetables",0.4,299,48,22),
        _c("Bitter Gourd",      17, 1.0,0.2,  3.7,2.8,   5,1.5,  0,"vegetables",0.4,296,84,19),
        _c("Asparagus",         20, 2.2,0.1,  3.9,2.1,   2,1.9,  0,"vegetables",2.1,202,5,24),
        _c("Celery",            16, 0.7,0.2,  3.0,1.6,  80,1.3,  0,"vegetables",0.2,260,3,40),
        _c("Fenugreek Seeds",  323,23.0,6.4, 58.4,24.6, 67,0.0,  0,"legumes",33.5,770,4,176),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Black Beans",      132, 8.9,0.5, 23.7,8.7,   1,0.3,  0,"legumes",3.6,355,0,27),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Tofu",              76, 8.1,4.8,  1.9,0.3,   7,0.5,  0,"legumes",5.4,121,0,350),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Sardines",         208,24.6,11.5, 0.0,0.0, 307,0.0,142,"proteins",2.9,397,0,382),
        _c("Eggs",             155,13.0,11.0, 1.1,0.0, 124,1.1,373,"proteins",1.8,138,0,56),
        _c("Turkey Breast",    189,28.6,7.4,  0.0,0.0,  73,0.0, 82,"proteins",1.5,293,0,21),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Walnuts",          185, 4.3,18.5, 3.9,1.9,   1,0.7,  0,"proteins",2.9,441,1,45),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Flaxseeds",        150, 5.1,12.0, 8.1,7.7,   8,0.4,  0,"proteins",11.9,255,1,255),
        _c("Pumpkin Seeds",    285,15.1,12.4,35.3,3.0,   4,0.7,  0,"proteins",15.0,806,2,46),
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Apple",             52, 0.3,0.2, 13.8,2.4,   1,10.4, 0,"fruits",0.1,107,5,6),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Barley",           354,12.5,2.3, 73.5,17.3, 12,0.8,  0,"grains",3.6,452,0,33),
        _c("Brown Rice",       216, 5.0,1.8, 45.0,3.5,  10,0.7,  0,"grains",1.5,154,0,23),
    ],
    "Hypertension": [
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Beets",             43, 1.6,0.2,  9.6,2.8,  78,6.8,  0,"vegetables",0.8,325,5,16),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Celery",            16, 0.7,0.2,  3.0,1.6,  80,1.3,  0,"vegetables",0.2,260,3,40),
        _c("Garlic",           149, 6.4,0.5, 33.1,2.1,  17,1.0,  0,"vegetables",1.7,401,31,181),
        _c("Sweet Potato",      86, 1.6,0.1, 20.0,3.0,  55,4.2,  0,"vegetables",0.6,337,3,30),
        _c("Tomatoes",          18, 0.9,0.2,  3.9,1.2,   5,2.6,  0,"vegetables",0.3,237,14,10),
        _c("Asparagus",         20, 2.2,0.1,  3.9,2.1,   2,1.9,  0,"vegetables",2.1,202,5,24),
        _c("Banana",            89, 1.1,0.3, 22.8,2.6,   1,12.2, 0,"fruits",0.3,358,9,5),
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Kiwi",              61, 1.1,0.5, 14.7,3.0,   3,9.0,  0,"fruits",0.3,312,93,34),
        _c("Pomegranate",       83, 1.7,1.2, 18.7,4.0,   3,13.7, 0,"fruits",0.3,236,10,10),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Watermelon",        30, 0.6,0.2,  7.6,0.4,   1,6.2,  0,"fruits",0.2,112,8,7),
        _c("Strawberries",      32, 0.7,0.3,  7.7,2.0,   1,4.9,  0,"fruits",0.4,153,59,16),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Mackerel",         205,18.6,13.9, 0.0,0.0,  90,0.0, 70,"proteins",1.6,314,0,12),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Pumpkin Seeds",    285,15.1,12.4,35.3,3.0,   4,0.7,  0,"proteins",15.0,806,2,46),
        _c("Flaxseeds",        150, 5.1,12.0, 8.1,7.7,   8,0.4,  0,"proteins",11.9,255,1,255),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Brown Rice",       216, 5.0,1.8, 45.0,3.5,  10,0.7,  0,"grains",1.5,154,0,23),
        _c("Barley",           354,12.5,2.3, 73.5,17.3, 12,0.8,  0,"grains",3.6,452,0,33),
    ],
    "Heart Risk": [
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Sardines",         208,24.6,11.5, 0.0,0.0, 307,0.0,142,"proteins",2.9,397,0,382),
        _c("Mackerel",         205,18.6,13.9, 0.0,0.0,  90,0.0, 70,"proteins",1.6,314,0,12),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Walnuts",          185, 4.3,18.5, 3.9,1.9,   1,0.7,  0,"proteins",2.9,441,1,45),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Flaxseeds",        150, 5.1,12.0, 8.1,7.7,   8,0.4,  0,"proteins",11.9,255,1,255),
        _c("Pistachio",        562,20.2,45.3,27.2,10.3,  1,7.7,  0,"proteins",3.9,1025,5,107),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Barley",           354,12.5,2.3, 73.5,17.3, 12,0.8,  0,"grains",3.6,452,0,33),
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Tomatoes",          18, 0.9,0.2,  3.9,1.2,   5,2.6,  0,"vegetables",0.3,237,14,10),
        _c("Garlic",           149, 6.4,0.5, 33.1,2.1,  17,1.0,  0,"vegetables",1.7,401,31,181),
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Pomegranate",       83, 1.7,1.2, 18.7,4.0,   3,13.7, 0,"fruits",0.3,236,10,10),
        _c("Raspberries",       52, 1.2,0.7, 11.9,6.5,   1,4.4,  0,"fruits",0.7,151,26,25),
        _c("Apple",             52, 0.3,0.2, 13.8,2.4,   1,10.4, 0,"fruits",0.1,107,5,6),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Black Beans",      132, 8.9,0.5, 23.7,8.7,   1,0.3,  0,"legumes",3.6,355,0,27),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
    ],
    "Anemia": [
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Collard Greens",    32, 3.0,0.6,  5.4,4.0,  20,0.5,  0,"vegetables",0.6,213,23,232),
        _c("Swiss Chard",       19, 1.8,0.2,  3.7,1.6, 213,1.1,  0,"vegetables",1.8,379,18,51),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Asparagus",         20, 2.2,0.1,  3.9,2.1,   2,1.9,  0,"vegetables",2.1,202,5,24),
        _c("Sweet Pepper",      46, 1.5,0.3,  8.8,3.1,  92,6.0,  0,"vegetables",1.2,314,184,10),
        _c("Tomatoes",          18, 0.9,0.2,  3.9,1.2,   5,2.6,  0,"vegetables",0.3,237,14,10),
        _c("Pomegranate",       83, 1.7,1.2, 18.7,4.0,   3,13.7, 0,"fruits",0.3,236,10,10),
        _c("Dried Apricots",   241, 3.4,0.5, 62.6,7.3,  10,53.0, 0,"fruits",6.3,1160,1,55),
        _c("Strawberries",      32, 0.7,0.3,  7.7,2.0,   1,4.9,  0,"fruits",0.4,153,59,16),
        _c("Kiwi",              61, 1.1,0.5, 14.7,3.0,   3,9.0,  0,"fruits",0.3,312,93,34),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Kidney Beans",     127, 8.7,0.5, 22.8,6.4,   2,0.3,  0,"legumes",5.2,600,1,143),
        _c("Black Beans",      132, 8.9,0.5, 23.7,8.7,   1,0.3,  0,"legumes",3.6,355,0,27),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Tofu",              76, 8.1,4.8,  1.9,0.3,   7,0.5,  0,"legumes",5.4,121,0,350),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Pumpkin Seeds",    285,15.1,12.4,35.3,3.0,   4,0.7,  0,"proteins",15.0,806,2,46),
        _c("Sesame Seeds",     573,17.7,49.7, 4.6,1.3,  11,0.1,  0,"proteins",14.6,468,0,975),
        _c("Sardines",         208,24.6,11.5, 0.0,0.0, 307,0.0,142,"proteins",2.9,397,0,382),
        _c("Tuna",             132,28.2,1.0,  0.0,0.0,  45,0.0, 49,"proteins",1.3,441,0,14),
        _c("Eggs",             155,13.0,11.0, 1.1,0.0, 124,1.1,373,"proteins",1.8,138,0,56),
        _c("Turkey Breast",    189,28.6,7.4,  0.0,0.0,  73,0.0, 82,"proteins",1.5,293,0,21),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Amaranth",         371,14.5,7.0, 65.3,6.7,  21,1.7,  0,"grains",7.6,508,4,159),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Brown Rice",       216, 5.0,1.8, 45.0,3.5,  10,0.7,  0,"grains",1.5,154,0,23),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
    ],
    "Obesity": [
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Cucumber",          16, 0.7,0.1,  3.6,0.5,   2,1.7,  0,"vegetables",0.3,147,3,16),
        _c("Celery",            16, 0.7,0.2,  3.0,1.6,  80,1.3,  0,"vegetables",0.2,260,3,40),
        _c("Cauliflower",       25, 1.9,0.3,  5.0,2.0,  30,1.9,  0,"vegetables",0.4,299,48,22),
        _c("Asparagus",         20, 2.2,0.1,  3.9,2.1,   2,1.9,  0,"vegetables",2.1,202,5,24),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Mushrooms",         22, 3.1,0.3,  3.3,1.0,   5,2.0,  0,"vegetables",0.5,318,2,3),
        _c("Zucchini",          17, 1.2,0.3,  3.1,1.0,   8,2.5,  0,"vegetables",0.4,261,17,16),
        _c("Lettuce",           15, 1.4,0.2,  2.9,1.3,  28,1.9,  0,"vegetables",0.8,194,4,35),
        _c("Tomatoes",          18, 0.9,0.2,  3.9,1.2,   5,2.6,  0,"vegetables",0.3,237,14,10),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Strawberries",      32, 0.7,0.3,  7.7,2.0,   1,4.9,  0,"fruits",0.4,153,59,16),
        _c("Grapefruit",        42, 0.8,0.1, 10.7,1.6,   0,6.9,  0,"fruits",0.1,135,43,22),
        _c("Apple",             52, 0.3,0.2, 13.8,2.4,   1,10.4, 0,"fruits",0.1,107,5,6),
        _c("Raspberries",       52, 1.2,0.7, 11.9,6.5,   1,4.4,  0,"fruits",0.7,151,26,25),
        _c("Eggs",             155,13.0,11.0, 1.1,0.0, 124,1.1,373,"proteins",1.8,138,0,56),
        _c("Chicken Breast",   165,31.0,3.6,  0.0,0.0,  74,0.0, 85,"proteins",1.0,256,0,15),
        _c("Turkey Breast",    189,28.6,7.4,  0.0,0.0,  73,0.0, 82,"proteins",1.5,293,0,21),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Tuna",             132,28.2,1.0,  0.0,0.0,  45,0.0, 49,"proteins",1.3,441,0,14),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Tofu",              76, 8.1,4.8,  1.9,0.3,   7,0.5,  0,"legumes",5.4,121,0,350),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
    ],
    "Underweight": [
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Banana",            89, 1.1,0.3, 22.8,2.6,   1,12.2, 0,"fruits",0.3,358,9,5),
        _c("Mango",             60, 0.8,0.4, 15.0,1.6,   1,13.7, 0,"fruits",0.2,168,36,11),
        _c("Dates",            277, 1.8,0.2, 75.0,6.7,   1,66.5, 0,"fruits",1.0,696,1,39),
        _c("Dried Apricots",   241, 3.4,0.5, 62.6,7.3,  10,53.0, 0,"fruits",6.3,1160,1,55),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Chicken Breast",   165,31.0,3.6,  0.0,0.0,  74,0.0, 85,"proteins",1.0,256,0,15),
        _c("Eggs",             155,13.0,11.0, 1.1,0.0, 124,1.1,373,"proteins",1.8,138,0,56),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Cashews",          553,18.2,43.9,30.2,3.3,  12,5.9,  0,"proteins",6.7,660,0,37),
        _c("Walnuts",          185, 4.3,18.5, 3.9,1.9,   1,0.7,  0,"proteins",2.9,441,1,45),
        _c("Peanut Butter",    588,25.1,50.4,20.1,6.0, 459,9.2,  0,"proteins",1.9,558,0,56),
        _c("Pistachio",        562,20.2,45.3,27.2,10.3, 1,7.7,  0,"proteins",3.9,1025,5,107),
        _c("Pumpkin Seeds",    285,15.1,12.4,35.3,3.0,   4,0.7,  0,"proteins",15.0,806,2,46),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Kidney Beans",     127, 8.7,0.5, 22.8,6.4,   2,0.3,  0,"legumes",5.2,600,1,143),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Cottage Cheese",    98,11.1,4.3,  3.4,0.0, 364,2.7, 17,"dairy",0.1,84,0,83),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Brown Rice",       216, 5.0,1.8, 45.0,3.5,  10,0.7,  0,"grains",1.5,154,0,23),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Whole Wheat Pasta", 174, 7.5,1.1,34.3,4.4,   9,1.2,  0,"grains",1.5,178,0,21),
        _c("Sweet Potato",      86, 1.6,0.1, 20.0,3.0,  55,4.2,  0,"vegetables",0.6,337,3,30),
    ],
    "Borderline Cholesterol": [
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Barley",           354,12.5,2.3, 73.5,17.3, 12,0.8,  0,"grains",3.6,452,0,33),
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Apple",             52, 0.3,0.2, 13.8,2.4,   1,10.4, 0,"fruits",0.1,107,5,6),
        _c("Pear",              57, 0.4,0.1, 15.2,3.1,   1,9.8,  0,"fruits",0.2,116,4,9),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Strawberries",      32, 0.7,0.3,  7.7,2.0,   1,4.9,  0,"fruits",0.4,153,59,16),
        _c("Pomegranate",       83, 1.7,1.2, 18.7,4.0,   3,13.7, 0,"fruits",0.3,236,10,10),
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Sweet Potato",      86, 1.6,0.1, 20.0,3.0,  55,4.2,  0,"vegetables",0.6,337,3,30),
        _c("Garlic",           149, 6.4,0.5, 33.1,2.1,  17,1.0,  0,"vegetables",1.7,401,31,181),
        _c("Tomatoes",          18, 0.9,0.2,  3.9,1.2,   5,2.6,  0,"vegetables",0.3,237,14,10),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Kidney Beans",     127, 8.7,0.5, 22.8,6.4,   2,0.3,  0,"legumes",5.2,600,1,143),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Walnuts",          185, 4.3,18.5, 3.9,1.9,   1,0.7,  0,"proteins",2.9,441,1,45),
        _c("Flaxseeds",        150, 5.1,12.0, 8.1,7.7,   8,0.4,  0,"proteins",11.9,255,1,255),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Pistachio",        562,20.2,45.3,27.2,10.3,  1,7.7,  0,"proteins",3.9,1025,5,107),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
    ],
    "Pre-Diabetes": [
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Sweet Potato",      86, 1.6,0.1, 20.0,3.0,  55,4.2,  0,"vegetables",0.6,337,3,30),
        _c("Cauliflower",       25, 1.9,0.3,  5.0,2.0,  30,1.9,  0,"vegetables",0.4,299,48,22),
        _c("Asparagus",         20, 2.2,0.1,  3.9,2.1,   2,1.9,  0,"vegetables",2.1,202,5,24),
        _c("Mushrooms",         22, 3.1,0.3,  3.3,1.0,   5,2.0,  0,"vegetables",0.5,318,2,3),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Apple",             52, 0.3,0.2, 13.8,2.4,   1,10.4, 0,"fruits",0.1,107,5,6),
        _c("Pear",              57, 0.4,0.1, 15.2,3.1,   1,9.8,  0,"fruits",0.2,116,4,9),
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Strawberries",      32, 0.7,0.3,  7.7,2.0,   1,4.9,  0,"fruits",0.4,153,59,16),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Black Beans",      132, 8.9,0.5, 23.7,8.7,   1,0.3,  0,"legumes",3.6,355,0,27),
        _c("Tofu",              76, 8.1,4.8,  1.9,0.3,   7,0.5,  0,"legumes",5.4,121,0,350),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Eggs",             155,13.0,11.0, 1.1,0.0, 124,1.1,373,"proteins",1.8,138,0,56),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Walnuts",          185, 4.3,18.5, 3.9,1.9,   1,0.7,  0,"proteins",2.9,441,1,45),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Barley",           354,12.5,2.3, 73.5,17.3, 12,0.8,  0,"grains",3.6,452,0,33),
        _c("Brown Rice",       216, 5.0,1.8, 45.0,3.5,  10,0.7,  0,"grains",1.5,154,0,23),
    ],
    "Pre-Hypertension": [
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Celery",            16, 0.7,0.2,  3.0,1.6,  80,1.3,  0,"vegetables",0.2,260,3,40),
        _c("Beets",             43, 1.6,0.2,  9.6,2.8,  78,6.8,  0,"vegetables",0.8,325,5,16),
        _c("Garlic",           149, 6.4,0.5, 33.1,2.1,  17,1.0,  0,"vegetables",1.7,401,31,181),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Sweet Potato",      86, 1.6,0.1, 20.0,3.0,  55,4.2,  0,"vegetables",0.6,337,3,30),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Banana",            89, 1.1,0.3, 22.8,2.6,   1,12.2, 0,"fruits",0.3,358,9,5),
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Kiwi",              61, 1.1,0.5, 14.7,3.0,   3,9.0,  0,"fruits",0.3,312,93,34),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Pomegranate",       83, 1.7,1.2, 18.7,4.0,   3,13.7, 0,"fruits",0.3,236,10,10),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Pumpkin Seeds",    285,15.1,12.4,35.3,3.0,   4,0.7,  0,"proteins",15.0,806,2,46),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Brown Rice",       216, 5.0,1.8, 45.0,3.5,  10,0.7,  0,"grains",1.5,154,0,23),
    ],
    "Normal": [
        _c("Spinach",           23, 2.9,0.4,  3.6,2.2,  79,0.4,  0,"vegetables",2.7,558,28,99),
        _c("Broccoli",          34, 2.8,0.4,  6.6,2.6,  33,1.7,  0,"vegetables",0.7,316,89,47),
        _c("Sweet Potato",      86, 1.6,0.1, 20.0,3.0,  55,4.2,  0,"vegetables",0.6,337,3,30),
        _c("Kale",              49, 4.3,0.9,  8.8,3.6,  38,2.3,  0,"vegetables",1.5,491,93,135),
        _c("Tomatoes",          18, 0.9,0.2,  3.9,1.2,   5,2.6,  0,"vegetables",0.3,237,14,10),
        _c("Garlic",           149, 6.4,0.5, 33.1,2.1,  17,1.0,  0,"vegetables",1.7,401,31,181),
        _c("Asparagus",         20, 2.2,0.1,  3.9,2.1,   2,1.9,  0,"vegetables",2.1,202,5,24),
        _c("Mushrooms",         22, 3.1,0.3,  3.3,1.0,   5,2.0,  0,"vegetables",0.5,318,2,3),
        _c("Blueberries",       57, 0.7,0.3, 14.5,2.4,   1,10.0, 0,"fruits",0.3,77,9,6),
        _c("Avocado",          160, 2.0,14.7, 8.5,6.7,   7,0.7,  0,"fruits",0.6,485,10,12),
        _c("Banana",            89, 1.1,0.3, 22.8,2.6,   1,12.2, 0,"fruits",0.3,358,9,5),
        _c("Apple",             52, 0.3,0.2, 13.8,2.4,   1,10.4, 0,"fruits",0.1,107,5,6),
        _c("Pomegranate",       83, 1.7,1.2, 18.7,4.0,   3,13.7, 0,"fruits",0.3,236,10,10),
        _c("Kiwi",              61, 1.1,0.5, 14.7,3.0,   3,9.0,  0,"fruits",0.3,312,93,34),
        _c("Lentils",          116, 9.0,0.4, 20.0,7.9,   2,1.8,  0,"legumes",3.3,369,4,19),
        _c("Chickpeas",        164, 8.9,2.6, 27.0,7.6,   7,4.8,  0,"legumes",2.9,291,1,49),
        _c("Edamame",          121,11.9,5.2,  8.9,5.2,   6,2.2,  0,"legumes",2.3,482,12,63),
        _c("Salmon",           208,20.4,13.4, 0.0,0.0,  59,0.0, 63,"proteins",0.8,628,0,13),
        _c("Sardines",         208,24.6,11.5, 0.0,0.0, 307,0.0,142,"proteins",2.9,397,0,382),
        _c("Eggs",             155,13.0,11.0, 1.1,0.0, 124,1.1,373,"proteins",1.8,138,0,56),
        _c("Almonds",          164, 6.0,14.2, 5.9,3.5,   1,1.1,  0,"proteins",3.7,733,0,264),
        _c("Walnuts",          185, 4.3,18.5, 3.9,1.9,   1,0.7,  0,"proteins",2.9,441,1,45),
        _c("Chia Seeds",       138, 4.7,8.7, 12.0,9.8,   5,0.0,  0,"proteins",7.7,407,1,631),
        _c("Greek Yogurt",      59,10.2,0.4,  3.6,0.0,  36,3.2,  5,"dairy",0.1,141,0,110),
        _c("Oatmeal",           68, 2.4,1.4, 12.0,1.7,  49,0.0,  0,"grains",1.2,143,0,12),
        _c("Quinoa",           120, 4.4,1.9, 21.3,2.8,   7,0.9,  0,"grains",1.5,318,0,17),
        _c("Brown Rice",       216, 5.0,1.8, 45.0,3.5,  10,0.7,  0,"grains",1.5,154,0,23),
        _c("Barley",           354,12.5,2.3, 73.5,17.3, 12,0.8,  0,"grains",3.6,452,0,33),
        _c("Cottage Cheese",    98,11.1,4.3,  3.4,0.0, 364,2.7, 17,"dairy",0.1,84,0,83),
        _c("Flaxseeds",        150, 5.1,12.0, 8.1,7.7,   8,0.4,  0,"proteins",11.9,255,1,255),
    ],
}

# Map goal keys to curated seed conditions for fallback
_GOAL_SEED_MAP = {
    "Weight Loss": "Obesity",
    "Weight Gain": "Underweight",
    "Gym & Muscle": "Normal",
    "Skin Health": "Normal",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CURATED RECIPES
# ══════════════════════════════════════════════════════════════════════════════

CURATED_RECIPES: dict[str, dict] = {
    "Spinach":          {"recipe_name":"Garlic Sautéed Spinach","ingredients":["2 cups fresh spinach","2 garlic cloves minced","1 tsp olive oil","Juice of ½ lemon","Salt and pepper"],"recipe":"Heat oil, sauté garlic 30 sec. Add spinach, toss until wilted (2–3 min). Squeeze lemon, season. Serve as side or over eggs."},
    "Broccoli":         {"recipe_name":"Oven-Roasted Broccoli","ingredients":["2 cups broccoli florets","1 tbsp olive oil","3 garlic cloves","Lemon zest","Salt and pepper"],"recipe":"Preheat 200°C. Toss florets with oil, garlic, salt. Roast 20–25 min until crispy-edged. Finish with lemon zest."},
    "Kale":             {"recipe_name":"Crispy Baked Kale Chips","ingredients":["2 cups kale","1 tbsp olive oil","Salt","Garlic powder","Nutritional yeast (optional)"],"recipe":"Preheat 150°C. Remove stems, tear kale, massage with oil and seasoning. Bake 15–20 min until completely crispy."},
    "Cauliflower":      {"recipe_name":"Cauliflower Rice Bowl","ingredients":["1 head cauliflower","1 tbsp olive oil","½ tsp turmeric","Cumin","Fresh coriander"],"recipe":"Pulse cauliflower until rice-like. Sauté in oil with spices 5–7 min. Top with coriander and lemon juice."},
    "Bitter Gourd":     {"recipe_name":"Bitter Gourd Stir-Fry","ingredients":["200g bitter gourd","1 onion","2 garlic cloves","1 tsp cumin","Turmeric"],"recipe":"Salt gourd 10 min, squeeze. Sauté onion and garlic, add gourd, cumin, turmeric. Stir-fry 8–10 min. Contains charantin — clinically shown to lower blood glucose."},
    "Fenugreek Seeds":  {"recipe_name":"Fenugreek Sprout Salad","ingredients":["2 tbsp fenugreek seeds","Spinach leaves","Cherry tomatoes","Lemon juice","1 tsp olive oil"],"recipe":"Soak seeds overnight, drain, sprout 1 day. Toss with spinach, tomatoes, lemon. Clinically proven to improve insulin sensitivity."},
    "Asparagus":        {"recipe_name":"Grilled Asparagus","ingredients":["200g asparagus","1 tbsp olive oil","Lemon juice","Parmesan shavings","Black pepper"],"recipe":"Toss asparagus with oil and pepper. Grill or roast at 200°C for 10–12 min. Finish with lemon and parmesan. High in folate and prebiotic fiber."},
    "Celery":           {"recipe_name":"Celery Almond Snack","ingredients":["4 celery stalks","3 tbsp almond butter","Sesame seeds"],"recipe":"Fill celery sticks with almond butter, sprinkle sesame seeds. 95% water content, virtually calorie-free — ideal for weight management."},
    "Beets":            {"recipe_name":"Roasted Beet Salad","ingredients":["2 beets","2 cups spinach","30g walnuts","1 tbsp olive oil","Balsamic vinegar"],"recipe":"Roast foil-wrapped beets at 200°C 45 min. Cool, peel, slice. Toss spinach with oil and vinegar. Top with beets and walnuts."},
    "Garlic":           {"recipe_name":"Garlic Roasted Vegetables","ingredients":["Mixed vegetables","6 garlic cloves","3 tbsp olive oil","Fresh rosemary","Salt and pepper"],"recipe":"Toss vegetables and whole garlic with oil and rosemary. Roast at 200°C 30–35 min. Garlic contains allicin — proven to lower blood pressure and LDL."},
    "Sweet Potato":     {"recipe_name":"Baked Sweet Potato with Yogurt","ingredients":["1 large sweet potato","2 tbsp Greek yogurt","Cumin and paprika","Coriander"],"recipe":"Pierce potato, rub with oil and spices. Bake 200°C 45 min. Fluff inside, top with yogurt and coriander."},
    "Tomatoes":         {"recipe_name":"Roasted Tomato Soup","ingredients":["500g tomatoes","1 onion","3 garlic cloves","1 tbsp olive oil","Fresh basil","Vegetable stock"],"recipe":"Roast tomatoes, onion, garlic at 200°C 30 min. Blend with stock and basil. Rich in lycopene — protects against UV damage and supports collagen."},
    "Mushrooms":        {"recipe_name":"Garlic Butter Mushrooms","ingredients":["300g mushrooms","2 tbsp olive oil","3 garlic cloves","Fresh thyme","Parsley"],"recipe":"High heat, add mushrooms in single layer. Don't stir 2 min until golden. Add garlic and thyme 2 more min. Rich in selenium and B vitamins."},
    "Zucchini":         {"recipe_name":"Zucchini Noodles Pesto","ingredients":["2 large zucchinis","2 tbsp basil pesto","Cherry tomatoes","Parmesan","Pine nuts"],"recipe":"Spiralise zucchini. Toss with pesto and tomatoes. Top with parmesan and pine nuts. Only 17 kcal/100g — excellent pasta alternative for weight loss."},
    "Lentils":          {"recipe_name":"Spiced Red Lentil Dal","ingredients":["1 cup red lentils","1 onion","2 garlic cloves","1 tsp turmeric","1 tsp cumin"],"recipe":"Simmer lentils in 2.5 cups water with turmeric 15 min. Fry cumin, onion, garlic until golden. Pour over lentils, simmer 5 min more. Serve with brown rice."},
    "Chickpeas":        {"recipe_name":"Roasted Spiced Chickpeas","ingredients":["400g canned chickpeas","1 tbsp olive oil","1 tsp cumin","½ tsp paprika","Salt and pepper"],"recipe":"Pat chickpeas dry. Toss with oil and spices. Roast 200°C 25–30 min until golden and crispy. Excellent snack or salad topper."},
    "Black Beans":      {"recipe_name":"Black Bean Tacos","ingredients":["400g black beans","Corn tortillas","Avocado","Salsa","Lime juice","Coriander"],"recipe":"Warm beans with cumin and garlic. Fill tortillas with beans, sliced avocado, salsa, lime and coriander. High in fiber and antioxidant anthocyanins."},
    "Kidney Beans":     {"recipe_name":"Spiced Kidney Bean Curry","ingredients":["400g kidney beans","1 onion","2 tomatoes","2 tsp curry powder","Light coconut milk"],"recipe":"Sauté onion, add curry powder 1 min. Add tomatoes 5 min. Add beans and coconut milk. Simmer 15 min. Very high in iron and fiber."},
    "Tofu":             {"recipe_name":"Crispy Pan-Fried Tofu","ingredients":["200g firm tofu","1 tbsp low-sodium soy sauce","1 tsp sesame oil","Garlic","Spring onions"],"recipe":"Press tofu dry, cube. Fry on high heat 3–4 min per side until golden. Add sauce last minute. Serve with brown rice and stir-fried vegetables."},
    "Edamame":          {"recipe_name":"Sesame Edamame","ingredients":["200g edamame","1 tsp sesame oil","Low-sodium soy sauce","Sesame seeds","Chilli flakes"],"recipe":"Steam edamame 3–4 min. Toss with oil, soy sauce, sesame seeds and chilli. Complete plant protein — all essential amino acids."},
    "Pumpkin Seeds":    {"recipe_name":"Toasted Pumpkin Seed Snack","ingredients":["½ cup pumpkin seeds","1 tsp olive oil","¼ tsp cumin","Pinch chilli","Salt"],"recipe":"Toss seeds with oil and spices. Roast 180°C 8–10 min until golden. One of the highest plant iron sources — 15mg/100g."},
    "Sesame Seeds":     {"recipe_name":"Sesame Energy Balls","ingredients":["½ cup sesame seeds","½ cup oats","3 tbsp tahini","2 tbsp honey","Sea salt"],"recipe":"Toast sesame seeds 2 min. Mix all ingredients. Form into balls. Refrigerate 30 min. Extremely high in iron, calcium and zinc."},
    "Salmon":           {"recipe_name":"Herb-Baked Salmon","ingredients":["150g salmon fillet","1 tbsp olive oil","Garlic","Fresh dill or parsley","Lemon slices"],"recipe":"Brush salmon with oil-garlic-herb mix. Bake 200°C 12–15 min. Serve with quinoa and steamed greens. Rich in omega-3 EPA and DHA."},
    "Sardines":         {"recipe_name":"Sardine Toast with Tomato","ingredients":["1 can sardines in water","2 slices whole-grain bread","2 tomatoes","Lemon juice","Parsley"],"recipe":"Toast bread. Layer tomatoes, top with sardines. Add lemon and parsley. Outstanding source of omega-3, heme iron, calcium and B12."},
    "Mackerel":         {"recipe_name":"Grilled Mackerel with Herbs","ingredients":["150g mackerel","Fresh dill","Lemon","1 tbsp olive oil","Black pepper"],"recipe":"Score skin, rub with oil and dill. Grill skin-side down 4–5 min, flip 3 min more. Top with lemon. Higher EPA content than salmon per gram."},
    "Tuna":             {"recipe_name":"Tuna Chickpea Salad","ingredients":["1 can tuna","400g chickpeas","½ red onion","Cherry tomatoes","Lemon-olive oil dressing"],"recipe":"Mix drained chickpeas, tuna, onion, halved tomatoes. Dress with oil, lemon, parsley. Provides both heme and non-heme iron plus Vit C."},
    "Eggs":             {"recipe_name":"Spinach & Egg Scramble","ingredients":["2 eggs","1 cup spinach","¼ onion diced","1 tsp olive oil","Salt and herbs"],"recipe":"Cook onion 2 min, wilt spinach 1 min. Pour whisked eggs, stir gently on low heat until just set. Serve on whole-grain toast."},
    "Chicken Breast":   {"recipe_name":"Lemon-Herb Grilled Chicken","ingredients":["150g chicken breast","Juice of 1 lemon","2 garlic cloves","Mixed herbs","1 tbsp olive oil"],"recipe":"Marinate in lemon, garlic, herbs and oil 30 min. Grill 6–7 min per side (74°C internal). Rest 5 min before slicing."},
    "Turkey Breast":    {"recipe_name":"Turkey Lettuce Wraps","ingredients":["200g lean turkey mince","Romaine lettuce leaves","1 onion","Garlic","Low-sodium soy sauce","Sesame oil"],"recipe":"Sauté onion and garlic in sesame oil. Cook turkey through, season with soy sauce. Spoon into lettuce leaves."},
    "Almonds":          {"recipe_name":"Overnight Almond Oats","ingredients":["½ cup oats","1 cup almond milk","2 tbsp almond butter","1 tsp honey","Berries"],"recipe":"Stir oats, milk and almond butter in jar. Refrigerate overnight. Top with almonds and berries in morning. No cooking needed."},
    "Walnuts":          {"recipe_name":"Walnut Banana Smoothie","ingredients":["1 banana","30g walnuts","1 cup milk","1 tsp honey","Cinnamon"],"recipe":"Blend all until smooth. Top with crushed walnuts. Rich in omega-3 ALA and potassium — excellent for heart health."},
    "Cashews":          {"recipe_name":"Cashew Vegetable Stir-Fry","ingredients":["100g cashews","Mixed vegetables","Soy sauce","Ginger","Garlic","Sesame oil"],"recipe":"Toast cashews in dry pan 2–3 min. Stir-fry vegetables with ginger and garlic in sesame oil 5 min. Add cashews and soy sauce. Serve over brown rice."},
    "Pistachio":        {"recipe_name":"Pistachio-Crusted Salmon","ingredients":["150g salmon","3 tbsp crushed pistachios","1 tbsp Dijon mustard","Lemon zest","Olive oil"],"recipe":"Brush salmon with mustard. Press crushed pistachios on top. Bake 200°C 12–15 min. Highest potassium of any nut — excellent for blood pressure."},
    "Chia Seeds":       {"recipe_name":"Chia Seed Pudding","ingredients":["3 tbsp chia seeds","1 cup coconut milk","1 tsp vanilla","Mango and berries","1 tsp honey"],"recipe":"Whisk chia into coconut milk with vanilla and honey. Refrigerate overnight. Top with mango and berries. Rich in omega-3, fiber and calcium."},
    "Flaxseeds":        {"recipe_name":"Flaxseed Berry Smoothie","ingredients":["2 tbsp ground flaxseeds","1 cup mixed berries","1 banana","1 cup almond milk","1 tsp honey"],"recipe":"Grind flaxseeds first. Blend all together. Drink immediately. Richest plant source of omega-3 ALA — clinically reduces LDL cholesterol."},
    "Peanut Butter":    {"recipe_name":"Peanut Butter Banana Toast","ingredients":["2 slices whole-grain bread","2 tbsp peanut butter","1 banana","Chia seeds","Cinnamon"],"recipe":"Toast bread. Spread peanut butter generously. Layer banana slices. Sprinkle chia seeds and cinnamon. High in protein and healthy fats."},
    "Greek Yogurt":     {"recipe_name":"Berry Parfait","ingredients":["200g plain Greek yogurt","½ cup mixed berries","2 tbsp oats","1 tsp honey","Walnuts"],"recipe":"Layer yogurt, berries, oats, honey and walnuts in a glass. Serve chilled. High-protein probiotic-rich snack excellent for gut and blood sugar control."},
    "Cottage Cheese":   {"recipe_name":"Cottage Cheese with Berries","ingredients":["200g low-fat cottage cheese","½ cup mixed berries","1 tbsp honey","Crushed walnuts","Cinnamon"],"recipe":"Spoon cottage cheese into bowl. Top with berries, honey, walnuts and cinnamon. High in casein protein — ideal bedtime snack for overnight muscle repair."},
    "Oatmeal":          {"recipe_name":"Heart-Healthy Porridge","ingredients":["½ cup oats","1 cup milk","1 banana","1 tbsp chia seeds","Cinnamon"],"recipe":"Boil milk, add oats, cook 5 min stirring. Top with banana, chia and cinnamon. No added sugar. Beta-glucan actively lowers LDL cholesterol."},
    "Quinoa":           {"recipe_name":"Quinoa Vegetable Bowl","ingredients":["½ cup quinoa","1 cup water","Roasted vegetables","Chickpeas","Tahini-lemon dressing"],"recipe":"Cook quinoa: boil, cover, simmer 15 min. Top with roasted vegetables, chickpeas and tahini-lemon dressing. Complete protein with all 9 essential amino acids."},
    "Brown Rice":       {"recipe_name":"Brown Rice Vegetable Stir-Fry","ingredients":["1 cup cooked brown rice","Mixed vegetables","2 eggs","Low-sodium soy sauce","Sesame oil"],"recipe":"Stir-fry vegetables in sesame oil. Scramble eggs alongside. Add rice and soy sauce. Stir-fry 2–3 min more. 3× more fiber than white rice."},
    "Barley":           {"recipe_name":"Mushroom Barley Soup","ingredients":["½ cup pearl barley","200g mushrooms","1 onion","2 garlic cloves","Vegetable stock","Thyme"],"recipe":"Sauté onion and garlic. Add mushrooms 5 min. Add barley and stock. Simmer 40 min until tender. Highest beta-glucan content of any grain."},
    "Avocado":          {"recipe_name":"Avocado Toast with Poached Egg","ingredients":["1 ripe avocado","2 slices whole-grain bread","2 eggs","Chilli flakes","Lemon juice"],"recipe":"Toast bread. Mash avocado with lemon, salt. Spread on toast. Poach eggs 3–4 min. Place on top with chilli flakes."},
    "Banana":           {"recipe_name":"Banana Almond Smoothie","ingredients":["2 bananas","2 tbsp almond butter","1 cup milk","Cinnamon","Ice"],"recipe":"Blend until smooth and creamy. Rich in potassium and magnesium — especially beneficial for blood pressure management."},
    "Blueberries":      {"recipe_name":"Antioxidant Smoothie","ingredients":["1 cup blueberries","200g Greek yogurt","½ cup oats","1 tsp honey","1 cup almond milk"],"recipe":"Blend all until smooth. Rich in anthocyanins reducing oxidative stress. One of the best evidence-backed superfoods for diabetes and heart health."},
    "Pomegranate":      {"recipe_name":"Pomegranate Green Salad","ingredients":["Pomegranate seeds","2 cups mixed greens","30g walnuts","1 tbsp olive oil","Lemon juice"],"recipe":"Toss greens with oil and lemon. Top with pomegranate seeds and walnuts. Vit C boosts non-heme iron absorption from plant foods."},
    "Strawberries":     {"recipe_name":"Strawberry Chia Pudding","ingredients":["1 cup strawberries","3 tbsp chia seeds","1 cup almond milk","1 tsp honey","Mint"],"recipe":"Blend half strawberries with milk and honey. Stir in chia seeds. Refrigerate 4+ hours. High Vit C boosts iron absorption."},
    "Raspberries":      {"recipe_name":"Raspberry Overnight Oats","ingredients":["½ cup oats","1 cup almond milk","½ cup raspberries","1 tsp honey","1 tbsp chia seeds"],"recipe":"Mix oats, milk, chia in jar. Refrigerate overnight. Top with raspberries and honey. Highest fiber of any berry — 6.5g/100g."},
    "Apple":            {"recipe_name":"Baked Cinnamon Apple","ingredients":["2 apples","1 tsp cinnamon","1 tbsp honey","2 tbsp oats","1 tsp coconut oil"],"recipe":"Core apples, fill with oats, honey, cinnamon and coconut oil. Bake 180°C 25 min. Rich in quercetin and pectin — lowers LDL and stabilises blood sugar."},
    "Kiwi":             {"recipe_name":"Kiwi Berry Smoothie Bowl","ingredients":["3 kiwis","½ cup blueberries","200g Greek yogurt","1 tbsp chia seeds","Granola"],"recipe":"Blend 2 kiwis with yogurt. Pour into bowl. Top with sliced kiwi, blueberries and chia. More Vit C per serve than an orange."},
    "Watermelon":       {"recipe_name":"Watermelon Feta Salad","ingredients":["3 cups watermelon","50g feta","Fresh mint","Lime juice","Pinch chilli flakes"],"recipe":"Combine watermelon and mint. Crumble feta, squeeze lime, sprinkle chilli. Contains L-citrulline — natural vasodilator that lowers blood pressure."},
    "Mango":            {"recipe_name":"Mango Lassi","ingredients":["1 ripe mango","200g Greek yogurt","½ cup milk","1 tsp honey","Cardamom"],"recipe":"Blend all until smooth. Serve chilled. Rich in Vit C, A and folate. Natural sugars with yogurt protein create sustained energy release."},
    "Dates":            {"recipe_name":"Date Energy Balls","ingredients":["10 Medjool dates","1 cup oats","3 tbsp peanut butter","2 tbsp chia seeds","2 tbsp cocoa"],"recipe":"Blend dates until paste forms. Mix in oats, peanut butter, chia, cocoa. Roll into balls. Refrigerate 30 min. Calorie-dense healthy snack."},
    "Dried Apricots":   {"recipe_name":"Apricot Trail Mix","ingredients":["½ cup dried apricots","¼ cup almonds","¼ cup pumpkin seeds","Pinch sea salt"],"recipe":"Mix and portion into small bags. 6mg iron per 100g and high beta-carotene. Pair with Vit C foods to maximise iron absorption."},
    "Pear":             {"recipe_name":"Spiced Pear Salad","ingredients":["2 pears sliced","2 cups rocket","30g blue cheese","30g walnuts","Honey-lemon dressing"],"recipe":"Arrange rocket. Layer pear slices, crumble cheese, scatter walnuts. Drizzle honey-lemon-olive oil dressing. High in soluble pectin that lowers cholesterol."},
    "Grapefruit":       {"recipe_name":"Grapefruit Mint Salad","ingredients":["1 large grapefruit","Fresh mint","1 tsp honey","Pinch chilli flakes"],"recipe":"Segment grapefruit, top with mint, honey and chilli. 42 kcal/100g — enzymes support fat metabolism and insulin sensitivity."},
    "Sweet Pepper":     {"recipe_name":"Stuffed Bell Peppers","ingredients":["4 bell peppers","1 cup cooked quinoa","1 can black beans","Cumin","Cheese","Coriander"],"recipe":"Fill hollowed peppers with quinoa-bean mix. Top with cheese. Bake 190°C 30–35 min. More Vit C than oranges — 184mg/100g."},
    "Collard Greens":   {"recipe_name":"Braised Collard Greens","ingredients":["2 cups collard greens","2 garlic cloves","1 tsp olive oil","Lemon juice","Red pepper flakes"],"recipe":"Sauté garlic, add greens with splash of water. Cover, cook 8–10 min until tender. Add lemon and chilli. High in iron, calcium and folate."},
    "Swiss Chard":      {"recipe_name":"Sautéed Swiss Chard","ingredients":["2 cups swiss chard","2 garlic cloves","1 tsp olive oil","Lemon juice","Pinch nutmeg"],"recipe":"Cook chard stems first 2 min, add leaves and garlic 3–4 min until wilted. Season with lemon and nutmeg. Very high in Vit K, C, magnesium and iron."},
    "Amaranth":         {"recipe_name":"Amaranth Porridge","ingredients":["½ cup amaranth","1.5 cups water","1 cup almond milk","Fresh berries","1 tsp honey","Cinnamon"],"recipe":"Boil amaranth in water 20 min covered. Add almond milk, stir until creamy. Top with berries, honey and cinnamon. 14.5g complete protein per 100g — higher than any other grain."},
    "Pistachio":        {"recipe_name":"Pistachio-Crusted Salmon","ingredients":["150g salmon","3 tbsp crushed pistachios","1 tbsp Dijon mustard","Lemon zest","Olive oil"],"recipe":"Brush salmon with mustard. Press crushed pistachios on top. Drizzle oil. Bake 200°C 12–15 min. Highest potassium of any nut — excellent for blood pressure."},
    "Pumpkin Seeds":    {"recipe_name":"Toasted Pumpkin Seed Snack","ingredients":["½ cup raw pumpkin seeds","1 tsp olive oil","¼ tsp cumin","Pinch chilli","Salt"],"recipe":"Toss with oil and spices. Roast 180°C 8–10 min until golden. Outstanding plant iron source — 15mg/100g."},
    "Whole Wheat Pasta":{"recipe_name":"Whole Wheat Pasta Primavera","ingredients":["200g whole wheat pasta","Mixed vegetables","2 garlic cloves","Olive oil","Parmesan","Fresh basil"],"recipe":"Cook pasta al dente. Sauté garlic, add vegetables 5 min. Toss with pasta, parmesan and basil. 3× more fiber than white pasta with lower GI."},
    "Lettuce":          {"recipe_name":"Chicken Lettuce Wraps","ingredients":["Romaine lettuce leaves","150g cooked chicken diced","Cherry tomatoes","Avocado","Lemon-tahini dressing"],"recipe":"Fill large lettuce leaves with chicken, tomatoes and avocado. Drizzle tahini-lemon dressing. Only 15 kcal/100g — perfect calorie-controlled meal."},
    "Cucumber":         {"recipe_name":"Cucumber Yogurt Raita","ingredients":["1 large cucumber grated","200g Greek yogurt","1 tsp cumin","Fresh mint","Salt"],"recipe":"Grate cucumber, squeeze dry. Mix with yogurt, cumin, mint and salt. Chill 15 min. 16 kcal/100g — excellent for hydration and weight control."},
    "Eggplant":         {"recipe_name":"Baked Stuffed Eggplant","ingredients":["2 large eggplants","1 can chopped tomatoes","½ cup cooked quinoa","1 onion","Garlic","Fresh basil"],"recipe":"Halve eggplants, scoop and sauté flesh with onion, garlic. Mix with tomatoes and quinoa. Fill shells. Bake 180°C 30 min."},
    "Sunflower Seeds":  {"recipe_name":"Sunflower Seed Salad Topper","ingredients":["3 tbsp sunflower seeds","2 cups mixed greens","Cucumber","Cherry tomatoes","Olive oil dressing"],"recipe":"Toast seeds in dry pan 2–3 min. Top salad with greens, cucumber, tomatoes. Dress with olive oil and lemon. Rich in Vit E and selenium."},
}


# ══════════════════════════════════════════════════════════════════════════════
#  NATURAL SKINCARE RECIPES  (topical use of food ingredients)
# ══════════════════════════════════════════════════════════════════════════════

SKINCARE_GOALS = {
    "✨ Glowing Skin": {
        "color": "#fbbf24",
        "desc": "Brighten, even skin tone and boost natural radiance",
        "foods_to_eat": ["Blueberries","Kiwi","Pomegranate","Spinach","Sweet Pepper","Avocado","Carrots","Tomatoes"],
        "key_nutrients": "Vitamin C · Lycopene · Beta-carotene · Anthocyanins",
        "eat_tip": "Eat 1 cup of berries and 1 citrus fruit daily for 4–6 weeks to visibly brighten skin.",
        "recipes": [
            {
                "name": "Turmeric Honey Glow Mask",
                "emoji": "🌟",
                "ingredients": ["½ tsp turmeric powder","1 tbsp raw honey","1 tbsp Greek yogurt","2 drops lemon juice"],
                "how_to": "Mix all ingredients into a paste. Apply evenly to clean face, avoiding eyes. Leave 15–20 min. Rinse with lukewarm water. Use 2–3×/week.",
                "why_it_works": "Turmeric's curcumin inhibits melanin production, reducing dark spots. Honey is a humectant and antibacterial. Yogurt's lactic acid gently exfoliates for instant brightness.",
                "frequency": "2–3× per week",
            },
            {
                "name": "Vitamin C Papaya Enzyme Mask",
                "emoji": "🍈",
                "ingredients": ["3 tbsp ripe papaya mashed","1 tsp honey","1 tsp lemon juice","1 tsp oatmeal (sensitive skin substitute lemon)"],
                "how_to": "Mash papaya until smooth. Add honey, lemon juice and oatmeal. Apply to face and neck. Leave 20 min. Rinse with cool water and moisturise.",
                "why_it_works": "Papain enzyme in papaya dissolves dead skin cells for a natural chemical exfoliation. Vitamin C from lemon visibly brightens in 2–4 weeks of regular use.",
                "frequency": "1–2× per week",
            },
            {
                "name": "Pomegranate Brightening Toner",
                "emoji": "🫀",
                "ingredients": ["3 tbsp fresh pomegranate juice","1 tbsp rosewater","1 tsp aloe vera gel"],
                "how_to": "Mix ingredients. Apply to clean face with a cotton pad after cleansing. No rinse needed. Store in fridge up to 5 days.",
                "why_it_works": "Pomegranate punicalagins are among the most potent antioxidants — they neutralise UV-generated free radicals and promote even skin tone.",
                "frequency": "Daily, after cleansing",
            },
        ],
    },
    "⏳ Anti-Aging": {
        "color": "#818cf8",
        "desc": "Reduce fine lines, firm skin and protect against oxidative aging",
        "foods_to_eat": ["Avocado","Blueberries","Salmon","Walnuts","Pomegranate","Kale","Tomatoes","Raspberries"],
        "key_nutrients": "Omega-3 · Vitamin E · Resveratrol · Lycopene · Collagen precursors",
        "eat_tip": "Eat fatty fish 3×/week and a handful of berries daily — omega-3 reduces inflammatory breakdown of collagen.",
        "recipes": [
            {
                "name": "Avocado & Vitamin E Anti-Aging Mask",
                "emoji": "🥑",
                "ingredients": ["½ ripe avocado mashed","1 tbsp olive oil","1 tsp honey","1 vitamin E capsule (pierce and squeeze)"],
                "how_to": "Mash avocado until smooth. Mix in olive oil, honey and vitamin E. Apply generously to face and neck. Leave 25–30 min. Rinse with warm water, pat dry.",
                "why_it_works": "Avocado is uniquely rich in oleic acid and beta-sitosterol — proven to penetrate deep layers and restore lipid barrier. Vitamin E directly neutralises free radicals that accelerate wrinkle formation.",
                "frequency": "2× per week",
            },
            {
                "name": "Coffee Collagen-Boost Scrub",
                "emoji": "☕",
                "ingredients": ["2 tbsp used coffee grounds","1 tbsp coconut oil","1 tbsp brown sugar","Few drops of rosehip oil"],
                "how_to": "Mix into a thick scrub. Gently massage onto damp face in circular motions for 2–3 min. Rinse with warm water. Follow with your moisturiser.",
                "why_it_works": "Caffeine temporarily tightens blood vessels reducing puffiness. Coffee's antioxidants fight photoaging. Rosehip oil contains natural retinoids (Vit A) that stimulate collagen synthesis.",
                "frequency": "2× per week (avoid if sensitive skin)",
            },
            {
                "name": "Green Tea Collagen Serum",
                "emoji": "🍵",
                "ingredients": ["2 green tea bags brewed strong","1 tbsp aloe vera gel","½ tsp rosehip oil","1 tsp glycerin"],
                "how_to": "Brew tea, cool completely. Mix with aloe, rosehip and glycerin. Pour into a small spray bottle. Apply 2–3 sprays after cleansing, before moisturiser.",
                "why_it_works": "Green tea EGCG catechins are the strongest botanical antioxidants — shown in studies to protect against UV-induced collagen degradation. Rosehip oil provides natural tretinoin-like compounds.",
                "frequency": "Daily, morning and evening",
            },
        ],
    },
    "🌿 Acne Control": {
        "color": "#34d399",
        "desc": "Control sebum, reduce inflammation and prevent breakouts",
        "foods_to_eat": ["Spinach","Broccoli","Blueberries","Eggs","Pumpkin Seeds","Greek Yogurt","Lentils","Almonds"],
        "key_nutrients": "Zinc · Omega-3 · Probiotics · Selenium · Vitamin A",
        "eat_tip": "Reduce dairy and high-GI foods. Zinc from pumpkin seeds (15mg/100g) is as effective as low-dose tetracycline for reducing acne lesions.",
        "recipes": [
            {
                "name": "Neem & Honey Anti-Acne Mask",
                "emoji": "🌿",
                "ingredients": ["1 tbsp neem powder (or brew neem tea)","1 tbsp raw honey","½ tsp turmeric","Few drops of tea tree oil"],
                "how_to": "Mix into a smooth paste. Apply to acne-prone areas. Leave 15 min. Rinse with cool water. Do not use on broken skin.",
                "why_it_works": "Neem's nimbidin and gedunin are clinically proven antibacterials that inhibit P. acnes bacteria. Tea tree oil (at 5% concentration) matches 5% benzoyl peroxide efficacy in trials.",
                "frequency": "3–4× per week",
            },
            {
                "name": "Oatmeal & Chamomile Calming Mask",
                "emoji": "🌾",
                "ingredients": ["2 tbsp finely ground oats","1 tbsp brewed chamomile tea (cooled)","1 tbsp aloe vera gel","1 tsp honey"],
                "how_to": "Mix ground oats with chamomile tea to form a paste. Add aloe and honey. Apply gently to clean face. Leave 20 min. Rinse with lukewarm water.",
                "why_it_works": "Oat avenanthramides are potent anti-inflammatory compounds that reduce redness within minutes. Chamomile's bisabolol soothes irritated skin. Aloe regulates sebum production.",
                "frequency": "3× per week",
            },
            {
                "name": "Apple Cider Vinegar & Tea Tree Toner",
                "emoji": "🍎",
                "ingredients": ["1 tbsp apple cider vinegar","3 tbsp water","2 drops tea tree oil","1 tsp witch hazel"],
                "how_to": "Mix diluted ACV with water, tea tree oil and witch hazel. Apply with cotton pad to clean face after cleansing. Avoid eye area. Do not rinse.",
                "why_it_works": "ACV restores skin's natural acidic pH (4.5–5.5) which inhibits bacterial growth. Tea tree oil is clinically proven to reduce both inflammatory and non-inflammatory acne lesions.",
                "frequency": "Once daily (evening)",
            },
        ],
    },
    "💧 Dry & Dull Skin": {
        "color": "#38bdf8",
        "desc": "Deeply hydrate, restore barrier and revive radiance",
        "foods_to_eat": ["Avocado","Salmon","Walnuts","Chia Seeds","Cucumber","Watermelon","Greek Yogurt","Flaxseeds"],
        "key_nutrients": "Omega-3 · Omega-6 · Hyaluronic acid precursors · Glycerin · Ceramides",
        "eat_tip": "Drink 2.5L water daily. Add 1 tbsp flaxseeds to meals — their omega-3 ALA significantly improves skin hydration within 6 weeks.",
        "recipes": [
            {
                "name": "Honey Banana Deep Moisture Mask",
                "emoji": "🍌",
                "ingredients": ["1 ripe banana mashed","1 tbsp raw honey","1 tbsp coconut oil","1 tsp aloe vera gel"],
                "how_to": "Mash banana until completely smooth. Mix in honey, coconut oil and aloe. Apply generously to dry areas including neck. Leave 25–30 min. Rinse with lukewarm water.",
                "why_it_works": "Banana potassium and natural sugar compounds act as humectants. Honey's hygroscopic nature draws moisture from the air into skin. Coconut oil's lauric acid restores lipid barrier function.",
                "frequency": "3× per week",
            },
            {
                "name": "Milk & Oat Soothing Bath Soak",
                "emoji": "🥛",
                "ingredients": ["1 cup whole milk powder","1 cup colloidal oatmeal (finely blended oats)","2 tbsp honey","10 drops lavender essential oil"],
                "how_to": "Mix all ingredients. Add to warm (not hot) bath water. Soak for 20–25 min. Pat skin dry gently — do not rinse off. Apply moisturiser immediately.",
                "why_it_works": "Milk's lactic acid gently dissolves dead skin cells. Oat beta-glucan forms a protective film on skin that holds moisture. Lavender reduces skin inflammation and sensitisation.",
                "frequency": "2× per week",
            },
            {
                "name": "Cucumber Aloe Hydrating Gel Mask",
                "emoji": "🥒",
                "ingredients": ["½ cucumber blended","3 tbsp aloe vera gel","1 tsp glycerin","1 tsp rosewater"],
                "how_to": "Blend cucumber, strain juice. Mix with aloe, glycerin and rosewater. Apply thick layer to clean face. Leave 20 min. Rinse with cool water. Use refrigerated for extra cooling.",
                "why_it_works": "Cucumber contains silica which promotes collagen cross-linking. Its 95% water content combined with aloe's mucopolysaccharides provides immediate and lasting hydration.",
                "frequency": "Daily (safe enough for everyday use)",
            },
        ],
    },
    "🌙 Dark Circles & Puffiness": {
        "color": "#a78bfa",
        "desc": "Reduce under-eye circles, depuff and brighten the eye area",
        "foods_to_eat": ["Cucumber","Watermelon","Blueberries","Spinach","Kale","Salmon","Eggs","Greek Yogurt"],
        "key_nutrients": "Vitamin K · Vitamin C · Caffeine (topical) · Iron · Retinol",
        "eat_tip": "Iron deficiency is a primary cause of dark circles. Ensure adequate iron intake through spinach, lentils and fortified foods alongside Vitamin C for absorption.",
        "recipes": [
            {
                "name": "Cold Cucumber Eye Pads",
                "emoji": "🥒",
                "ingredients": ["½ fresh cucumber","2 cotton pads soaked in cold green tea","1 tsp aloe vera gel (optional)"],
                "how_to": "Refrigerate cucumber 30 min. Cut into thin rounds. Place 2 slices directly over closed eyes. Additionally place cold tea pads if puffiness is significant. Rest for 15–20 min.",
                "why_it_works": "Cold temperature constricts blood vessels reducing puffiness within minutes. Cucumber's ascorbic acid and caffeic acid reduce inflammation and water retention around the eye area.",
                "frequency": "Daily (especially in the morning)",
            },
            {
                "name": "Coffee Under-Eye Treatment",
                "emoji": "☕",
                "ingredients": ["1 tsp fine coffee grounds","½ tsp coconut oil","Few drops of vitamin E oil"],
                "how_to": "Mix into a smooth paste. Gently dab (do NOT rub) under the eye area with fingertip. Leave 5–10 min maximum. Rinse gently with cool water. Follow with eye cream.",
                "why_it_works": "Topical caffeine constricts capillaries reducing the discoloration from blood pooling under thin under-eye skin. Antioxidants protect the delicate periorbital area from oxidative stress.",
                "frequency": "3–4× per week",
            },
            {
                "name": "Potato & Vitamin K Eye Mask",
                "emoji": "🥔",
                "ingredients": ["½ raw potato grated","2 cotton pads","1 tsp fresh lemon juice","1 tsp honey"],
                "how_to": "Grate potato, squeeze out juice. Mix juice with lemon and honey. Soak cotton pads in mixture. Apply to under-eye area. Leave 20 min. Rinse with cool water.",
                "why_it_works": "Raw potato contains catecholase enzyme which lightens hyperpigmentation. Vitamin K in potato juice has been shown in research to reduce capillary leakage that causes dark circles.",
                "frequency": "Daily for 4–6 weeks (results are gradual)",
            },
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  PERSONALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _score(item: dict, condition: str, gender: str, age: int, bmi_val: float) -> float:
    s = 0.0
    cal  = item.get("calories", 0)
    prot = item.get("protein",  0)
    iron = item.get("iron",     0)
    potk = item.get("potassium",0)
    vitc = item.get("vit_c",   0)
    fiber= item.get("fiber",   0)

    criteria = CONDITION_CRITERIA.get(condition, CONDITION_CRITERIA["Normal"])
    sort_by  = criteria.get("sort_by","nutrition_density")
    val = item.get(sort_by, 0) or 0
    s += float(val) * 0.5

    g = gender.lower()
    if g == "female":
        s += iron * 3.0  # women need more iron
        if condition == "Anemia": s += iron * 5.0
    elif g == "male":
        s += prot * 0.3
    if age >= 50:
        s += item.get("calcium", 0) * 0.01
        s += iron * 1.5
    elif age <= 25:
        s += prot * 0.2

    if bmi_val >= 30:
        s -= abs(cal - 80) * 0.02
    elif bmi_val < 18.5:
        s += cal * 0.01
        s += prot * 0.3

    s += random.uniform(0, 0.3)  # controlled variety
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET-DRIVEN SUPPLEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _dataset_foods(condition: str, exclude_names: set,
                   gender: str, age: int, bmi_val: float,
                   n: int = 15) -> list:
    """Pull n additional foods from the CSV dataset matching condition criteria."""
    df = load_food_dataset()
    if df.empty: return []

    criteria = CONDITION_CRITERIA.get(condition, CONDITION_CRITERIA["Normal"])
    fd = df.copy()

    # Apply numerical filters
    if "max_calories"    in criteria: fd = fd[fd["calories"]     <= criteria["max_calories"]]
    if "min_calories"    in criteria: fd = fd[fd["calories"]     >= criteria["min_calories"]]
    if "max_fat"         in criteria: fd = fd[fd["fat"]          <= criteria["max_fat"]]
    if "max_carbs"       in criteria and "carbs" in fd: fd = fd[fd["carbs"] <= criteria["max_carbs"]]
    if "max_sugars"      in criteria and "sugars" in fd: fd = fd[fd["sugars"] <= criteria["max_sugars"]]
    if "max_sodium"      in criteria: fd = fd[fd["sodium"]       <= criteria["max_sodium"]]
    if "max_cholesterol" in criteria and "cholesterol" in fd: fd = fd[fd["cholesterol"] <= criteria["max_cholesterol"]]
    if "min_protein"     in criteria: fd = fd[fd["protein"]      >= criteria["min_protein"]]
    if "min_fiber"       in criteria: fd = fd[fd["fiber"]        >= criteria["min_fiber"]]
    if "min_iron"        in criteria and "iron" in fd: fd = fd[fd["iron"] >= criteria.get("min_iron",0)]

    # Prefer categories
    prefer = criteria.get("prefer_cats", [])
    if prefer:
        pref_mask = fd["category"].isin(prefer)
        fd = pd.concat([fd[pref_mask], fd[~pref_mask]]).reset_index(drop=True)

    # Sort
    sort_col = criteria.get("sort_by", "nutrition_density")
    asc      = criteria.get("ascending", False)
    if sort_col in fd.columns:
        fd = fd.sort_values(sort_col, ascending=asc)

    results = []
    seen = set(n.lower()[:15] for n in exclude_names)
    for _, row in fd.iterrows():
        fname = str(row["food"])
        key   = fname.lower()[:15]
        if key in seen: continue
        if _is_junk(fname): continue
        seen.add(key)
        cat = row.get("category", "other")
        results.append({
            "food":        fname.title(),
            "calories":    round(float(row.get("calories", 0)), 1),
            "protein":     round(float(row.get("protein",  0)), 1),
            "fat":         round(float(row.get("fat",      0)), 1),
            "carbs":       round(float(row.get("carbs",    0)), 1),
            "fiber":       round(float(row.get("fiber",    0)), 1),
            "sodium":      round(float(row.get("sodium",   0)), 1),
            "sugars":      round(float(row.get("sugars",   0)), 1),
            "cholesterol": round(float(row.get("cholesterol", 0)), 1),
            "iron":        round(float(row.get("iron",     0)), 3),
            "potassium":   round(float(row.get("potassium",0)), 1),
            "vit_c":       round(float(row.get("vit_c",   0)), 1),
            "calcium":     round(float(row.get("calcium",  0)), 1),
            "nutrition_density": round(float(row.get("nutrition_density",0)),1),
            "category":    cat,
            "from_dataset": True,
        })
        if len(results) >= n: break
    return results


def _build_item(item: dict) -> dict:
    cat      = item.get("category", "other")
    raw_name = item["food"]
    # Clean "Broccoli Cooked" → "Broccoli" for display/image matching
    display  = _clean_food_name(raw_name) if item.get("from_dataset") else raw_name
    img      = get_food_image(display, cat)
    rec      = CURATED_RECIPES.get(raw_name) or CURATED_RECIPES.get(display)
    return {
        "food":              display,
        "food_raw":          raw_name,
        "calories":          round(float(item.get("calories",   0)), 1),
        "protein":           round(float(item.get("protein",    0)), 1),
        "fat":               round(float(item.get("fat",        0)), 1),
        "carbs":             round(float(item.get("carbs",      0)), 1),
        "fiber":             round(float(item.get("fiber",      0)), 1),
        "sodium":            round(float(item.get("sodium",     0)), 1),
        "sugars":            round(float(item.get("sugars",     0)), 1),
        "cholesterol":       round(float(item.get("cholesterol",0)), 1),
        "iron":              round(float(item.get("iron",       0)), 3),
        "potassium":         round(float(item.get("potassium",  0)), 1),
        "vit_c":             round(float(item.get("vit_c",      0)), 1),
        "calcium":           round(float(item.get("calcium",    0)), 1),
        "nutrition_density": round(float(item.get("nutrition_density", 0)), 1),
        "category":          cat,
        "from_dataset":      item.get("from_dataset", False),
        "recipe":            rec["recipe"]      if rec else _default_recipe(display),
        "recipe_name":       rec["recipe_name"] if rec else f"How to Prepare {display}",
        "ingredients":       rec["ingredients"] if rec else [],
        "image_svg":         img["svg"],
        "image_mdb":         img["mdb"],
    }


def _clean_food_name(food: str) -> str:
    """Strip common dataset suffixes like 'Cooked', 'Raw', 'Dried' for natural recipe text."""
    stop_suffixes = [" cooked"," raw"," dried"," canned"," boiled"," steamed",
                     " roasted"," frozen"," fresh"," baked"," grilled",
                     " whole"," sliced"," chopped"," crushed"]
    n = food.lower()
    for s in stop_suffixes:
        if n.endswith(s):
            food = food[:len(food)-len(s)].strip()
            break
    return food.title()


def _default_recipe(food: str) -> str:
    name = _clean_food_name(food)
    return random.choice([
        f"Steam {name} for 5–7 minutes until tender. Drizzle with extra virgin olive oil, "
        f"fresh lemon juice, cracked black pepper and a pinch of sea salt. "
        f"Serve warm as a side dish or toss into a salad.",
        f"Heat 1 tsp olive oil in a pan over medium heat. Add 2 minced garlic cloves and sauté "
        f"30 seconds until fragrant. Add {name} and cook 4–5 minutes. Season with cumin, "
        f"a squeeze of lemon juice and fresh herbs. Serve immediately.",
        f"Preheat oven to 200°C. Toss {name} with 1 tbsp olive oil, mixed herbs, salt and pepper. "
        f"Spread on a baking tray in a single layer. Roast 20–25 minutes until golden. "
        f"Great as a side dish, in grain bowls or added to salads.",
        f"Enjoy {name} as part of a balanced meal. Pair with a lean protein source and a "
        f"complex carbohydrate for a complete, nutritionally balanced plate. "
        f"Season lightly with herbs and olive oil to enhance natural flavour.",
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def recommend_food(condition: str, food_df=None, recipe_df=None,
                   top_n: int = 15,
                   gender: str = "Male", age: int = 30,
                   bmi_val: float = 22.0) -> list:
    """
    Return top_n personalised food recommendations for a health condition.
    Combines curated medical seeds + dataset-driven supplement.
    """
    seeds = list(CURATED_SEEDS.get(condition, CURATED_SEEDS["Normal"]))
    seeds.sort(key=lambda x: -_score(x, condition, gender, age, bmi_val))

    seen, result = set(), []
    for item in seeds:
        key = item["food"].lower()[:15]
        if key not in seen:
            seen.add(key)
            result.append(_build_item(item))
        if len(result) >= top_n:
            break

    # Supplement from dataset if we need more variety
    if len(result) < top_n:
        extras = _dataset_foods(condition, {x["food"] for x in result},
                                 gender, age, bmi_val, n=top_n - len(result) + 10)
        for item in extras:
            if len(result) >= top_n: break
            result.append(_build_item(item))

    return result[:top_n]


def recommend_food_for_goal(goal: str, top_n: int = 15,
                             gender: str = "Male", age: int = 30,
                             bmi_val: float = 22.0) -> list:
    """
    Return food recommendations for a wellness goal.
    Uses goal-specific criteria applied to the dataset.
    """
    # Use corresponding condition seeds as base
    base_cond = _GOAL_SEED_MAP.get(goal, "Normal")
    seeds = list(CURATED_SEEDS.get(base_cond, CURATED_SEEDS["Normal"]))

    # Apply goal-specific filtering on top
    gcrit = GOAL_CRITERIA.get(goal, {})
    filtered = []
    for s in seeds:
        if gcrit.get("min_protein", 0) and s.get("protein", 0) < gcrit["min_protein"]: continue
        if gcrit.get("max_calories") and s.get("calories", 0) > gcrit["max_calories"]: continue
        if gcrit.get("min_calories") and s.get("calories", 0) < gcrit.get("min_calories", 0): continue
        filtered.append(s)

    # Prefer goal categories
    prefer = set(gcrit.get("prefer_cats", []))
    filtered.sort(key=lambda x: (
        0 if x.get("category") in prefer else 1,
        -_score(x, base_cond, gender, age, bmi_val)
    ))

    seen, result = set(), []
    for item in filtered:
        key = item["food"].lower()[:15]
        if key not in seen:
            seen.add(key)
            result.append(_build_item(item))
        if len(result) >= top_n: break

    # Dataset supplement
    if len(result) < top_n:
        goal_crit_key = base_cond
        extras = _dataset_foods(goal_crit_key, {x["food"] for x in result},
                                 gender, age, bmi_val, n=top_n - len(result) + 8)
        for item in extras:
            if len(result) >= top_n: break
            # Apply goal filters
            if gcrit.get("min_protein", 0) and item.get("protein", 0) < gcrit["min_protein"]: continue
            if gcrit.get("max_calories") and item.get("calories", 0) > gcrit["max_calories"]: continue
            result.append(_build_item(item))

    return result[:top_n]


# Legacy helpers
HEALTHY_CATEGORIES = {
    "fruits":    ["apple","banana","mango","orange","grape","berry","watermelon","papaya","guava","pomegranate","kiwi","pear","peach","plum","cherry","apricot","fig","lychee","melon","pineapple","avocado","lemon"],
    "vegetables":["spinach","broccoli","carrot","tomato","cucumber","kale","lettuce","cabbage","cauliflower","beet","sweet potato","pumpkin","zucchini","celery","asparagus","pepper","collard","chard","leek","mushroom","garlic","onion","ginger"],
    "legumes":   ["lentil","chickpea","black bean","kidney bean","pea","soybean","tofu","mung bean","edamame","tempeh","bean","fenugreek"],
    "grains":    ["oat","quinoa","brown rice","barley","buckwheat","millet","wheat","bread","pasta","grain","rye","amaranth"],
    "proteins":  ["salmon","tuna","sardine","chicken","turkey","egg","almond","walnut","cashew","peanut","pistachio","chia","flaxseed","sunflower seed","pumpkin seed","hemp","mackerel","cod","tilapia"],
    "dairy":     ["yogurt","cheese","kefir","cottage","ricotta","mozzarella"],
}
CAT_EMOJI = {"fruits":"🍎","vegetables":"🥦","legumes":"🫘","grains":"🌾","proteins":"🥩","dairy":"🥛","other":"🥗"}
CONDITION_RULES = {k: {"prefer": v.get("prefer_cats",[])} for k, v in CONDITION_CRITERIA.items()}

def load_food_dataset_compat():
    return load_food_dataset()