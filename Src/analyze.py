"""
Fast Food Nutrition Analysis
Run from the repository root: python Src/analyze.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "fastfood_calories.csv"
OUT = ROOT / "Outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
numeric_cols = [
    "calories", "cal_fat", "total_fat", "sat_fat", "trans_fat",
    "cholesterol", "sodium", "total_carb", "fiber", "sugar",
    "protein", "vit_a", "vit_c", "calcium"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Restaurant-level descriptive statistics.
restaurant_summary = df.groupby("restaurant")[numeric_cols].mean().round(2)
restaurant_summary.insert(0, "menu_items", df.groupby("restaurant").size())
restaurant_summary.to_csv(OUT / "restaurant_summary.csv")

# Highest-calorie items.
df.nlargest(10, "calories")[["restaurant", "item", "calories", "total_fat", "sodium"]].to_csv(
    OUT / "top_10_calorie_items.csv", index=False
)

# Salad versus other menu items.
salad_summary = df.groupby("salad")[["calories", "total_fat", "total_carb", "protein"]].mean().round(2)
salad_summary.to_csv(OUT / "salad_vs_other.csv")

# Random Forest feature importance for predicting calories from nutrition fields.
features = ["total_fat", "sat_fat", "trans_fat", "cholesterol", "sodium",
            "total_carb", "fiber", "sugar", "protein", "vit_a", "vit_c", "calcium"]
model_data = df[features + ["calories"]].dropna(subset=["calories"])
X, y = model_data[features], model_data["calories"]
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
)
model.fit(X, y)
rf = model.named_steps["randomforestregressor"]
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
importance.rename("importance").to_csv(OUT / "random_forest_feature_importance.csv")

# Chart 1: mean calories by restaurant.
calories = restaurant_summary["calories"].sort_values(ascending=False)
ax = calories.plot(kind="bar", figsize=(10, 5), title="Average Calories by Restaurant")
ax.set_ylabel("Average calories per menu item")
ax.set_xlabel("Restaurant")
plt.tight_layout()
plt.savefig(OUT / "average_calories_by_restaurant.png", dpi=160)
plt.close()

# Chart 2: Random Forest feature importance.
ax = importance.sort_values().plot(kind="barh", figsize=(9, 6), title="Random Forest Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Nutrition feature")
plt.tight_layout()
plt.savefig(OUT / "random_forest_feature_importance.png", dpi=160)
plt.close()

print(f"Analysis complete. Results saved in: {OUT}")
print(f"Rows: {len(df)} | Restaurants: {df['restaurant'].nunique()}")
