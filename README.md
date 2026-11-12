# Fast Food Nutrition Analysis

**Data Science Project | MCA Semester 3**

- **Student:** Ananya Dey
- **Student code:** BWU/MCA/25/050
- **Section:** A
- **Subject:** Data Science
- **Course code:** MCA37505

## Project overview

This project explores nutritional information for fast-food menu items across eight restaurant chains. It includes descriptive statistics, comparisons of calories/protein/fat/sugar, salad-versus-other comparisons, and Random Forest feature-importance analysis.

## Repository structure

```text
Fast_Food_Nutrition_Analysis/
├── Data/
│   ├── fastfood_calories.csv
│   ├── Fast-food-dataset-source.pdf
│   └── Fast_Food_Nutrition_Analysis_Presentation.pptx
├── Outputs/
│   └── (analysis tables and charts are generated here)
├── Src/
│   └── analyze.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset and attribution

The CSV contains 515 menu-item records and 18 columns. The uploaded PDF source was used to reconstruct the CSV while retaining its original column names and values; line breaks within item names were normalized to spaces.

Original dataset:
- TidyTuesday, 2018-09-04 — `fastfood_calories.csv`
- https://github.com/rfordatascience/tidytuesday/blob/master/data/2018/2018-09-04/fastfood_calories.csv
- Direct CSV: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-09-04/fastfood_calories.csv

The source PDF supplied with this project is retained in `Data/` for reference. Please cite the TidyTuesday dataset when reusing the data.

## Run the analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run from the repository root:

```bash
python Src/analyze.py
```

The script writes summary CSV files and PNG charts to `Outputs/`.

## Important methodology note

The presentation's health-risk scores, overall health ranking, and Random Forest feature weights are derived analyses, not original columns in the dataset. The script demonstrates reproducible descriptive analysis and a Random Forest model; its generated results should not be assumed to reproduce every presentation figure unless the exact preprocessing, feature selection, scaling, and scoring formula used to create the presentation are also specified.

## License / reuse

The dataset remains subject to the terms and attribution requirements of its original publisher. This repository is an educational project package; check the upstream repository before redistributing the dataset separately.
