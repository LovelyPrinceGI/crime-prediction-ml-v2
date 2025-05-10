# **🔍 Crime Prediction using Machine Learning**

A machine learning project aimed at analyzing and predicting crime patterns using various features such as offender count, victim count, crime type, area type and ethnicity of offender.

## **📁 Project Structure**

 ```bash
CP_Project_V2/
├── app/
│   └── main.py
├── datasets/
│   ├── hate_crime.csv
│   └── updated_hate_crime.csv
├── dependencies/
│   └── requirements.txt
├── source_code/
│   └── hate_crime_prediction_V3.ipynb
├── .gitignore
└── README.md
```

## **🎯 Objectives**

- Analyze correlations between crime-related features

- Use feature selection techniques to find the most impactful features

- Apply machine learning models such as Decision Tree and Random Forest

- Visualize feature importance and data relationships

## **📌 Selected Features**

Based on correlation analysis and feature importance from models:

| Feature Name             | Description                                 |
|--------------------------|---------------------------------------------|
| `total_offender_count`   | Total number of offenders in a crime        |
| `total_individual_victims` | Unique number of individuals who were victims, regardless of how many times they were victimized |
| `crime_type_violent`     | Whether the crime was violent or not       |
| `victim_count`           |  Total number of victimization incidents, including repeated crimes against the same person |
| `area_type_Urban`        | Indicates if the crime occurred in urban areas |
| `offender_ethnicity_Not Hispanic or Latino` | The ethicity of the offender, indicating the race of the offender who is excluded from a person of Cuban, Mexican  Puerto Rican, South or Central American, or other Spanish culture or origin that wasn't being in common races in USA.|

These were chosen for their consistently high scores across multiple feature importance analyses (Random Forest).


## **📌 Selected Target Variables**

1. `gender_bias`

2. `racial_bias`

3. `religion_bias`

4. `disability_bias`

## **🧠 Models Used**

- `RandomForestClassifier`

- ✅ `BalancedRandomForestClassifier` + `GridSearchCV`

Performance comparison and feature importances are visualized through bar charts and heatmaps.

## **📊 Visual Results**

✅ Heatmaps and bar charts are included in the notebook to show:

- Correlation between features

- Model-based feature importance

> You can find these visualizations in `source_code/hate_crime_prediction_V3.ipynb`.

## **🚀 Getting Started**

1. Clone this repository:

   ```bash
   git clone https://github.com/LovelyPrinceGI/crime-prediction-ml-v2

2. Create and activate the virtual environment:

    ```bash
    python -m venv dsai_venv
    dsai_venv\Scripts\activate

3. Install dependencies:

    ```bash
    pip install -r dependencies/requirements.txt

4. Run the notebook


## **🛠 Dependencies**

### Key packages:

- `pandas`

- `scikit-learn`

- `matplotlib`

- `seaborn`

- `streamlit (for optional app deployment)`

`Note`: All dependencies are listed in dependencies/requirements.txt.


### **Author**

- Patsakorn Tangkachaiyanunt, IM student in AIT

- Shreeyukta Pradhanang, CS student in AIT