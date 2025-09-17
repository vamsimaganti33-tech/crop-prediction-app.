 # --- Install required libraries (Colab-specific)
!pip install shap ipywidgets openpyxl --quiet
from google.colab import output
output.enable_custom_widget_manager()

# --- Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import shap
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings("ignore")

# --- Upload Excel
from google.colab import files
uploaded = files.upload()
file_name = next(iter(uploaded))
df = pd.read_excel(file_name).copy()

# --- Clean Temperature
df['Temperature'] = df['Temperature'].astype(str).str.replace('°C', '').str.strip()
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df = df.dropna(subset=['Temperature'])

# --- Features
numeric_features = [
    'n_High','n_Medium','n_Low',
    'p_High','p_Medium','p_Low',
    'k_High','k_Medium','k_Low',
    'OC_High','OC_Medium','OC_Low',
    'pH_Alkaline','pH_Acidic','pH_Neutral',
    'EC_NonSaline','EC_Saline',
    'S_Sufficient','S_Deficient',
    'Fe_Sufficient','Fe_Deficient',
    'Zn_Sufficient','Zn_Deficient',
    'Cu_Sufficient','Cu_Deficient',
    'B_Sufficient','B_Deficient',
    'Mn_Sufficient','Mn_Deficient',
    'Temperature'
]

categorical_features = ['District', 'Soil Type', 'Crop']

# --- Clean column types
for col in categorical_features:
    df[col] = df[col].astype(str)
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_features)
# --- Handle rare crops safely
min_samples = 2
rare_mask = df["Crop"].map(df["Crop"].value_counts() < min_samples)
df.loc[rare_mask, "Crop"] = "Other"

# --- Encode after replacement
crop_encoder_final = LabelEncoder()
y_crop = crop_encoder_final.fit_transform(df["Crop"])

# If you have category target also
y_cat  = LabelEncoder().fit_transform(df["Category"])

# Final target DataFrame
y_all = pd.DataFrame({"Crop": y_crop})



# --- Input Features
X_all = df[numeric_features + categorical_features]

# --- Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

# --- Final Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", MultiOutputClassifier(
        XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    ))
])

# --- Train/Test Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all["Crop"]
)

# --- Train & Evaluate
model_pipeline.fit(X_train, y_train)

# Predictions
y_test_pred = model_pipeline.predict(X_test)
y_train_pred = model_pipeline.predict(X_train)

# Convert predictions into DataFrames (align with y_test columns)
y_test_pred = pd.DataFrame(y_test_pred, columns=y_all.columns, index=y_test.index)
y_train_pred = pd.DataFrame(y_train_pred, columns=y_all.columns, index=y_train.index)

print("\n📊 Model Evaluation on Test Set (20% unseen data):")
for col in y_all.columns:
    print(f"\n🔹 Target: {col}")
    print(classification_report(y_test[col], y_test_pred[col]))

print("\n📊 Train vs Test Accuracy Comparison:")
for col in y_all.columns:
    acc_train = accuracy_score(y_train[col], y_train_pred[col])
    acc_test = accuracy_score(y_test[col], y_test_pred[col])
    print(f"  {col}: Train={acc_train:.2f} | Test={acc_test:.2f}")

# --- Retrain on 100% for Deployment
print("\n🚀 Retraining final model on 100% of data for deployment...")
model_pipeline.fit(X_all, y_all)
print("✅ Final model is now trained using the complete dataset.")

# =========================
# 🌍 INTERACTIVE UI
# =========================

# --- Widgets
district_dropdown = widgets.Dropdown(
    options=sorted(df['District'].unique()), description="District:", layout=widgets.Layout(width='50%')
)
block_dropdown = widgets.Dropdown(
    options=[], description="Block:", layout=widgets.Layout(width='50%')
)
soil_texture_dropdown = widgets.Dropdown(
    options=[], description="Soil Texture:", layout=widgets.Layout(width='50%')
)


button_predict = widgets.Button(description="🌱Suitable Crops", button_style="success")
output_predict = widgets.Output()

# --- Update logic
def update_block_options(change):
    selected_district = change['new']
    if 'Block' in df.columns:
        block_options = df[df['District'] == selected_district]['Block'].unique()
        block_dropdown.options = sorted(block_options)
        block_dropdown.value = block_dropdown.options[0] if len(block_options) > 0 else None
        update_soil_texture_options({'new': block_dropdown.value})
    else:
        block_dropdown.options = []
        block_dropdown.value = None
        update_soil_texture_options({'new': None})
def update_soil_texture_options(change):
    selected_district = district_dropdown.value
    selected_block = block_dropdown.value if block_dropdown.value else None

    if selected_block and 'Block' in df.columns:
        soil_options = df[
            (df['District'] == selected_district) &
            (df['Block'] == selected_block)
        ]['Soil Texture'].unique()
    else:
        soil_options = df[df['District'] == selected_district]['Soil Texture'].unique()

    soil_texture_dropdown.options = sorted(soil_options)
    soil_texture_dropdown.value = soil_texture_dropdown.options[0] if len(soil_options) > 0 else None



# --- Prediction Logic
def on_predict_clicked(b):
    with output_predict:
        clear_output()

        # Collect inputs
        user_input = {
            'District': district_dropdown.value,
            'Block': block_dropdown.value if 'Block' in df.columns else None,
            'Soil Texture': soil_texture_dropdown.value
        }

        # Get first matching row of numeric features (for demo)
        row = df[
            (df['District'] == user_input['District']) &
            ((df['Block'] == user_input['Block']) if user_input['Block'] else True) &
            (df['Soil Texture'] == user_input['Soil Texture'])
        ]

        if row.empty:
            print("⚠ No matching data found for this selection.")
            return

        # Take the first row for prediction
        X_input = row[numeric_features + categorical_features].iloc[[0]]

        # Predict probabilities
        preds_proba = model_pipeline.named_steps['classifier'].estimators_[0].predict_proba(
            model_pipeline.named_steps['preprocessor'].transform(X_input)
        )[0]

        # ✅ Get crop classes from the model
        crop_classes = model_pipeline.named_steps['classifier'].estimators_[0].classes_

        def decode_crop(cls):
            if cls == -1:
                return "Other"
            else:
                return crop_encoder_final.inverse_transform([cls])[0]

        crop_labels = [decode_crop(c) for c in crop_classes]

        # Build probability DataFrame (Top 10)
        crop_probs = pd.DataFrame({
            "Crop": crop_labels,
            "Suitability (%)": preds_proba * 100
        }).sort_values(by="Suitability (%)", ascending=False).head(10)

        # --- Bar Plot ---
        plt.figure(figsize=(8, 5))
        plt.barh(crop_probs["Crop"], crop_probs["Suitability (%)"], color="green")
        plt.gca().invert_yaxis()
        plt.xlabel("Suitability (%)")
        plt.title(f"Top 10 Predicted Suitable Crops\n{user_input['District']} ({user_input['Soil Texture']})")

        for i, v in enumerate(crop_probs["Suitability (%)"]):
            plt.text(v + 0.5, i, f"{v:.2f}%", va="center")
        plt.show()

        # --- Show BEST crop details automatically ---
        best_crop = crop_probs.iloc[0]["Crop"]
        print(f"🌟 Best Recommended Crop: {best_crop}\n")

        details = df[df["Crop"] == best_crop][
            ["Season", "Category", "Crop", "Soil Type", "Soil Texture", "Sowing Time", "Spacing"]
        ].drop_duplicates()

        display(details)








# --- Observers
district_dropdown.observe(update_block_options, names='value')
block_dropdown.observe(update_soil_texture_options, names='value')
button_predict.on_click(on_predict_clicked)

# --- Init
update_block_options({'new': district_dropdown.value})

# --- Display
print("📊 Model Accuracy:")
for col in y_all.columns:
    acc_train = accuracy_score(y_train[col], y_train_pred[col])
    acc_test = accuracy_score(y_test[col], y_test_pred[col])
    print(f"  {col}: Train={acc_train:.2f} | Test={acc_test:.2f}")

display(district_dropdown, block_dropdown, soil_texture_dropdown, button_predict, output_predict)
