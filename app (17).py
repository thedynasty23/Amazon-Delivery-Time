import os
import glob
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

st.set_page_config(page_title="Amazon Delivery Time Prediction", layout="wide")

# -------------------------
# Utility helpers
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Calculate haversine distance (km) between two points. Accepts floats or NaN."""
    try:
        lon1, lat1, lon2, lat2 = map(float, (lon1, lat1, lon2, lat2))
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km
    except Exception:
        return np.nan

def safe_choose_cols(df, candidates):
    for cand in candidates:
        if all([c in df.columns for c in cand]):
            return cand
    return None

def find_file_anywhere(name_substr):
    candidates = [
        f"./{name_substr}",
        os.path.join(os.getcwd(), name_substr),
        f"/workspace/{name_substr}",
        f"/home/app/{name_substr}",
        f"/mnt/data/{name_substr}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for p in glob.glob("**/*", recursive=True):
        if os.path.basename(p) == name_substr or name_substr in os.path.basename(p):
            return p
    return None

def safe_make_ohe(**kwargs):
    try:
        return OneHotEncoder(sparse=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse_output=False, **kwargs)

def get_ohe_feature_names(ohe, input_features):
    try:
        return list(ohe.get_feature_names_out(input_features))
    except Exception:
        try:
            return list(ohe.get_feature_names(input_features))
        except Exception:
            names = []
            if hasattr(ohe, "categories_"):
                for i, feat in enumerate(input_features):
                    cats = list(ohe.categories_[i])
                    for c in cats[1:]:
                        names.append(f"{feat}_{c}")
            return names

# -------------------------
# Robust artifact loaders
# -------------------------
@st.cache_resource
def load_model_if_exists(model_name="xgb_tuned_model_pca.pkl"):
    path = find_file_anywhere(model_name)
    if path is None:
        return None, None
    try:
        model = joblib.load(path)
        expected_n = getattr(model, "n_features_in_", None)
        return model, {"path": path, "expected_n": expected_n}
    except Exception as e:
        return None, {"path": path, "error": str(e)}

@st.cache_data
def load_training_csv(csv_name="amazon_distance.csv"):
    path = find_file_anywhere(csv_name)
    if path is None:
        return None, None
    try:
        df = pd.read_csv(path)
        return df, path
    except Exception:
        return None, path

# -------------------------
# Preprocessing / artifact fitting
# -------------------------
def fit_preprocessors(df, target_pca_n=None):
    artifacts = {}

    lat_candidates = [
        ("Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude"),
        ("Store_Lat", "Store_Long", "Drop_Lat", "Drop_Lng"),
        ("Store_lat", "Store_lng", "Drop_lat", "Drop_lng"),
        ("StoreLatitude","StoreLongitude","DropLatitude","DropLongitude"),
    ]
    loc_cols = safe_choose_cols(df, lat_candidates)

    core_map = {}
    core_map["Category"] = next((c for c in df.columns if c.lower().startswith("category")), None)
    core_map["Weather"] = next((c for c in df.columns if c.lower().startswith("weather")), None)
    core_map["Traffic"] = next((c for c in df.columns if c.lower().startswith("traffic")), None)
    core_map["Vehicle"] = next((c for c in df.columns if c.lower().startswith("vehicle")), None)
    core_map["Area"] = next((c for c in df.columns if c.lower().startswith("area")), None)
    core_map["Agent_Age"] = next((c for c in df.columns if "agent" in c.lower() and "age" in c.lower()), None)
    core_map["Agent_Rating"] = next((c for c in df.columns if "agent" in c.lower() and "rating" in c.lower()), None)
    core_map["Distance"] = next((c for c in df.columns if c.lower().startswith("distance")), None)
    core_map["Delivery_Time"] = next((c for c in df.columns if "delivery" in c.lower() and "time" in c.lower()), None)

    df2 = df.copy()

    if loc_cols is not None:
        s_lat, s_lon, d_lat, d_lon = loc_cols
        df2["Distance_calc"] = df2.apply(
            lambda r: haversine(r.get(d_lat, np.nan), r.get(d_lon, np.nan),
                                r.get(s_lat, np.nan), r.get(s_lon, np.nan)),
            axis=1
        )
        if core_map["Distance"] is None:
            df2["Distance"] = df2["Distance_calc"]

    possible_order_time_cols = [c for c in df2.columns if "order" in c.lower() and ("time" in c.lower() or "date" in c.lower())]
    if len(possible_order_time_cols) > 0:
        col = possible_order_time_cols[0]
        try:
            df2["Order_DT"] = pd.to_datetime(df2[col], errors="coerce")
            df2["Order_Year"] = df2["Order_DT"].dt.year
            df2["Order_Month"] = df2["Order_DT"].dt.month
            df2["Order_DayOfWeek"] = df2["Order_DT"].dt.dayofweek
            df2["Is_Weekend"] = df2["Order_DayOfWeek"].isin([5,6]).astype(int)
            df2["Order_Hour"] = df2["Order_DT"].dt.hour.fillna(0).astype(int)
        except Exception:
            pass

    if core_map["Traffic"] is not None:
        df2["Traffic_clean"] = df2[core_map["Traffic"]].astype(str).str.lower().str.strip()
    else:
        df2["Traffic_clean"] = "unknown"

    if core_map["Weather"] is not None:
        df2["Weather_clean"] = df2[core_map["Weather"]].astype(str).str.title().str.strip()
    else:
        df2["Weather_clean"] = "Unknown"

    if core_map["Category"] is not None:
        df2["Category_clean"] = df2[core_map["Category"]].astype(str).str.strip()
        df2["Category_FreqEnc"] = df2["Category_clean"].map(
            df2["Category_clean"].value_counts(normalize=True))
    else:
        df2["Category_clean"] = "Unknown"
        df2["Category_FreqEnc"] = 0.0

    if core_map["Delivery_Time"] is not None:
        df2["Category_TE"] = df2.groupby("Category_clean")[core_map["Delivery_Time"]].transform("mean")
    else:
        df2["Category_TE"] = df2["Category_clean"].map(
            df2["Category_clean"].value_counts(normalize=True))

    df2["Traffic_Weather"] = df2["Traffic_clean"].astype(str) + "_" + df2["Weather_clean"].astype(str)
    df2["Vehicle_clean"] = df2[core_map["Vehicle"]].astype(str) if core_map["Vehicle"] in df2.columns else "unknown"
    df2["Area_clean"] = df2[core_map["Area"]].astype(str) if core_map["Area"] in df2.columns else "unknown"

    if loc_cols is not None:
        _, _, d_lat, d_lon = loc_cols
        coords = df2[[d_lat, d_lon]].dropna()
        if len(coords) >= 5:
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(coords)
            def safe_cluster(r):
                if pd.isna(r[d_lat]) or pd.isna(r[d_lon]):
                    return -1
                return int(kmeans.predict([[r[d_lat], r[d_lon]]])[0])
            df2["Location_Cluster"] = df2.apply(safe_cluster, axis=1)
            artifacts["kmeans"] = kmeans
        else:
            df2["Location_Cluster"] = -1
    else:
        df2["Location_Cluster"] = -1

    ohe_features = ["Weather_clean", "Traffic_clean", "Vehicle_clean", "Area_clean", "Traffic_Weather"]
    ohe = safe_make_ohe(handle_unknown="ignore", drop="first")
    fit_df_ohe = df2[ohe_features].fillna("missing")
    ohe.fit(fit_df_ohe)
    artifacts["ohe"] = ohe
    artifacts["ohe_features"] = ohe_features

    numeric_cols = []
    for colname in ["Distance", "Order_Hour", "Agent_Age", "Agent_Rating", "Category_FreqEnc"]:
        if colname in df2.columns:
            numeric_cols.append(colname)
    if numeric_cols:
        df2[numeric_cols] = df2[numeric_cols].fillna(df2[numeric_cols].median())

    scaler = StandardScaler()
    if numeric_cols:
        scaler.fit(df2[numeric_cols])
    artifacts["scaler"] = scaler
    artifacts["numeric_cols"] = numeric_cols

    X_num = scaler.transform(df2[numeric_cols]) if numeric_cols else np.zeros((len(df2), 0))
    X_ohe_raw = ohe.transform(df2[ohe_features].fillna("missing"))
    X_ohe = X_ohe_raw.toarray() if hasattr(X_ohe_raw, "toarray") else X_ohe_raw

    extra_cols = []
    X_extra = np.empty((len(df2), 0))
    if "Category_TE" in df2.columns:
        X_extra = np.hstack([X_extra, df2[["Category_TE"]].fillna(0).values])
        extra_cols.append("Category_TE")
    if "Category_FreqEnc" in df2.columns:
        X_extra = np.hstack([X_extra, df2[["Category_FreqEnc"]].fillna(0).values])
        extra_cols.append("Category_FreqEnc")
    if "Location_Cluster" in df2.columns:
        X_extra = np.hstack([X_extra, df2[["Location_Cluster"]].fillna(-1).values])
        extra_cols.append("Location_Cluster")

    X_full = np.hstack([X_num, X_ohe, X_extra])

    artifacts["rf"] = None
    if core_map["Delivery_Time"] is not None:
        y = df2[core_map["Delivery_Time"]].values
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_full, y)
        artifacts["rf"] = rf

    if target_pca_n is not None and isinstance(target_pca_n, int) and target_pca_n > 0:
        desired_n = min(target_pca_n, X_full.shape[1])
    else:
        desired_n = min(17, X_full.shape[1])

    pca = PCA(n_components=desired_n, random_state=42)
    pca.fit(X_full)
    artifacts["pca"] = pca
    artifacts["pca_n"] = desired_n

    artifacts["feature_order"] = {
        "numeric": numeric_cols,
        "ohe": get_ohe_feature_names(ohe, ohe_features),
        "extra": extra_cols
    }
    artifacts["ohe_features"] = ohe_features
    artifacts["train_X_shape"] = X_full.shape

    return artifacts

# -------------------------
# Load model and train CSV
# -------------------------
model, model_meta = load_model_if_exists("xgb_tuned_model_pca.pkl")
train_df, train_csv_path = load_training_csv("amazon_distance.csv")

artifacts = None
if train_df is not None:
    target_pca_n = None
    if model_meta is not None and isinstance(model_meta, dict):
        target_pca_n = model_meta.get("expected_n")
    artifacts = fit_preprocessors(train_df, target_pca_n=target_pca_n)

# -------------------------
# UI
# -------------------------
st.title("🚚 Amazon Delivery Time Prediction")
st.markdown("Fill the form to predict delivery time")

default_weather_opts = ["Cloudy", "Sunny", "Rainy"]
default_traffic_opts = ["low", "medium", "high", "jam"]
default_vehicle_opts = ["motorcycle", "car", "van"]
default_area_opts = ["Metropolitian", "Urban", "Rural"]
default_category_opts = ["Apparel", "Electronics", "Home"]

def find_column_values(df, keyword, fallback):
    if df is None:
        return fallback
    col = next((c for c in df.columns if keyword in c.lower()), None)
    if col is not None:
        try:
            vals = sorted(df[col].dropna().unique())
            if len(vals) > 0:
                return vals
        except Exception:
            pass
    return fallback

weather_opts = find_column_values(train_df, "weather", default_weather_opts)
traffic_opts = find_column_values(train_df, "traffic", default_traffic_opts)
vehicle_opts = find_column_values(train_df, "vehicle", default_vehicle_opts)
area_opts = find_column_values(train_df, "area", default_area_opts)
category_opts = find_column_values(train_df, "category", default_category_opts)

left, right = st.columns([2,2])

with left:
    weather = st.selectbox("Weather", options=weather_opts)
    vehicle = st.selectbox("Vehicle", options=vehicle_opts)
    category = st.selectbox("Category", options=category_opts)
    agent_age = st.number_input("Agent_Age (range: 20 - 80)", min_value=20.0, max_value=80.0, value=30.0, step=1.0)
    distance_user = st.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1)

with right:
    traffic = st.selectbox("Traffic", options=traffic_opts)
    area = st.selectbox("Area", options=area_opts)
    agent_rating = st.number_input("Agent_Rating (range: 0.0 - 5.0)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    order_hour = st.number_input("Order_Hour (0 - 23)", min_value=0, max_value=23, value=12, step=1)

if train_df is None:
    st.error("Training CSV not found. Place 'amazon_distance.csv' in repo root or /mnt/data/ for preprocessing parity.")

# -------------------------
# Predict button logic
# -------------------------
if st.button("🚀 Predict Delivery Time"):
    raw = {
        "Weather_clean": weather,
        "Traffic_clean": traffic,
        "Vehicle_clean": vehicle,
        "Area_clean": area,
        "Category_clean": category,
        "Agent_Age": agent_age,
        "Agent_Rating": agent_rating,
        "Distance": distance_user * 1000,
        "Order_Hour": order_hour,
        "Traffic_Weather": f"{traffic}_{weather}"
    }
    X_raw = pd.DataFrame([raw])

    if artifacts is None:
        st.error("Missing preprocessing artifacts (training CSV not loaded). Cannot preprocess reliably.")
        st.stop()

    ohe = artifacts["ohe"]
    numeric_cols = artifacts["numeric_cols"]
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]
    rf = artifacts["rf"]
    ohe_features = artifacts["ohe_features"]

    for nc in numeric_cols:
        if nc not in X_raw.columns:
            X_raw[nc] = 0
    X_raw[numeric_cols] = X_raw[numeric_cols].fillna(0)
    X_num = scaler.transform(X_raw[numeric_cols]) if numeric_cols else np.zeros((1,0))

    X_ohe_raw = ohe.transform(X_raw[ohe_features].fillna("missing"))
    X_ohe = X_ohe_raw.toarray() if hasattr(X_ohe_raw, "toarray") else X_ohe_raw

    extras = []
    cat_col = next((c for c in train_df.columns if c.lower().startswith("category")), None) if train_df is not None else None
    if "Category_TE" in artifacts["feature_order"]["extra"]:
        try:
            target_col = next((c for c in train_df.columns if "delivery" in c.lower() and "time" in c.lower()), None)
            if cat_col is not None and target_col is not None:
                te_map = train_df.groupby(cat_col)[target_col].mean()
                te_val = te_map.get(category, 0) if hasattr(te_map, "get") else 0
            else:
                te_val = 0
        except Exception:
            te_val = 0
        extras.append([te_val])
    if "Category_FreqEnc" in artifacts["feature_order"]["extra"]:
        try:
            freq_val = train_df[cat_col].value_counts(normalize=True).get(category, 0) if cat_col in train_df.columns else 0
        except Exception:
            freq_val = 0
        extras.append([freq_val])
    if "Location_Cluster" in artifacts["feature_order"]["extra"]:
        extras.append([-1])

    if len(extras) > 0:
        cols = [np.asarray(e).reshape(1, -1) for e in extras]
        X_extra = np.hstack(cols)
    else:
        X_extra = np.empty((1,0))

    X_full = np.hstack([X_num, X_ohe, X_extra])
    X_pca = pca.transform(X_full)

    chosen_prediction = None

    if model is not None:
        expected_n = getattr(model, "n_features_in_", None)
        if expected_n is not None and expected_n != X_pca.shape[1]:
            if X_pca.shape[1] > expected_n:
                X_pca_used = X_pca[:, :expected_n]
            else:
                pad = np.zeros((X_pca.shape[0], expected_n - X_pca.shape[1]))
                X_pca_used = np.hstack([X_pca, pad])
        else:
            X_pca_used = X_pca

        try:
            raw_pred = model.predict(X_pca_used)
            chosen_prediction = float(np.array(raw_pred).ravel()[0])
        except Exception:
            chosen_prediction = None

    if chosen_prediction is None and rf is not None:
        try:
            rf_pred = rf.predict(X_full)
            chosen_prediction = float(np.array(rf_pred).ravel()[0])
        except Exception:
            pass

    if chosen_prediction is None and train_df is not None:
        target_col = next((c for c in train_df.columns if "delivery" in c.lower() and "time" in c.lower()), None)
        if target_col is not None:
            chosen_prediction = float(train_df[target_col].median())

    if chosen_prediction is None:
        st.error("Failed to produce a prediction.")
    else:
        st.success(f"Predicted Delivery Time: {chosen_prediction:.2f} minutes")
