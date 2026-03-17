# 🏭 Tata Steel Machine Failure Prediction (End-to-End ML Project)

## 📌 Project Overview
This project focuses on predicting machine failures in a steel manufacturing environment using an **end-to-end machine learning pipeline**.

It combines:
- **Exploratory Data Analysis (EDA)**
- **Feature Engineering**
- **Imbalance-aware Model Building**
- **Threshold Optimization**
- **Business-driven Evaluation**

The goal is to build an **early-warning system** that enables proactive maintenance and reduces unplanned downtime.

---

## 🎯 Business Objective
- Predict machine failures before they occur  
- Identify key operational risk factors  
- Reduce downtime and maintenance costs  
- Enable proactive maintenance planning  

---

## ⚠️ Problem Statement
In steel manufacturing, unexpected machine failures lead to:
- Production loss  
- Increased maintenance costs  
- Operational inefficiencies  

The challenge is to detect failures **early and accurately**, especially since failures are rare events.

---

## 📊 Dataset Description
The dataset contains **industrial machine sensor and operational data**:

- Product Type (L, M, H)  
- Air & Process Temperature  
- Rotational Speed  
- Torque  
- Tool Wear  
- Failure Indicators (TWF, HDF, PWF, OSF, RNF)  
- Target: **Machine Failure (0/1)**  

📌 Dataset Size:
- Train: 136,429 rows  
- Test: 90,954 rows  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost, LightGBM  
- Joblib (Model Saving)  

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings:

- **Severe class imbalance** (~1.57% failures)  
- Failures strongly linked to:
  - High torque  
  - High tool wear  
  - Low rotational speed  
- Product Type **L** shows higher failure rate  
- Failure modes (HDF, OSF, etc.) are highly correlated with target  
  ⚠️ Potential **data leakage risk**

---

## 🧠 Feature Engineering

Created domain-driven features:

- **Temperature Difference** = Process Temp − Air Temp  
- **Power (kW)** from speed & torque  
- **Torque per Wear** (load efficiency)  
- **Product Type Prefix**  

These features improved model interpretability and predictive power.

---

## ⚖️ Modeling Strategy

Due to class imbalance:

- Used **Stratified Train-Test Split**
- Applied **Class Weighting**
- Focused on:
  - PR-AUC  
  - Recall  
  - F1 Score  
- Avoided leakage by excluding failure mode indicators  

---

## 🤖 Models Used

- Logistic Regression  
- Random Forest  
- Extra Trees  
- XGBoost ✅ (Best Model)  
- LightGBM  

---

## 📈 Model Performance

### 🏆 Best Model: XGBoost

- **PR-AUC:** ~0.445  
- **Recall:** ~50.7%  
- **Precision:** ~52.0%  
- **F1 Score:** ~0.51  
- **ROC-AUC:** ~0.91  

📌 Model optimized using **threshold tuning (0.92)** instead of default 0.5.

---

## 🎯 Key Insights

- Failures cluster in **low-speed + high-torque conditions**  
- High tool wear significantly increases failure risk  
- Temperature differences influence machine stability  
- Failure prediction requires **multi-feature interaction**, not single thresholds  

---

## 📉 Risk Segmentation

- High Torque + High Tool Wear → **Highest Risk Zone (~11%)**  
- Low Torque + Low Wear → **Lowest Risk (~0.09%)**  

This enables:
- Maintenance prioritization  
- Resource allocation  
- Real-time monitoring  

---

## 📊 Evaluation Approach

- Cross-validation using **PR-AUC**
- Threshold optimization for **business trade-offs**
- Confusion matrix analysis:
  - Minimize **False Negatives (missed failures)**  
- Precision-Recall curve for imbalance handling  

---

## 💰 Business Impact

- Reduces unplanned downtime  
- Enables proactive maintenance  
- Improves production reliability  
- Optimizes maintenance cost  

📌 Example:
- Missed failure cost assumed >> false alert cost  
- Model tuned accordingly  

---

## 📦 Model Deployment Artifacts

- Saved model: `artifacts/best_model.joblib`  
- Reports generated:
  - Model leaderboard  
  - Threshold sensitivity  
  - Feature importance  
  - Metrics JSON  

---

## 🚀 Deployment Strategy

- Serve model via **Streamlit / FastAPI**
- Input: real-time machine parameters  
- Output:
  - Failure probability  
  - Risk alerts  

---

## 📂 Project Structure
