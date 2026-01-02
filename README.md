# 🏎️ F1 Race Time Prediction Using Machine Learning & Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-OLS%20%7C%20DNN%20%7C%20LSTM-orange)
![TimeSeries](https://img.shields.io/badge/Time%20Series-LSTM-purple)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

**Author:** Anurag Surve
**Domain:** Sports Analytics · Machine Learning · Deep Learning

---

## 🔍 Project Snapshot (TL;DR)

This project predicts **total Formula 1 race completion time** using **lap-level telemetry, pit stop information, qualifying performance, and tyre dynamics**.

The work combines:

* **Statistical modeling (OLS) for interpretability**
* **Deep Neural Networks (DNN) for nonlinear relationships**
* **LSTM + DNN hybrid architecture** to capture **temporal race dynamics**

The final model achieves **strong predictive performance (R² ≈ 0.92)** by jointly modeling **race context + historical lap behavior** 

---

## 🎯 Problem Statement

Given partial race information (current lap status, lap times, pit stops, tyre age, and qualifying data), predict the **total race time** for an F1 driver.

This is a realistic motorsports analytics problem with:

* Noisy lap times
* Strategy-dependent pit stops
* Strong temporal dependencies

---

## 📊 Dataset Overview

**Target Variable**

* `total_time`: Total race completion time (seconds)

**Key Features**

* `lap`: Current lap number
* `lap_position`: Driver’s current position
* `lap_time`: Time for the most recent lap
* `quali_position`: Final qualifying position
* `q1`, `q2`, `q3`: Qualifying session times
* `no_of_stops`: Number of pit stops so far
* `pit_stop_duration`: Latest pit stop duration
* `total_laps`: Total race laps
* `tyre_age`: Engineered feature (laps since last pit stop)

---

## 🧹 Data Cleaning & Feature Engineering

* **Lap Time Outliers**

  * Extreme lap times replaced with **driver-specific race median**
* **Total Race Time Filtering**

  * Removed unrealistic values (< 4000s or > 8000s)
* **Tyre Age (Engineered Feature)**

  * Calculated from last pit stop timing
  * Critical for modeling degradation and strategy effects

---

## 📐 Feature Selection (OLS)

Ordinary Least Squares regression was used for **interpretability and feature validation**.

Key findings:

* `lap_time`, `total_laps`, `no_of_stops`, and `tyre_age` strongly increase race time
* `lap` and `lap_position` show negative coefficients (race progression effect)
* Pit stops add ~140 seconds per stop on average

OLS provided **domain validation** before deep learning modeling 

---

## 🤖 Models Implemented

### 1️⃣ Deep Neural Network (DNN)

**Input:** Static race features
**Loss:** MSE

**Performance**

* Test R²: **0.65**
* Test MAE: **0.45**
* Test MSE: **0.49**

Captures nonlinearities but ignores temporal lap history.

---

### 2️⃣ LSTM + DNN (Best Model)

**Architecture**

* LSTM processes **previous 5 lap times**
* Output concatenated with static race features
* Final prediction via feedforward network

**Why this works**

* LSTM captures **pace evolution**
* DNN captures **race context & strategy**

**Performance**

* Test R²: **0.919**
* Test MAE: **0.238**
* Test MSE: **0.112**

This model significantly outperforms the standalone DNN by modeling **temporal dependencies** 

---

## 📈 Key Insights

* Tyre age is one of the strongest predictors of race time
* Pit strategy dominates long-run race outcomes
* Temporal modeling is essential—treating laps independently underperforms
* Hybrid LSTM architectures are well-suited for motorsports analytics

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repository
2. Open notebooks in order:

   * Data preparation
   * EDA & visualization
   * Model training
3. Run `model_train.ipynb` to reproduce results

---

## ⚠️ Limitations & Future Work

* Incorporate weather and safety-car data
* Use rolling inference across laps
* Explore Transformer-based temporal models
* Extend to race strategy optimization

---

## 💼 Why This Project Matters

This project demonstrates:

* **End-to-end ML pipeline design**
* **Time-series modeling with real constraints**
* **Strong performance gains through model architecture choices**
* **Sports analytics with real strategic implications**

---

## 📬 Contact

**Anurag Surve**
LinkedIn: [www.linkedin.com/in/anurag-surve-b37291265](http://www.linkedin.com/in/anurag-surve-b37291265)
