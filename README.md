
# Real-time Anomaly Detection in Data Streams using Sliding Window K-Nearest Neighbors (SWKNN)

This project demonstrates **real-time anomaly detection** using a **Sliding Window K-Nearest Neighbors (SWKNN)** algorithm to monitor streaming data. The program is designed to identify unusual patterns or outliers in the data stream, and it uses a sliding window to adapt to recent data while detecting anomalies efficiently.

---

## Table of Contents

- [Algorithm Explanation](#algorithm-explanation)
- [Project Structure](#project-structure)
- [How the Algorithm Works](#how-the-algorithm-works)
- [Key Features](#key-features)
- [Fine-Tuning Parameters](#fine-tuning-parameters)
- [How to Run](#how-to-run)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Limitations](#limitations)

---

## Algorithm Explanation

The **Sliding Window K-Nearest Neighbors (SWKNN)** algorithm is used for real-time anomaly detection in continuous data streams. It maintains a sliding window of the most recent data points and, for each new point, calculates the average distance to its **K nearest neighbors**. If the average distance exceeds a predefined **threshold**, the point is flagged as an anomaly.

**Key Characteristics:**

1. **Sliding Window:** Only the most recent points are considered, making the algorithm adaptable to changes in the data (concept drift).
2. **K-Nearest Neighbors (KNN):** For every incoming data point, the algorithm looks for the K nearest points in the sliding window to calculate an anomaly score.
3. **Threshold-based Detection:** If the average distance to neighbors exceeds the threshold, the point is considered an anomaly.

---

## Project Structure

- **Streamlit Frontend:** Real-time visualization of the data stream and anomalies.
- **SWKNN Class:** Implements the sliding window KNN algorithm for anomaly detection.
- **Data Stream Generator:** Simulates a continuous data stream with periodic patterns, noise, and occasional anomalies.

---

## How the Algorithm Works

### 1. Sliding Window

The **sliding window** maintains a limited number of recent data points (defined by `WINDOW_SIZE`). New points are appended to the window, and old points are discarded as the window moves forward.

### 2. K-Nearest Neighbors (K)

For each new data point, the algorithm calculates the distance to its **K nearest neighbors** within the sliding window. The **K nearest neighbors** are the points with the smallest absolute distance to the new point.

### 3. Anomaly Detection

The algorithm computes the average distance between the new point and its K nearest neighbors. If this average distance exceeds a user-defined **threshold**, the point is flagged as an anomaly. Anomalies are typically data points that deviate significantly from the recent pattern.

**Strength:**

- The approach is capable of detecting both **global** and **local anomalies**. A point could be flagged as an anomaly either because it is far from all other points or because it's inconsistent with recent trends.

**Limitation:**

- The algorithm's sensitivity heavily depends on the parameter selection (`WINDOW_SIZE`, `K`, and `THRESHOLD`).

---

## Key Features

- **Real-Time Processing:** The data is processed in batches, and both the data stream and anomalies are updated in real-time.
- **Interactive Plot:** The program uses Plotly to create an interactive plot that displays the data stream, anomalies, and statistics.
- **Dynamic Adjustment:** The sliding window ensures that the algorithm adjusts to recent changes in the data pattern, making it suitable for environments where data distribution changes over time.

---

## Fine-Tuning Parameters

The performance of SWKNN is highly sensitive to three parameters: **K**, **WINDOW_SIZE**, and **THRESHOLD**. Below is a guide on how to fine-tune these parameters:

### 1. **WINDOW_SIZE**

- **What it controls:** The number of recent data points to retain in memory.
- **How it impacts performance:**
  - **Too small:** The algorithm will only consider a small portion of the data, potentially missing trends and yielding noisy results.
  - **Too large:** The algorithm may become sluggish to adapt to changes in the data.
- **Recommended starting point:** Start with 200-500 points and adjust based on the size of your data stream and how quickly patterns change.

### 2. **K (Nearest Neighbors)**

- **What it controls:** The number of nearest neighbors used for anomaly detection.
- **How it impacts performance:**
  - **Too small:** The algorithm may detect too many false positives as it becomes overly sensitive to outliers.
  - **Too large:** The algorithm may miss anomalies by averaging over too many points.
- **Recommended starting point:** Start with `K = 5` and gradually increase based on your dataset. Typically, K values between 3 and 10 work well for streaming data.

### 3. **THRESHOLD**

- **What it controls:** The threshold for determining whether a point is an anomaly based on the average distance to its K nearest neighbors.
- **How it impacts performance:**
  - **Too low:** The algorithm will flag too many points as anomalies, leading to false positives.
  - **Too high:** The algorithm may miss true anomalies, leading to false negatives.
- **Recommended starting point:** Begin with a threshold of `2.5`. Experiment with values between 2 and 5 depending on the noise level in your data.

### Fine-tuning Tips:

- **Start simple:** Begin with small data streams and set moderate values for `WINDOW_SIZE`, `K`, and `THRESHOLD`.
- **Monitor false positives:** If too many normal points are flagged as anomalies, increase `K` and/or the `THRESHOLD`.
- **Monitor false negatives:** If the algorithm is missing anomalies, try reducing `K` or lowering the `THRESHOLD`.

---

## How to Run

### Virtual Environment Setup

Before running the program, it's recommended to set up a virtual environment to manage the dependencies. You can use either `conda` or Python's built-in `venv`:

#### Using `conda`

1. Create a new environment:
   ```bash
   conda create --name myenv python=3.9
   ```
2. Activate the environment:
   ```bash
   conda activate myenv
   ```

#### Using `venv` (Python's built-in virtual environment)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   - **On Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     venv\Scripts\activate
     ```

### Install Dependencies

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

---

### Running the Application

1. **Start the Streamlit App:**
   To run the Streamlit frontend and visualize real-time anomaly detection:

   ```bash
   streamlit run app.py
   ```
2. **Visualization:**
   The Streamlit app will display a real-time graph of the data stream and mark any detected anomalies in red.

---

## Limitations

1. **Parameter Sensitivity:** As noted, the SWKNN algorithm is sensitive to the choice of parameters, and fine-tuning is often necessary for optimal performance.
2. **Performance on Highly Noisy Data:** With high noise, the algorithm may generate false positives, and parameter adjustments may be needed.
3. **Processing Speed:** Although efficient for most real-time applications, very high-frequency data streams may slow down the anomaly detection process.
