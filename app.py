import streamlit as st
import plotly.graph_objs as go
import numpy as np
from collections import deque
import time

"""
Anomaly Detection in Data Streams using Sliding Window K-Nearest Neighbors (SWKNN)

This script implements real-time anomaly detection on a continuous data stream using the
SWKNN algorithm. It's designed to identify unusual patterns or outliers in streaming data,
which could represent various metrics such as financial transactions or system performance.

Algorithm Explanation:
The Sliding Window K-Nearest Neighbors (SWKNN) algorithm is used for anomaly detection.
This algorithm maintains a sliding window of the most recent data points. For each new 
point, it calculates the average distance to its K nearest neighbors within the window.
If this average distance exceeds a predefined threshold, the point is flagged as an anomaly.

Effectiveness:
1. Adaptability: By using a sliding window, the algorithm can adapt to concept drift and
   seasonal variations in the data.
2. Simplicity: The algorithm is intuitive and easy to implement, making it suitable for
   real-time applications.
3. Efficiency: With proper optimization, SWKNN can process data streams quickly.
4. Sensitivity: The algorithm can detect both global and local anomalies by comparing
   each point to its nearest neighbors.

However, it's worth noting that the effectiveness can be sensitive to the choice of
parameters (window size, K, and threshold), which may need tuning for specific use cases.
"""

# Constants
WINDOW_SIZE = 500  # Number of recent points to consider
K = 5  # Number of nearest neighbors to compare
THRESHOLD = 2.5  # Threshold for anomaly detection
UPDATE_INTERVAL = 0.05  # Time between plot updates (seconds)
BATCH_SIZE = 10  # Number of points to process before updating the plot

class SWKNN:
    """
    Sliding Window K-Nearest Neighbors algorithm for Real-time Stream Anomaly Detection.
    """

    def __init__(self, window_size, k, threshold):
        """
        Initialize the SWKNN anomaly detector.

        :param window_size: Size of the sliding window
        :param k: Number of nearest neighbors to consider
        :param threshold: Anomaly threshold
        """
        self.window = deque(maxlen=window_size)
        self.k = k
        self.threshold = threshold

    def add_point(self, point):
        """
        Add a new point to the sliding window.

        :param point: New data point to be added
        """
        self.window.append(point)

    def detect_anomaly(self, point):
        """
        Detect if a point is an anomaly based on its distance to K-nearest neighbors.

        :param point: Point to check for anomaly
        :return: True if the point is an anomaly, False otherwise
        """
        if len(self.window) < self.k:
            return False
        distances = sorted(abs(point - p) for p in self.window)[:self.k]
        return sum(distances) / self.k > self.threshold

def generate_data_stream():
    """
    Generate a continuous stream of data points.

    The stream includes:
    - A regular pattern (sine waves)
    - Seasonal variations (slower sine wave)
    - Random noise
    - Occasional anomalies

    :yield: Tuple of (timestamp, value)
    """
    t = 0
    while True:
        # Generate base value with regular and seasonal patterns
        value = 10 + np.sin(t * 0.1) * 5 + np.sin(t * 0.01) * 3
        
        # Add random noise
        value += np.random.normal(0, 0.5)
        
        # Introduce occasional anomalies (5% chance)
        if np.random.random() < 0.05:
            value += np.random.choice([-1, 1]) * np.random.uniform(5, 10)
        
        yield t, value
        t += 1

def safe_division(numerator, denominator):
    """
    Perform safe division to avoid divide-by-zero errors.

    :param numerator: Dividend
    :param denominator: Divisor
    :return: Result of division, or 0 if denominator is 0
    """
    return numerator / denominator if denominator != 0 else 0

def main():
    """
    Main function to run the anomaly detection and visualization.
    """
    st.set_page_config(page_title="Real-time Anomaly Detection 📈", layout="wide")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = {'x': [], 'y': [], 'anomaly': []}
        st.session_state.total_points = 0
        st.session_state.total_anomalies = 0
        st.session_state.is_running = True
        st.session_state.stop_time = None

    # Create layout
    col1, col2 = st.columns([3, 1])

    with col1:
        plot_placeholder = st.empty()

    with col2:
        stats_placeholder = st.empty()
        stop_button = st.button("Stop")

    # Initialize plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Data Stream'))
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
    fig.update_layout(
        title='Real-time Data Stream with Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Value',
        height=600,
        xaxis=dict(rangeslider=dict(visible=True), type="linear"),
        yaxis=dict(fixedrange=False)
    )

    # Initialize detector and data stream
    detector = SWKNN(WINDOW_SIZE, K, THRESHOLD)
    data_stream = generate_data_stream()

    def update_plot_and_stats():
        """
        Update the plot and statistics display.
        """
        # Update main data series
        fig.data[0].x = st.session_state.data['x']
        fig.data[0].y = st.session_state.data['y']
        
        # Update anomaly points
        anomaly_x = [x for x, is_anomaly in zip(st.session_state.data['x'], st.session_state.data['anomaly']) if is_anomaly]
        anomaly_y = [y for y, is_anomaly in zip(st.session_state.data['y'], st.session_state.data['anomaly']) if is_anomaly]
        fig.data[1].x = anomaly_x
        fig.data[1].y = anomaly_y

        # Add stop time indicator if applicable
        if st.session_state.stop_time is not None:
            fig.add_vline(x=st.session_state.stop_time, line_dash="dash", line_color="red", annotation_text="Stopped")

        # Display updated plot
        plot_placeholder.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
        # Calculate and display statistics
        anomaly_rate = safe_division(st.session_state.total_anomalies, st.session_state.total_points)
        stats_placeholder.markdown(f"""
        **Statistics:**
        - Total points processed: {st.session_state.total_points}
        - Total anomalies detected: {st.session_state.total_anomalies}
        - Anomaly rate: {anomaly_rate:.2%}
        """)

    try:
        while st.session_state.is_running:
            batch_x, batch_y, batch_anomaly = [], [], []

            # Process a batch of data points
            for _ in range(BATCH_SIZE):
                t, value = next(data_stream)
                is_anomaly = detector.detect_anomaly(value)
                detector.add_point(value)

                st.session_state.total_points += 1
                if is_anomaly:
                    st.session_state.total_anomalies += 1

                batch_x.append(t)
                batch_y.append(value)
                batch_anomaly.append(is_anomaly)

            # Update data in session state
            st.session_state.data['x'].extend(batch_x)
            st.session_state.data['y'].extend(batch_y)
            st.session_state.data['anomaly'].extend(batch_anomaly)

            # Update x-axis range to show the last WINDOW_SIZE points
            if len(st.session_state.data['x']) > WINDOW_SIZE:
                fig.update_xaxes(range=[st.session_state.data['x'][-WINDOW_SIZE], st.session_state.data['x'][-1]])

            # Update plot and stats
            update_plot_and_stats()

            # Check if stop button was pressed
            if stop_button:
                st.session_state.is_running = False
                st.session_state.stop_time = st.session_state.data['x'][-1]
                break

            # Wait before processing the next batch
            time.sleep(UPDATE_INTERVAL * BATCH_SIZE)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Ensure final update of plot and stats
        update_plot_and_stats()

if __name__ == "__main__":
    main()