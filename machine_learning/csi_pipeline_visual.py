import sys
import csv
import json
import argparse
import pandas as pd
import numpy as np
import re

import serial
from os import path
from io import StringIO

from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pq

import threading
import time
import queue
from tensorflow.keras.models import load_model

# Define heart rate class names
CLASS_NAMES = ["70-75 bpm", "75-80 bpm", "80-85 bpm", "85-90 bpm", "90-95 bpm", "95-100 bpm"]

# Reduce displayed waveforms to avoid display freezes
CSI_VAID_SUBCARRIER_INTERVAL = 3

# Remove invalid subcarriers
# secondary channel : below, HT, 40 MHz, non STBC, v, HT-LFT: 0~63, -64~-1, 384
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_color = []
color_step = 255 // (28 // CSI_VAID_SUBCARRIER_INTERVAL + 1)

# LLTF: 52
csi_vaid_subcarrier_index += [i for i in range(6, 32, CSI_VAID_SUBCARRIER_INTERVAL)]     # 26  red
csi_vaid_subcarrier_color += [(i * color_step, 0, 0) for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
csi_vaid_subcarrier_index += [i for i in range(33, 59, CSI_VAID_SUBCARRIER_INTERVAL)]    # 26  green
csi_vaid_subcarrier_color += [(0, i * color_step, 0) for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
CSI_DATA_LLFT_COLUMNS = len(csi_vaid_subcarrier_index)

# HT-LFT: 56 + 56
csi_vaid_subcarrier_index += [i for i in range(66, 94, CSI_VAID_SUBCARRIER_INTERVAL)]    # 28  blue
csi_vaid_subcarrier_color += [(0, 0, i * color_step) for i in range(1,  28 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
csi_vaid_subcarrier_index += [i for i in range(95, 123, CSI_VAID_SUBCARRIER_INTERVAL)]   # 28  White
csi_vaid_subcarrier_color += [(i * color_step, i * color_step, i * color_step) for i in range(1,  28 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]

CSI_DATA_INDEX = 200  # buffer size
CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]
csi_data_array = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)

class CSIPredictionEngine:
    def __init__(self, model_path='csi_lstm_model.h5', buffer_size=10):
        """
        Initialize the CSI prediction engine
        
        Args:
            model_path: Path to the saved LSTM model
            buffer_size: Number of samples to accumulate before prediction
        """
        self.model_path = model_path
        self.buffer_size = buffer_size
        self.samples_buffer = []
        
        # Load the model
        print(f"Loading model from {model_path}")
        self.model = load_model(model_path)
        print("Model loaded successfully")
        
        # Latest prediction result
        self.latest_prediction = None
        self.latest_confidence = 0.0
        
    def add_sample(self, csi_raw_data):
        """Add a CSI sample to the buffer and make prediction if buffer is full"""
        if len(csi_raw_data) in [128, 256, 384]:
            self.samples_buffer.append(csi_raw_data)
            
            # Make prediction when buffer is full
            if len(self.samples_buffer) >= self.buffer_size:
                self.make_prediction()
                
    def make_prediction(self):
        """Process buffered data and make prediction"""
        if not self.samples_buffer:
            return
            
        try:
            # Preprocess data
            processed_data = self.preprocess_data(self.samples_buffer)
            
            # Make prediction
            predictions = self.model.predict(processed_data)
            
            # Process predictions (example: get class with highest probability)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate most common prediction
            unique_classes, counts = np.unique(predicted_classes, return_counts=True)
            most_common_class = unique_classes[np.argmax(counts)]
            confidence = counts[np.argmax(counts)] / len(predictions)
            
            # Update latest prediction
            self.latest_prediction = most_common_class
            self.latest_confidence = confidence
            
            # Print prediction
            print(f"Prediction: Class {most_common_class} (confidence: {confidence:.2f})")
            
            # Clear buffer
            self.samples_buffer = []
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Clear buffer on error
            self.samples_buffer = []
            
    def preprocess_data(self, samples):
        """
        Preprocess the CSI data samples
        
        Args:
            samples: List of CSI data samples
            
        Returns:
            Preprocessed data ready for model inference
        """
        # Convert to numpy array
        X = np.array([np.array(sample) for sample in samples])
        
        # Find the maximum length
        max_length = max(len(sample) for sample in X)
        
        # Pad sequences to the same length
        X_padded = np.array([np.pad(sample, (0, max_length - len(sample)), mode='constant') for sample in X])
        
        # Normalize
        X_normalized = (X_padded - np.mean(X_padded)) / (np.std(X_padded) + 1e-8)
        
        return X_normalized
        
    def get_latest_prediction(self):
        """Get the latest prediction and confidence"""
        return self.latest_prediction, self.latest_confidence

class csi_data_graphical_window(QWidget):
    def __init__(self, prediction_engine):
        super().__init__()
        
        self.prediction_engine = prediction_engine
        self.setup_ui()
        
    def setup_ui(self):
        self.resize(1280, 720)
        layout = QVBoxLayout(self)
        
        # Create prediction display
        self.prediction_label = QLabel("Prediction: Waiting for data...")
        self.prediction_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #2980b9;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)
        
        # Create confidence display
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("font-size: 14pt; color: #27ae60;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        
        # Create plot widget
        self.plotWidget_ted = PlotWidget(self)
        layout.addWidget(self.plotWidget_ted)
        
        self.plotWidget_ted.setYRange(-20, 100)
        self.plotWidget_ted.addLegend()
        
        self.csi_amplitude_array = np.abs(csi_data_array)
        self.curve_list = []
        
        for i in range(CSI_DATA_COLUMNS):
            curve = self.plotWidget_ted.plot(
                self.csi_amplitude_array[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
            self.curve_list.append(curve)
        
        self.timer = pq.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)
        
        self.setLayout(layout)
        self.setWindowTitle("CSI Visualization with Heart Rate Prediction")

    def update_data(self):
        # Update CSI plot
        self.csi_amplitude_array = np.abs(csi_data_array)
        for i in range(CSI_DATA_COLUMNS):
            self.curve_list[i].setData(self.csi_amplitude_array[:, i])
            
        # Update prediction display
        pred_class, confidence = self.prediction_engine.get_latest_prediction()
        
        if pred_class is not None:
            # Display the class name instead of just the class number
            class_name = CLASS_NAMES[pred_class] if 0 <= pred_class < len(CLASS_NAMES) else f"Unknown Class {pred_class}"
            self.prediction_label.setText(f"Prediction: {class_name}")
            self.confidence_label.setText(f"Confidence: {confidence:.2f}")
            
            # Change color based on confidence
            if confidence > 0.8:
                self.confidence_label.setStyleSheet("font-size: 14pt; color: #27ae60;")  # Green for high confidence
            elif confidence > 0.5:
                self.confidence_label.setStyleSheet("font-size: 14pt; color: #f39c12;")  # Orange for medium confidence
            else:
                self.confidence_label.setStyleSheet("font-size: 14pt; color: #c0392b;")  # Red for low confidence


def csi_data_read_parse(port: str, csv_writer, log_file_fd, prediction_engine):
    ser = serial.Serial(port=port, baudrate=921600,
                        bytesize=8, parity='N', stopbits=1)
    if ser.isOpen():
        print("open success")
    else:
        print("open failed")
        return

    while True:
        strings = str(ser.readline())
        if not strings:
            break

        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        index = strings.find('CSI_DATA')

        if index == -1:
            # Save serial output other than CSI data
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        csv_reader = csv.reader(StringIO(strings))
        csi_data = next(csv_reader)

        if len(csi_data) != len(DATA_COLUMNS_NAMES):
            print("element number is not equal")
            log_file_fd.write("element number is not equal\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        try:
            csi_raw_data = json.loads(csi_data[-1])
        except json.JSONDecodeError:
            print("data is incomplete")
            log_file_fd.write("data is incomplete\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Reference on the length of CSI data and usable subcarriers
        # https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html#wi-fi-channel-state-information
        if len(csi_raw_data) != 128 and len(csi_raw_data) != 256 and len(csi_raw_data) != 384:
            print(f"element number is not equal: {len(csi_raw_data)}")
            log_file_fd.write(f"element number is not equal: {len(csi_raw_data)}\n")
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        csv_writer.writerow(csi_data)

        # Rotate data to the left
        csi_data_array[:-1] = csi_data_array[1:]

        if len(csi_raw_data) == 128:
            csi_vaid_subcarrier_len = CSI_DATA_LLFT_COLUMNS
        else:
            csi_vaid_subcarrier_len = CSI_DATA_COLUMNS

        for i in range(csi_vaid_subcarrier_len):
            csi_data_array[-1][i] = complex(csi_raw_data[csi_vaid_subcarrier_index[i] * 2 + 1],
                                            csi_raw_data[csi_vaid_subcarrier_index[i] * 2])
                                            
        # Add data to prediction engine
        prediction_engine.add_sample(csi_raw_data)

    ser.close()
    return


class SubThread (QThread):
    def __init__(self, serial_port, save_file_name, log_file_name, prediction_engine):
        super().__init__()
        self.serial_port = serial_port
        self.prediction_engine = prediction_engine

        save_file_fd = open(save_file_name, 'w')
        self.log_file_fd = open(log_file_name, 'w')
        self.csv_writer = csv.writer(save_file_fd)
        self.csv_writer.writerow(DATA_COLUMNS_NAMES)

    def run(self):
        csi_data_read_parse(self.serial_port, self.csv_writer, self.log_file_fd, self.prediction_engine)

    def __del__(self):
        self.wait()
        self.log_file_fd.close()


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port, display it graphically, and show heart rate predictions")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_data.csv',
                        help="Save the data printed by the serial port to a file")
    parser.add_argument('-l', '--log', dest="log_file", action="store", default="./csi_data_log.txt",
                        help="Save other serial data the bad CSI data to a log file")
    parser.add_argument('-m', '--model', dest='model_path', action='store', default='csi_lstm_model.h5',
                        help="Path to the saved LSTM model")
    parser.add_argument('-b', '--buffer', dest='buffer_size', action='store', type=int, default=10,
                        help="Number of samples to accumulate before prediction")

    args = parser.parse_args()
    serial_port = args.port
    file_name = args.store_file
    log_file_name = args.log_file
    model_path = args.model_path
    buffer_size = args.buffer_size

    app = QApplication(sys.argv)
    
    # Initialize prediction engine
    prediction_engine = CSIPredictionEngine(model_path=model_path, buffer_size=buffer_size)

    # Start data collection thread
    subthread = SubThread(serial_port, file_name, log_file_name, prediction_engine)
    subthread.start()

    # Create and show window
    window = csi_data_graphical_window(prediction_engine)
    window.show()

    sys.exit(app.exec())