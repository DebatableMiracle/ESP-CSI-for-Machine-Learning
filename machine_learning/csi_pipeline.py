import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import sys
import json
import argparse
import numpy as np
import serial
import time
import re
from tensorflow.keras.models import load_model
import threading
import queue

class CSIPipeline:
    def __init__(self, serial_port, baud_rate=921600, model_path='csi_lstm_model.h5', buffer_size=10):
        """
        Initialize the CSI processing pipeline
        
        Args:
            serial_port: Serial port to read from (e.g., '/dev/ttyUSB1')
            baud_rate: Baud rate for serial communication
            model_path: Path to the saved LSTM model
            buffer_size: Number of samples to accumulate before prediction
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.model_path = model_path
        self.buffer_size = buffer_size
        self.running = False
        
        # Queue for thread communication
        self.data_queue = queue.Queue(maxsize=100)
        
        # Load the model
        print(f"Loading model from {model_path}")
        self.model = load_model(model_path)
        print("Model loaded successfully")
        
        # Statistics
        self.processed_samples = 0
        self.start_time = None
    
    def start(self):
        """Start the CSI pipeline"""
        if self.running:
            print("Pipeline is already running")
            return
            
        self.running = True
        self.start_time = time.time()
        
        # Start threads
        self.reader_thread = threading.Thread(target=self.serial_reader)
        self.processor_thread = threading.Thread(target=self.processor)
        
        self.reader_thread.daemon = True
        self.processor_thread.daemon = True
        
        self.reader_thread.start()
        self.processor_thread.start()
        
        print(f"Pipeline started. Reading from {self.serial_port}")
    
    def stop(self):
        """Stop the CSI pipeline"""
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'reader_thread') and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2)
        
        print("Pipeline stopped")
        
        # Print statistics
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            print(f"Processed {self.processed_samples} samples in {elapsed_time:.2f} seconds")
            print(f"Average rate: {self.processed_samples / elapsed_time:.2f} samples/second")
    
    def serial_reader(self):
        """Read CSI data from serial port and put it in the queue"""
        try:
            ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=1
            )
            print(f"Serial port {self.serial_port} opened successfully")
            
            while self.running:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='replace').strip()
                    
                    # Check if this is a CSI data line
                    if 'CSI_DATA' in line:
                        # Process and enqueue the CSI data
                        try:
                            # Extract the data part (assuming it's in JSON format as in your original code)
                            parts = line.split(',')
                            if len(parts) > 24:  # Check if we have enough parts (based on your DATA_COLUMNS_NAMES)
                                json_data = parts[-1]  # Last part should be the data
                                
                                # Parse the JSON data
                                csi_raw_data = json.loads(json_data)
                                
                                # Validate data length
                                if len(csi_raw_data) in [128, 256, 384]:
                                    # Put data in queue
                                    if not self.data_queue.full():
                                        self.data_queue.put(csi_raw_data)
                                    else:
                                        print("Warning: Queue is full, dropping sample")
                        except Exception as e:
                            print(f"Error processing CSI data: {e}")
                else:
                    # No data available, sleep a bit
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Serial reader error: {e}")
            self.running = False
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()
                print("Serial port closed")
    
    def preprocess_data(self, samples):
        """
        Preprocess the CSI data samples similar to your notebook code
        
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
        X_normalized = (X_padded - np.mean(X_padded)) / (np.std(X_padded) + 1e-8)  # Added small epsilon to avoid division by zero
        
        return X_normalized
    
    def processor(self):
        """Process CSI data from the queue and make predictions"""
        samples_buffer = []
        
        while self.running:
            try:
                # Try to get data from queue, timeout to check if running flag changed
                try:
                    csi_data = self.data_queue.get(timeout=0.5)
                    samples_buffer.append(csi_data)
                    self.data_queue.task_done()
                except queue.Empty:
                    # No data available in queue
                    continue
                
                # Make prediction when buffer is full
                if len(samples_buffer) >= self.buffer_size:
                    # Preprocess data
                    processed_data = self.preprocess_data(samples_buffer)
                    
                    # Make prediction
                    predictions = self.model.predict(processed_data)
                    
                    # Process predictions (example: get class with highest probability)
                    predicted_classes = np.argmax(predictions, axis=1)
                    
                    # Calculate most common prediction
                    unique_classes, counts = np.unique(predicted_classes, return_counts=True)
                    most_common_class = unique_classes[np.argmax(counts)]
                    
                    # Print prediction
                    print(f"Prediction: Class {most_common_class} (confidence: {counts[np.argmax(counts)] / len(predictions):.2f})")
                    
                    # Update statistics
                    self.processed_samples += len(samples_buffer)
                    
                    # Clear buffer
                    samples_buffer = []
                    
            except Exception as e:
                print(f"Processor error: {e}")
                
                # Clear buffer on error
                samples_buffer = []

def main():
    parser = argparse.ArgumentParser(description="Real-time CSI Processing Pipeline")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of CSI device (e.g., /dev/ttyUSB1)")
    parser.add_argument('-m', '--model', dest='model_path', action='store', default='csi_lstm_model.h5',
                        help="Path to the saved LSTM model")
    parser.add_argument('-b', '--buffer', dest='buffer_size', action='store', type=int, default=10,
                        help="Number of samples to accumulate before prediction")
    
    args = parser.parse_args()
    
    # Create and start the pipeline
    pipeline = CSIPipeline(
        serial_port=args.port,
        model_path=args.model_path,
        buffer_size=args.buffer_size
    )
    
    try:
        pipeline.start()
        
        # Keep running until user stops
        print("Press Ctrl+C to stop the pipeline")
        while pipeline.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()