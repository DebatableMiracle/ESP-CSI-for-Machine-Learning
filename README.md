# ESP-CSI-for-Machine-Learning

## Introduction to CSI

Channel State Information (CSI) is a crucial parameter in wireless communication that describes the characteristics of a wireless channel, including signal amplitude, phase, and delay. By analyzing changes in CSI, we can infer environmental variations that affect the wireless channel. CSI is highly sensitive to both large movements (walking, running) and fine actions (breathing, chewing). These capabilities make it applicable in various fields like smart environment monitoring, human activity detection, and wireless positioning.



## Enhancements in ESP-CSI-for-Machine-Learning

This repository builds upon the original [esp-csi](https://github.com/espressif/esp-csi) project by introducing several new features, making it more suitable for real-time machine learning applications:

- **Bluetooth Integration:** The ESP32’s BLE capabilities are leveraged to enhance detection and interaction with nearby devices. (not working totally yet)
- **Real-time Machine Learning Pipeline:** Allows direct execution of ML models on CSI data for real-time inference.
- **Expanded Dataset Support:** Compatible with multiple data sources for training and testing ML models.
- **Improved CSI Processing Tools:** Enhanced scripts for real-time parsing, visualization, and feature extraction.

## Basic Knowledge

For better understanding of CSI technology, refer to these documents:

- [Signal Processing Fundamentals](./docs/en/Signal-Processing-Fundamentals.md)
- [OFDM Introduction](./docs/en/OFDM-introduction.md)
- [Wireless Channel Fundamentals](./docs/en/Wireless-Channel-Fundamentals.md)
- [Introduction to Wireless Location](./docs/en/Introduction-to-Wireless-Location.md)
- [Wireless Indicators CSI and RSSI](./docs/en/Wireless-indicators-CSI-and-RSSI.md)
- [CSI Applications](./docs/en/CSI-Applications.md)

## Advantages of ESP-CSI-for-Machine-Learning

- **Full ESP32 Series Support:** Works with ESP32, ESP32-S2, ESP32-C3, ESP32-S3, and ESP32-C6.
- **Built-in Bluetooth Assistance:** BLE scanning enhances detection in CSI-based applications.
- **On-Device Machine Learning:** ESP32’s AI capabilities enable lightweight ML inference.
- **OTA Updates:** Upgrade features remotely without additional hardware costs.

## Example Projects

### [get-started](./examples/get-started)

Basic CSI acquisition and initial analysis.

- [csi_recv](./examples/get-started/csi_recv) – ESP32 as CSI receiver.
- [csi_send](./examples/get-started/csi_send) – ESP32 as CSI sender.
- [csi_recv_router](./examples/get-started/csi_recv_router) – Router as sender, ESP32 receives CSI data.
- [tools](./examples/get-started/tools) – Scripts for CSI data processing and visualization.

### [esp-radar](./examples/esp-radar)

CSI-based applications, including cloud reporting and activity detection.

- [connect_rainmaker](./examples/esp-radar/connect_rainmaker) – Upload CSI data to Espressif’s RainMaker cloud.
- [console_test](./examples/esp-radar/console_test) – Interactive console for real-time CSI data analysis.

### [machine-learning](./machine_learning)

Real-time machine learning pipeline for visualising and predictions of CSI-based applications.
(I'm working on heart rate prediction using CSI so some resources are made for those)

- **Feature Extraction:** Scripts for preprocessing CSI data.
- **Real-time Model Execution:** Runs machine learning models on incoming CSI data.
- **Training Pipeline:** Links to external repository for training custom CSI-based models.

## Getting Started with some of the ESP32's CSI and AI (my current workflow and soon yours!)

### Collecting data:
Use [esp-csi](https://github.com/espressif/esp-csi) or my own repository's components (I've built my repo upon that).

Connect two Esp32's to your PC, upload esp_recv and esp_send to the respective ESP32s using the following codes.

''' 
# csi_send
cd /examples/get-started/csi_send
idf.py set-target esp32
idf.py flash -b 921600 -p /dev/ttyUSB0 monitor
'''

'''
cd /examples/get-started/csi_recv
idf.py set-target esp32
idf.py flash -b 921600 -p /dev/ttyUSB1
'''

Change set-target depending upon your ESP32 chip, if wrong one is selected, you'll see some errors. You can raise issues if you need help.
-b sets the baud rate of the device, make sure the baud rates match your whole setup (I use 921600 but increased baud rates may show some better results too)
-p is the port of the device, the port in codeblock is default for linux systems, you can check your ports and use accordingly.

Yayy you're seeing the data in terminal probably!

'''
#to check your data in terminal
idf.py -b 921600 -p /dev/ttyUSB1 monitor
'''

**Visualize the data**

'''
cd /examples/get-started/tools
# Install python related dependencies
pip install -r requirements.txt

# Graphical display
python csi_data_read_parse.py -p /dev/ttyUSB1

'''
**Saving the data**

But you're probably not just going to copy paste your data from terminal right?
'''
idf.py -b 921600 -p /dev/ttyUSB1 monitor | grep "CSI_DATA" > my-experiment-file.csv
'''

 Now you can label them as you want, use an RTC to add labels through other sensors and concatenate. (coming soon too!)


## Getting CSI Data

### Router-Based CSI

ESP32 sends a ping to the router and receives CSI from the reply.
- **Pros:** Simple setup (ESP32 + router).
- **Cons:** Dependent on router capabilities.

### Device-to-Device CSI

Two ESP32 devices exchange packets and extract CSI data.
- **Pros:** No reliance on a fixed router.
- **Cons:** Requires multiple ESP32 devices.

### Broadcast CSI

A dedicated transmitter broadcasts packets, with multiple ESP32s receiving CSI.
- **Pros:** High accuracy, no router dependency.
- **Cons:** Requires an additional transmitting device.

## Notes

1. External IPEX antennas provide better CSI results than PCB antennas.
2. Conduct tests in an unmanned environment to avoid interference.

## Related Resources

- [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/index.html)
- [ESP-WIFI-CSI Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information)
- [Issue Tracker](https://github.com/espressif/esp-csi/issues)

## Additional Repositories

For training and improving CSI-based ML models, check out my other repository:

🔗 [CSI Machine Learning Training Repository](https://github.com/DebatableMiracle/csi-ml-train)

## References

1. [Through-Wall Human Pose Estimation Using Radio Signals](http://rfpose.csail.mit.edu/)
2. [Awesome WiFi CSI Sensing Resources](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing)

