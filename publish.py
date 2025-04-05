import math
import time
import paho.mqtt.publish as publish
from PyQt5.QtCore import QTimer, QObject
from PyQt5.QtWidgets import QApplication
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MQTTPublisher(QObject):
    def __init__(self, broker, topics):
        super().__init__()
        self.broker = broker
        self.topics = topics if isinstance(topics, list) else [topics]
        self.count = 0

        self.frequency = 10
        self.amplitude = (46537 - 16390) / 2
        self.offset = (46537 + 16390) / 2

        self.sample_rate = 16384
        self.time_per_message = 1.0
        self.current_time = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.publish_message)
        self.timer.start(1000)

    def publish_message(self):
        if self.count < 200:
            values = []
            for i in range(self.sample_rate):
                t = self.current_time + (i / self.sample_rate)
                value = self.offset + self.amplitude * math.sin(2 * math.pi * self.frequency * t)
                values.append(round(value, 2))

            self.current_time += 1
            message = ",".join(map(str, values))
            # logging.debug(message)

            for topic in self.topics:
                try:
                    publish.single(topic, message, hostname=self.broker, qos=1)
                    logging.info(f"[{self.count}] Published to {topic}: {message[:50]}... ({self.sample_rate} values)")
                except Exception as e:
                    logging.error(f"Failed to publish to {topic}: {str(e)}")

            self.count += 1
        else:
            self.timer.stop()
            # logging.info("Publishing stopped after 50 messages.")

if __name__ == "__main__":
    app = QApplication([])
    broker = "192.168.1.171"
    topics = ["sarayu/tag2/topic2|m/s"]
    mqtt_publisher = MQTTPublisher(broker, topics)
    app.exec_()








# import math
# import time
# import paho.mqtt.publish as publish
# from PyQt5.QtCore import QTimer, QObject
# from PyQt5.QtWidgets import QApplication
# import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# class MQTTPublisher(QObject):
#     def __init__(self, broker, topics):
#         super().__init__()
#         self.broker = broker
#         self.topics = topics if isinstance(topics, list) else [topics]
#         self.count = 0

#         self.frequency = 7
#         self.amplitude = (65279 - 0) / 2
#         self.offset = (65279 + 0) / 2

#         self.sample_rate = 4096
#         self.time_per_message = 1.0
#         self.current_time = 0.0

#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.publish_message)
#         self.timer.start(1000)

#     def format_16bit_values(self, value):
#         value = int(value)  # Ensure it's an integer
#         formatted = f"{value:016b}"  # Convert to 16-bit binary
#         chunks = [formatted[i:i+4] for i in range(0, 16, 4)]  # Split into 4-bit chunks
#         return ",".join(chunks)  # Return comma-separated chunks

#     def publish_message(self):
#         if self.count < 50:
#             values = []
#             for i in range(self.sample_rate):
#                 t = self.current_time + (i / self.sample_rate)
#                 value = self.offset + self.amplitude * math.sin(1 * math.pi * self.frequency * t)
#                 value = max(0, min(65279, round(value)))  # Clamp to 16-bit range
#                 values.append(self.format_16bit_values(value))

#             self.current_time += 1
#             message = ",".join(values)

#             for topic in self.topics:
#                 try:
#                     publish.single(topic, message, hostname=self.broker, qos=1)
#                     logging.info(f"[{self.count}] Published to {topic}: {message[:50]}... ({self.sample_rate} values)")
#                 except Exception as e:
#                     logging.error(f"Failed to publish to {topic}: {str(e)}")

#             self.count += 1
#         else:
#             self.timer.stop()
#             logging.info("Publishing stopped after 50 messages.")

# if __name__ == "__main__":
#     app = QApplication([])
#     broker = "192.168.1.173"
#     topics = ["sarayu/tag2/topic2|m/s"]
#     mqtt_publisher = MQTTPublisher(broker, topics)
#     app.exec_()




# import math
# import time
# import struct
# import paho.mqtt.publish as publish
# from PyQt5.QtCore import QTimer, QObject
# from PyQt5.QtWidgets import QApplication
# import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# class MQTTPublisher(QObject):
#     def __init__(self, broker, topics):
#         super().__init__()
#         self.broker = broker
#         self.topics = topics if isinstance(topics, list) else [topics]
#         self.count = 0

#         self.frequency = 10
#         self.amplitude = (65279 - 0) / 2
#         self.offset = (65279 + 0) / 2

#         self.sample_rate = 4096
#         self.time_per_message = 1.0
#         self.current_time = 0.0

#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.publish_message)
#         self.timer.start(1000)

#     def generate_values(self):
#         values = []
#         for i in range(self.sample_rate):
#             t = self.current_time + (i / self.sample_rate)
#             value = self.offset + self.amplitude * math.sin(1 * math.pi * self.frequency * t)
#             value = max(0, min(65279, round(value)))  # Clamp to 16-bit range
#             values.append(value)
#         return values

#     def publish_message(self):
#         if self.count < 50:
#             values = self.generate_values()
#             packed_data = struct.pack('<' + 'H' * len(values), *values)
#             unpacked_values = list(struct.unpack('<4096H', packed_data))
#             message = ",".join(map(str, unpacked_values))

#             self.current_time += 1

#             for topic in self.topics:
#                 try:
#                     publish.single(topic, message, hostname=self.broker, qos=1)
#                     logging.info(f"[{self.count}] Published to {topic}: {message[:50]}... ({self.sample_rate} values)")
#                 except Exception as e:
#                     logging.error(f"Failed to publish to {topic}: {str(e)}")

#             self.count += 1
#         else:
#             self.timer.stop()
#             logging.info("Publishing stopped after 50 messages.")

# if __name__ == "__main__":
#     app = QApplication([])
#     broker = "192.168.1.173"
#     topics = ["sarayu/tag2/topic2|m/s"]
#     mqtt_publisher = MQTTPublisher(broker, topics)
#     app.exec_()




# import math
# import logging
# import sys
# from PyQt5.QtCore import QTimer, QObject
# from PyQt5.QtWidgets import QApplication
# import paho.mqtt.publish as publish

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# class MQTTPublisher(QObject):
#     def __init__(self, broker, topics):
#         super().__init__()
#         self.broker = broker
#         self.topics = topics if isinstance(topics, list) else [topics]
#         self.count = 0

#         # Signal parameters
#         self.frequency = 5  # Hz
#         self.amplitude = (65279 - 0) / 2  # Half of 16-bit range
#         self.offset = (65279 + 0) / 2  # Midpoint of 16-bit range

#         # Sampling parameters
#         self.sample_rate = 4096  # Samples per second
#         self.time_per_message = 1.0  # Seconds per message
#         self.current_time = 0.0

#         # Set up timer to publish every second
#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.publish_message)
#         self.timer.start(1000)  # 1000 ms = 1 second

#     def generate_values(self):
#         """Generate a list of sinusoidal values and split into 4-bit segments."""
#         values = []
#         for i in range(self.sample_rate):
#             t = self.current_time + (i / self.sample_rate)
#             value = self.offset + self.amplitude * math.sin(2 * math.pi * self.frequency * t)
#             value = max(0, min(65279, round(value)))  # Clamp to 16-bit range
            
#             # Split 16-bit value into four 4-bit segments
#             segments = [
#                 (value >> 12) & 0xF,  # Bits 15-12
#                 (value >> 8) & 0xF,   # Bits 11-8
#                 (value >> 4) & 0xF,   # Bits 7-4
#                 value & 0xF           # Bits 3-0
#             ]
#             values.extend(segments)  # Add all four segments
#         return values

#     def publish_message(self):
#         """Publish generated 4-bit segments to MQTT topics."""
#         if self.count >= 50:
#             self.timer.stop()
#             logging.info("Publishing stopped after 50 messages.")
#             QApplication.quit()
#             return

#         # Generate values (now 4-bit segments)
#         segments = self.generate_values()

#         # Update time for next batch
#         self.current_time += self.time_per_message

#         # Convert the list of segments to a string message
#         message = ''.join(map(str, segments))  # Join list elements into a single string

#         # Publish to all topics
#         for topic in self.topics:
#             try:
#                 publish.single(topic, message, hostname=self.broker, qos=1)
#                 logging.info(f"[{self.count}] Published to {topic}: {message[:50]}... ({len(segments)} values)")
#             except Exception as e:
#                 logging.error(f"Failed to publish to {topic}: {str(e)}")

#         self.count += 1

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     broker = "192.168.1.173"
#     topics = ["sarayu/tag2/topic2|m/s"]
#     mqtt_publisher = MQTTPublisher(broker, topics)
#     sys.exit(app.exec_())
