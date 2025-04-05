import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime
import logging
import struct

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class MQTTHandler(QObject):
    data_received = pyqtSignal(str, list)  # Signal: tag_name, values

    def __init__(self, db, project_name):
        super().__init__()
        self.db = db
        self.project_name = project_name
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker = "192.168.1.171"  # Replace with your actual broker IP
        self.port = 1883  # Default MQTT port
        self.subscribed_topics = set()
        self.running = False

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            logging.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {str(e)}")
            raise

    def start(self):
        if not self.running:
            self.connect()
            self.client.loop_start()
            self.running = True
            logging.info("MQTT loop started")

    def stop(self):
        if self.running:
            self.client.loop_stop()
            self.client.disconnect()
            self.running = False
            self.subscribed_topics.clear()
            logging.info("MQTT loop stopped and client disconnected")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Connected to MQTT broker with result code {rc}")
            self.subscribe_to_topics()
        else:
            logging.error(f"Connection failed with result code {rc}")

    def subscribe_to_topics(self):
        tags = list(self.db.tags_collection.find({"project_name": self.project_name}))
        if not tags:
            logging.warning(f"No tags found for project {self.project_name}")
            return
        for tag in tags:
            topic = tag["tag_name"]
            if topic not in self.subscribed_topics:
                self.client.subscribe(topic, qos=1)
                self.subscribed_topics.add(topic)
                logging.info(f"Subscribed to topic: {topic}")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode("utf-8")
        logging.debug(f"Received message on {topic}: {payload[:50]}...")

        try:
            values = [float(x.strip()) for x in payload.split(",") if x.strip()]
            logging.debug(values)
            if not values:
                raise ValueError("Empty or invalid payload")
            tag_name = topic
            timestamp = datetime.now().isoformat()
            success, message = self.db.update_tag_value(self.project_name, tag_name, values, timestamp)
            if success:
                logging.info(f"Stored {len(values)} values for {tag_name}")
                self.data_received.emit(tag_name, values)
            else:
                logging.error(f"Failed to store values: {message}")
        except ValueError as ve:
            logging.error(f"Invalid payload format on {topic}: {str(ve)}")
        except Exception as e:
            logging.error(f"Error processing message on {topic}: {str(e)}")


    # def on_message(self, client, userdata, msg):
    #     topic = msg.topic
    #     payload = msg.payload  # This is binary data

    #     logging.debug(f"Received message on {topic}, payload size: {len(payload)} bytes")

    #     try:
    #         # Assuming uint16_t array (2 bytes per value)
    #         values = list(struct.unpack(f"{len(payload) // 2}H", payload))
    #         logging.debug(values)
            
    #         if not values:
    #             raise ValueError("Empty or invalid payload")
            
    #         tag_name = topic
    #         timestamp = datetime.now().isoformat()
            
    #         success, message = self.db.update_tag_value(self.project_name, tag_name, values, timestamp)
    #         if success:
    #             logging.info(f"Stored {len(values)} values for {tag_name}")
    #             self.data_received.emit(tag_name, values)
    #         else:
    #             logging.error(f"Failed to store values: {message}")
        
    #     except struct.error as se:
    #         logging.error(f"Failed to unpack binary data on {topic}: {str(se)}")
    #     except Exception as e:
    #         logging.error(f"Error processing message on {topic}: {str(e)}")