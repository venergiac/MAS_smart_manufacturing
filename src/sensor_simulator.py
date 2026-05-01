#!/usr/bin/env python3
"""IIoT Sensor Simulator — publishes synthetic sensor readings via MQTT."""
import os, time, json, random
import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC = os.getenv("MQTT_TOPIC", "iiot/sensors")
INTERVAL = float(os.getenv("PUBLISH_INTERVAL_SEC", 2))
N_MACH = int(os.getenv("N_MACHINES", 10))

client = mqtt.Client()
client.connect(BROKER, PORT, 60)
client.loop_start()

while True:
    for i in range(1, N_MACH + 1):
        payload = {
            "machine_id": f"M{str(i).zfill(3)}",
            "timestamp": time.time(),
            "temperature": round(random.gauss(75, 15), 2),
            "vibration": round(random.gauss(0.5, 0.2), 3),
            "pressure": round(random.gauss(100, 20), 2),
            "acoustic_level": round(random.gauss(65, 10), 2),
            "downtime_cost": round(random.uniform(500, 5000), 2)
        }
        client.publish(TOPIC, json.dumps(payload))
    time.sleep(INTERVAL)
