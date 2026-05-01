#!/usr/bin/env python3
"""
Hybrid Agentic AI MAS Demonstrator for Prescriptive Maintenance.
Based on: Farahani, Khan, Wuest (2026) - Journal of Manufacturing Systems.
Keywords: Agentic AI, Industrial IoT, Predictive Maintenance, Multi-Agent Systems
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    from langchain.llms import Ollama
except ImportError:
    try:
        from langchain.chat_models import Ollama
    except ImportError:
        from langchain import Ollama

try:
    from langchain.agents import initialize_agent, Tool
except ImportError:
    from langchain.agents import initialize_agent
    from langchain.tools import Tool
from langchain.agents import AgentType

import paho.mqtt.client as mqtt
import json
import time

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC = os.getenv("MQTT_TOPIC", "iiot/sensors")

def get_iiot_dataset(n_samples=500, broker=BROKER, port=PORT, topic=TOPIC):
    messages = []
    
    def on_message(client, userdata, msg):
        data = json.loads(msg.payload.decode())
        messages.append(data)
        if len(messages) >= n_samples:
            client.disconnect()
    
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker, port, 60)
    client.subscribe(topic)
    client.loop_start()
    
    # Wait until collected or timeout
    timeout = time.time() + 60  # 60s timeout
    while len(messages) < n_samples and time.time() < timeout:
        time.sleep(0.1)
    
    client.loop_stop()
    client.disconnect()
    
    # Build DataFrame
    df = pd.DataFrame(messages)
    
    # Add inspection_hrs if not present
    if "inspection_hrs" not in df.columns:
        df["inspection_hrs"] = np.random.uniform(1, 8, len(df))
    
    # Compute maintenance_priority
    conditions = [
        (df["vibration"] > 0.7) | (df["temperature"] > 90) | (df["downtime_cost"] > 4000),
        (df["vibration"] > 0.5) | (df["temperature"] > 80) | (df["downtime_cost"] > 2500)
    ]
    df["maintenance_priority"] = np.select(conditions, ["High", "Medium"], default="Low")
    return df

class PerceptionAgent:
    def run(self, df):
        print("[PerceptionAgent] Inspecting dataset...")
        report = {"shape": df.shape, "columns": list(df.columns),
                  "missing_values": df.isnull().sum().to_dict()}
        print(f"  Shape: {report['shape']}")
        return report

class PreprocessingAgent:
    def __init__(self):
        self.scaler = StandardScaler()
    def run(self, df, target_col="maintenance_priority"):
        id_cols = [c for c in df.columns if "id" in c.lower()]
        feature_cols = [c for c in df.columns if c != target_col
                        and c not in id_cols
                        and df[c].dtype in [np.float64, np.int64]]
        X = df[feature_cols].fillna(df[feature_cols].median())
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=feature_cols)
        return X_scaled, df[target_col], feature_cols

class AnalysisAgent:
    def run(self, X, y, task="classification"):
        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"[AnalysisAgent] Classification Accuracy: {acc:.4f}")
            self.feature_importances_ = dict(zip(X.columns, model.feature_importances_))
            return {"accuracy": acc, "predictions": y_pred, "X_test": X_test}
        elif task == "anomaly_detection":
            iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
            iso.fit(X)
            labels = iso.predict(X)
            n_anom = (labels == -1).sum()
            print(f"[AnalysisAgent] Anomalies detected: {n_anom}")
            return {"anomaly_labels": labels, "n_anomalies": n_anom}

class OptimizationAgent:
    def run(self, df, predictions, feature_importances, top_n=5):
        df = df.copy()
        df["predicted_priority"] = predictions
        critical = df[df["predicted_priority"] == "High"].nlargest(top_n, "downtime_cost")
        recs = []
        for _, r in critical.iterrows():
            rec = {"machine_id": r.get("machine_id", "N/A"),
                   "priority": r["predicted_priority"],
                   "cost": f"${r['downtime_cost']:.0f}",
                   "action": "Immediate inspection required"}
            recs.append(rec)
            print(f"[OptimizationAgent] {rec}")
        return recs

class OrchestratorAgent:
    def __init__(self):
        self.llm = Ollama(model="llama2", base_url="http://ollama:11434")
        self.tools = [
            Tool(
                name="Preprocessing",
                description="Preprocess the IIoT dataset by scaling numerical features and preparing for machine learning analysis.",
                func=self._run_preprocessing
            ),
            Tool(
                name="Analysis",
                description="Perform classification to predict maintenance priority and anomaly detection to identify outliers in the data.",
                func=self._run_analysis
            ),
            Tool(
                name="Optimization",
                description="Generate optimization recommendations for maintenance based on analysis results, prioritizing high-risk machines.",
                func=self._run_optimization
            )
        ]
        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    def _run_preprocessing(self, query):
        pre = PreprocessingAgent()
        self.X, self.y, self.feats = pre.run(self.df)
        return "Preprocessing completed successfully. Features scaled and prepared."

    def _run_analysis(self, query):
        ana = AnalysisAgent()
        self.clf = ana.run(self.X, self.y, "classification")
        ana2 = AnalysisAgent()
        self.ad = ana2.run(self.X, self.y, "anomaly_detection")
        self.feature_importances = ana.feature_importances_
        return f"Analysis completed. Classification accuracy: {self.clf['accuracy']:.4f}, Anomalies detected: {self.ad['n_anomalies']}"

    def _run_optimization(self, query):
        opt = OptimizationAgent()
        test_idx = self.clf["X_test"].index
        self.recs = opt.run(self.df.loc[test_idx], self.clf["predictions"], self.feature_importances)
        return f"Optimization completed. Generated {len(self.recs)} maintenance recommendations."

    def run(self, df, input_text="run full predictive maintenance analysis"):
        self.df = df
        print("=== AGENTIC AI MAS — PRESCRIPTIVE MAINTENANCE ===")
        result = self.agent.run(input_text)
        print(f"Agent result: {result}")
        if hasattr(self, 'recs'):
            print(f"Done. Accuracy={self.clf['accuracy']:.4f}, Anomalies={self.ad['n_anomalies']}, Recs={len(self.recs)}")
            return self.recs
        else:
            return []

if __name__ == "__main__":
    
    orchestrator = OrchestratorAgent()

    time.sleep(120)  # wait

    while True:
        
        # 1. get data
        df = get_iiot_dataset(500)
    
        # 2. run MAS
        recs = orchestrator.run(df)

        # 3. (optional) send recommendations to log to file
        for rec in recs:
            print(f"[yellow] Recommended action for {rec['machine_id']}: {rec['action']} (Priority: {rec['priority']}, Estimated Cost: {rec['cost']})")

            # Example: publish to a folder /app/results
            with open("/app/results/maintenance_recommendations.txt", "a") as f:
                f.write(f"{time.ctime()}: {rec['machine_id']} - {rec['action']} (Priority: {rec['priority']}, Estimated Cost: {rec['cost']})\n")

        # 4. wait before next cycle
        time.sleep(120)  # Run every 2 minutes (120s)



