import logging
from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random


## dessa kan du inte ta med
#from sklearn.cluster import AffinityPropagation
#from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# --- Helper functions for hybrid clustering ---


class ServerFunctions(ServerFunctionsBase):
    def __init__(self):
        self.client_clusters = {}  # Dictionary to track client-cluster assignments
        self.cluster_models = {}   # Stores models for each cluster
        self.client_data = {}      # Stores latest updates from clients
        self.num_clusters = 0
        

    # Called at the beginning of each round to select clients
    def client_selection(self, client_ids: List[str]) -> List[str]:
        print(f"[DEBUG] Selecting all {len(client_ids)} clients for the round.")
        return client_ids

    
    @staticmethod
    def flatten_model(model):
        return np.concatenate([p.flatten() for p in model])

    @staticmethod
    def extract_metadata_features(metadata):
        keys = sorted(metadata.keys())
        return np.array([
            float(metadata[k])
            for k in keys
            if isinstance(metadata[k], (int, float))
        ])
    
    # add an idea of having a parameter that says which rolling mean to use
    # like a parameter that dictates/says "use the rolling mean of the next 14 days"
    def aggregate(self, previous_global, client_updates):
        try:
            client_ids = list(client_updates.keys())
            cluster_clients = {}

            # Extract driving behavior features from metadata for clustering
            for cid in client_ids:
                metadata = client_updates[cid][1]
                
                # Extract driving behavior features
                driving_features = {
                    "rms_current_1h": metadata.get("rms_current_1h_avg", 0),
                    "rms_current_1d": metadata.get("rms_current_1d_avg", 0),
                    "max_acceleration": metadata.get("max_acceleration_avg", 0),
                    "avg_speed": metadata.get("avg_speed_avg", 0),
                    "driving_aggressiveness": metadata.get("driving_aggressiveness_avg", 0),
                    "battery_stress": metadata.get("battery_stress_avg", 0),
                    "temperature": metadata.get("ambairtemp", 25)  # Keep temperature as a feature
                }
                
                # Calculate a driving behavior score for clustering
                # This combines multiple features into a single score
                driving_score = (
                    driving_features["rms_current_1h"] * 0.2 +
                    driving_features["rms_current_1d"] * 0.15 +
                    driving_features["max_acceleration"] * 0.15 +
                    driving_features["avg_speed"] * 0.1 +
                    driving_features["driving_aggressiveness"] * 0.2 +
                    driving_features["battery_stress"] * 0.1 +
                    (driving_features["temperature"] - 25) * 0.1  # Normalize temperature around 25°C
                )
                
                print(f"[DEBUG] ===== Driving behavior score for client {cid}: {driving_score:.2f}")
                
                # Rule-based clustering by driving behavior
                if driving_score > 0.7:
                    behavior_zone = "aggressive"
                elif driving_score < 0.3:
                    behavior_zone = "conservative"
                else:
                    behavior_zone = "moderate"
                
                # Also consider temperature for more nuanced clustering
                temp = driving_features["temperature"]
                if temp > 35:
                    temp_zone = "hot"
                elif temp < 15:
                    temp_zone = "cold"
                else:
                    temp_zone = "moderate"
                
                # Combine behavior and temperature for final cluster
                cluster_id = f"{behavior_zone}_{temp_zone}"
                self.client_clusters[cid] = cluster_id
                cluster_clients.setdefault(cluster_id, []).append(cid)

            # --- Cluster-wise FedAvg ---
            new_cluster_models = {}
            for cluster_id, clients in cluster_clients.items():
                try:
                    total_weight = sum(client_updates[cid][1].get("num_examples", 1) for cid in clients)
                    aggregated_model = [np.zeros_like(param) for param in previous_global]

                    for cid in clients:
                        weight = client_updates[cid][1].get("num_examples", 1) / total_weight
                        for i, param in enumerate(client_updates[cid][0]):
                            aggregated_model[i] += param * weight

                    new_cluster_models[cluster_id] = aggregated_model

                except Exception as e:
                    print(f"[ERROR] ===== Failed aggregating cluster {cluster_id}: {e}, line 83")
                    raise

            self.cluster_models = new_cluster_models

            # Logging cluster summary
            print(f"✅ Created {len(new_cluster_models)} behavior-temperature clusters.")
            for cluster_id, clients in cluster_clients.items():
                print(f" - Cluster '{cluster_id}' has clients: {clients}")

            # Return largest cluster model (fallback model)
            largest_cluster = max(new_cluster_models.keys(), key=lambda x: len(cluster_clients[x]))
            return new_cluster_models[largest_cluster]

        except Exception as e:
            print(f"[ERROR] ===== Failed in aggregate(): {e}")
            print(f"[ERROR] ===== ")
            raise



def client_settings(self, global_model):
    settings = {}

    # If no global model is available, return safe fallback settings.
    if global_model is None:
        print("[WARNING] No global model available yet — likely no session started.")
        # Return safe fallback for all clients.
        return {client_id: {"learning_rate": self.lr} for client_id in self.client_clusters}

    # Optional learning rate decay logic.
    if self.round % 10 == 0:
        self.lr *= 0.1

    # Prepare common data orchestration parameters.
    # For example, you can calculate a window offset: e.g., every round covers a new segment of data.
    # Assume each round should shift the window by one day, or define a policy: e.g., "use the next 14-day segment."
    data_orch_params = {
        "window_offset": self.round,      # Example: current round number as a shift in days
        "window_length": 14,              # 14 days rolling window
        "orchestrate": True               # A flag to indicate that data orchestration is active
    }

    # At round 0, send seed parameters; otherwise, send cluster-specific models and data instructions.
    if self.round == 0:
        print("[INFO] Sending seed model to all clients.")
        for client_id in self.client_clusters:
            settings[client_id] = {
                "in_model_path": "seed.npz",
                "learning_rate": self.lr,
                "data_orch_params": data_orch_params  # Include our orchestration parameters
            }
    else:
        print(f"[INFO] Sending cluster-specific models, round {self.round}")
        for client_id, cluster_id in self.client_clusters.items():
            model = self.cluster_models.get(cluster_id)
            settings[client_id] = {
                "model": model if model is not None else global_model,
                "learning_rate": self.lr,
                "data_orch_params": data_orch_params  # Include orchestration params for each client
            }
    self.round += 1
    return settings



    