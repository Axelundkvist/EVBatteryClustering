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

        # not used as they are a part of unallowed packages imports
        #self.similarity_threshold = 0.2
        #self.last_silhouette_score = None
        

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
    
    
    def aggregate(self, previous_global, client_updates):
        try:
            client_ids = list(client_updates.keys())
            cluster_clients = {}

            for cid in client_ids:
                metadata = client_updates[cid][1]
                temp = metadata.get("ambairtemp", 25) # sets default value to 25 if not getting a value
                print(f"[DEBUG] ===== temp for client {cid}: {temp}")

                # Rule-based clustering by temperature zone
                if temp > 35:
                    temp_zone = "hot"
                elif temp < 15:
                    temp_zone = "cold"
                else:
                    temp_zone = "moderate"

                cluster_id = temp_zone
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
            print(f"✅ Created {len(new_cluster_models)} temperature-based clusters.")
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

    if global_model is None:
        print("[WARNING] No global model available yet — likely no session started.")
        # Return empty safe fallback
        return {client_id: {"learning_rate": self.lr} for client_id in self.client_clusters}



    if self.round % 10 == 0:
        self.lr *= 0.1

    if self.round == 0:
        print("[INFO] Sending seed model to all clients.")
        for client_id in self.client_clusters:
            settings[client_id] = {
                "in_model_path": "seed.npz",
                "learning_rate": self.lr
            }
    else:
        print(f"[INFO] Sending cluster-specific models, round {self.round}")
        for client_id, cluster_id in self.client_clusters.items():
            model = self.cluster_models.get(cluster_id)
            settings[client_id] = {
                "model": model if model is not None else global_model,
                "learning_rate": self.lr
            }

    self.round += 1
    return settings



    # def client_settings(self, global_model):
    #     settings = {}

    #     if self.round % 10 == 0:
    #         self.lr *= 0.1

    #     if self.round == 0:
    #         print("[INFO] Sending seed model to all clients.")
    #         for client_id in self.client_clusters:
    #             settings[client_id] = {
    #                 "in_model_path": "seed.npz",
    #                 "learning_rate": self.lr,
    #             }
    #     else:
    #         print(f"[INFO] Sending cluster-specific models, round {self.round}")
    #         for client_id in self.client_clusters:
    #             cluster_id = self.client_clusters.get(client_id)
    #             model = self.cluster_models.get(cluster_id) if self.cluster_models else None

    #             if model is None:
    #                 print(f"[WARNING] No cluster model found for client {client_id} (cluster_id={cluster_id}), sending global model instead.")

    #             settings[client_id] = {
    #                 "model": model if model is not None else global_model,
    #                 "learning_rate": self.lr,
    #             }

    #     self.round += 1
    #     return settings




    '''
    def Oldaggregate(self, previous_global, client_updates):
        client_ids = list(client_updates.keys())

        # --- Hybrid client vector: model weights + metadata ---
        client_vectors = [
            np.concatenate([
                self.flatten_model(client_updates[cid][0]),
                self.extract_metadata_features(client_updates[cid][1])
            ])
            for cid in client_ids
        ]
                
        # --- Affinity Propagation ---
        ap = AffinityPropagation(damping=0.75, preference=-50).fit(client_vectors)
        labels = ap.labels_

        # --- Silhouette score logging ---
        if len(set(labels)) > 1:
            self.last_silhouette_score = silhouette_score(client_vectors, labels)
            logger.info(f"Silhouette Score before merging: {self.last_silhouette_score:.4f}")
        else:
            self.last_silhouette_score = -1
            logger.info("Silhouette Score could not be computed (only one cluster).")

        # --- Cluster merging if centroids too close ---
        cluster_vectors = {}
        for idx, label in enumerate(labels):
            cluster_vectors.setdefault(label, []).append(client_vectors[idx])
        cluster_centroids = {label: np.mean(vecs, axis=0) for label, vecs in cluster_vectors.items()}

        label_map = {}
        new_label = 0
        used = set()
        labels_updated = labels.copy()

        for i, ci in enumerate(cluster_centroids):
            if ci in used:
                continue
            label_map[ci] = new_label
            for cj in cluster_centroids:
                if ci != cj and cj not in used:
                    dist = np.linalg.norm(cluster_centroids[ci] - cluster_centroids[cj])
                    if dist < self.similarity_threshold:
                        label_map[cj] = new_label
                        used.add(cj)
            used.add(ci)
            new_label += 1

        for idx, l in enumerate(labels):
            labels_updated[idx] = label_map[l]

        # --- Aggregate per-cluster models ---
        new_cluster_models = {}
        cluster_clients = {}
        for idx, cid in enumerate(client_ids):
            cluster_id = labels_updated[idx]
            self.client_clusters[cid] = cluster_id
            cluster_clients.setdefault(cluster_id, []).append(cid)

        for cluster_id, clients in cluster_clients.items():
            total_weight = sum(client_updates[cid][1].get("num_examples", 1) for cid in clients)
            aggregated_model = [np.zeros_like(param) for param in previous_global]

            for cid in clients:
                weight = client_updates[cid][1].get("num_examples", 1) / total_weight
                for i, param in enumerate(client_updates[cid][0]):
                    aggregated_model[i] += param * weight

            new_cluster_models[cluster_id] = aggregated_model

        self.cluster_models = new_cluster_models
        self.num_clusters = len(new_cluster_models)

        logger.info(f"Number of clusters after merging: {self.num_clusters}")
        for cluster_id, clients in cluster_clients.items():
            logger.info(f"Cluster {cluster_id} has clients: {clients}")

        # Return largest cluster model for fallback
        largest_cluster = max(new_cluster_models.keys(), key=lambda x: len(cluster_clients[x]))
        return new_cluster_models[largest_cluster]
'''



