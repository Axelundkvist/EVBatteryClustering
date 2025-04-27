# --- Helper functions for hybrid clustering ---

import logging
from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random
from fedn.common.log_config import logger


## dessa kan du inte ta med
#from sklearn.cluster import AffinityPropagation
#from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# --- Helper functions for hybrid clustering ---


from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        # state across rounds
        self.round = 0
        self.lr = 0.1
        self.cluster_labels = None       # type: List[int]
        
        self.cluster_models = None
        self.client_clusters = {} # client_id → cluster_id
        
        self.cluster_centers = None      # type: List[np.ndarray]
        self.initial_K = 3               # pick some reasonable default
        self.silhouette_threshold = 0.5  # ε from the paper

    def client_selection(self, client_ids: List[str]) -> List[str]:
        print(f"[DEBUG] Selecting all {len(client_ids)} clients for the round.")
        return client_ids


    def client_settings(self, global_model):
        settings = {}
        print(f"Function Activated! running client_settings function")
        print(f"round = {self.round}")
        
        
        '''dehär vill du nog inte ha, om rundan är 0 så vill du ju skicka ut din initial seed model och sen 
        låta alla clienter (EV bilar) köra på sin data först
        +
        du kan även tänka att det blir skumt och köra clustering på dina clienter om de inte har hunnit träna något heller
        '''
        # # If no global model is available, return safe fallback settings.
        # if global_model is None:
        #     print("[WARNING] No global model available yet — likely no session started.")
        #     # Return safe fallback for all clients.
        #     return {client_id: {"learning_rate": self.lr} for client_id in self.client_clusters}

        # At round 0, send seed parameters; otherwise, send cluster-specific models and data instructions.
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
                    # here is the asigned cluster model
                    "model": model if model is not None else global_model,
                    
                    # updated learning rate which as of now is not changed
                    # see if you can delete learning rate
                    "learning_rate": self.lr
                }
        self.round += 1
        return settings


    # här är den riktiga clusteringsalgoritmen
    def aggregate(self, previous_global: List[np.ndarray],
                  client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        # 1. Extract and flatten client parameter updates into a data matrix X
        print(f"Function Activated! running aggregatate function")
        print(f"round = {self.round}")
        
        client_ids = list(client_updates.keys())
        print(f"client_ids = {client_ids}")
        
        # fråga till gpt: vad betyder X? 
        X = []
        weights = []
        # för varje client
        for cid in client_ids:
            # fråga till gpt: params är datan från clienten?
            # fråga till gpt: och så använder vi meta data som innehåller num_examples 
            #                 ett sätt och disponera dess påverkan i clustermodellen senare?
            # följdfråga till gpt: äre detta som gör att vi får 
            
            params, meta = client_updates[cid]
            print(f"params = {params}")
            print(f"meta = {meta}")
            
            # 1) flatten the model weights using your helper
            weight_vec = self._flatten(params)  # → shape (D,)

            # 2) extract & vectorize your recent_stats
            recent = meta.get("recent_stats", {})
            if isinstance(recent, dict):
                # pick a consistent key order
                stat_keys = sorted(recent.keys())
                stat_vec = np.array([recent[k] for k in stat_keys], dtype=float)  # → shape (M,)
            else:
                # if it comes as a list/array
                stat_vec = np.array(recent, dtype=float).ravel()                    

            # 3) concatenate weights + stats
            feature_vec = np.concatenate([weight_vec, stat_vec])  # → shape (D+M,)

            X.append(feature_vec)
            weights.append(meta.get("num_examples", 1))

            # optional debug
            print(f"[DEBUG] {cid}: weights {weight_vec.shape}, stats {stat_vec.shape}, combined {feature_vec.shape}")

            
        # 
        X = np.stack(X)                         # shape (N_clients, D)
        total_weight = sum(weights)

        # 2. INITIAL CLUSTERING or SILHOUETTE STABILITY CHECK
        # choose K for this round
        if self.round == 0 or self.cluster_centers is None:
            K = self.initial_K
        else:
            K = len(self.cluster_centers)

        # 1) fresh K-means
        labels_km, centers_km = self._kmeans(X, K)

        # 2) silhouette _on those_:
        if np.mean(self._silhouette_scores(X, labels_km)) < self.silhouette_threshold:
            # fallback to AP
            labels, centers = self._affinity_propagation(X)
        else:
            labels, centers = labels_km, centers_km

        # save everything
        self.cluster_labels  = labels
        self.cluster_centers = centers
        self.client_clusters = {cid: lbl for cid, lbl in zip(client_ids, labels)}
        self.initial_K       = len(centers)   # next round’s K
                

        # 3. PER-CLUSTER FEDAVG
        cluster_models = {}
        cluster_weights = {}
        for idx, cid in enumerate(client_ids):
            
            lbl = labels[idx]
            params, _ = client_updates[cid]
            w = weights[idx]
            # här checkar vi om labeln för cluster x finns
            # om inte, börja skapa ett tomt sklett för cluster x
            if lbl not in cluster_models:
                # initialize accumulator
                cluster_models[lbl] = [np.zeros_like(p) for p in params]
                cluster_weights[lbl] = 0
            # accumulate weighted parameters
            for i, p in enumerate(params):
                cluster_models[lbl][i] += p * w
            cluster_weights[lbl] += w

        # finalize per-cluster models
        for lbl in cluster_models:
            cw = cluster_weights[lbl]
            if cw == 0:
                continue
            else:
                cluster_models[lbl] = [p / cw for p in cluster_models[lbl]]
            
            
        # nu kommer cluster modellerna sparas
        # borde du spara 
        self.cluster_models = cluster_models

        # 4. GLOBAL AVERAGE OF CLUSTER MODELS (weighted by cluster size)
        #    i.e. one more FedAvg over the cluster‐level models
        global_model = [np.zeros_like(p) for p in previous_global]
        for lbl, cm in cluster_models.items():
            cw = cluster_weights[lbl]
            for i, p in enumerate(cm):
                global_model[i] += p * cw
        global_model = [p / total_weight for p in global_model]

        #self.round += 1
        return global_model

    # ─────────── Helper methods ───────────

    def _flatten(self, params: List[np.ndarray]) -> np.ndarray:
        """Concatenate and flatten a list of arrays into one long vector."""
        return np.concatenate([p.ravel() for p in params])

    def _kmeans(self, X: np.ndarray, K: int):
        """Simple K-means (initialize with random points)."""
        # init centroids
        idx = random.sample(range(len(X)), K)
        centroids = X[idx]
        labels = [0] * len(X)
        for _ in range(10):  # fixed small number of iterations
            # assign
            for i, x in enumerate(X):
                dists = np.linalg.norm(centroids - x, axis=1)
                labels[i] = int(np.argmin(dists))
            # update
            for k in range(K):
                members = X[np.array(labels) == k]
                if len(members) > 0:
                    centroids[k] = np.mean(members, axis=0)
        return labels, centroids

    def _silhouette_scores(self, X: np.ndarray, labels: List[int]) -> List[float]:
        """Compute silhouette score per sample."""
        n = len(X)
        scores = [0.0] * n
        for i in range(n):
            own = labels[i]
            # intra‐cluster
            same = X[np.array(labels) == own]
            a = np.mean(np.linalg.norm(same - X[i], axis=1)) if len(same) > 1 else 0
            # nearest other cluster
            bs = []
            for other in set(labels):
                if other == own: continue
                group = X[np.array(labels) == other]
                bs.append(np.mean(np.linalg.norm(group - X[i], axis=1)))
            b = min(bs) if bs else 0
            scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
        return scores

    def _affinity_propagation(self, X: np.ndarray):
        """Pseudocode for Affinity Propagation (messages R, A)."""
        N = len(X)
        # similarity matrix
        S = -np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        R = np.zeros((N, N))
        A = np.zeros((N, N))
        for _ in range(50):  # max iterations
            # update responsibilities
            for i in range(N):
                for j in range(N):
                    R[i, j] = S[i, j] - np.max(A[i, :] + S[i, :], where=np.arange(N) != j)
            # update availabilities
            for i in range(N):
                for j in range(N):
                    if i == j:
                        A[j, j] = np.sum(np.maximum(0, R[:, j])) - R[j, j]
                    else:
                        A[i, j] = min(0, R[j, j] + np.sum(np.maximum(0, R[:, j]), where=np.arange(N) != i))
        # exemplars where R + A > 0
        exemplars = [j for j in range(N) if R[j, j] + A[j, j] > 0]
        # assign each point to nearest exemplar
        labels = [int(np.argmin([np.linalg.norm(X[i] - X[e]) for e in exemplars])) for i in range(N)]
        centers = [X[e] for e in exemplars]
        return labels, centers

#debugging purposes at the start of aggregate function
'''vec = self._flatten(params)         # flatten list of arrays → 1D vector
            X.append(vec)
            weights.append(meta.get("num_examples", 1))
            
            
            recent = meta.get("recent_stats", {})
            # assume recent is a dict of numeric values; pick a consistent key order
            stat_keys = sorted(recent.keys())
            stat_vec = np.array([recent[k] for k in stat_keys], dtype=float)  # → shape (M,)
            
 
            # Log shapes of the incoming parameter arrays
            shapes = [p.shape for p in params]
            logger.info(f"Client {cid} sent {len(params)} arrays, shapes: {shapes}")

            # Dump the metadata dict (this is where your recent_stats will live)
            logger.info(f"Metadata for {cid}: {meta}")

            # If you just want to see your recent_stats entry:
            stats = meta.get("recent_stats", None)
            logger.info(f"recent_stats for {cid}: {stats}")
'''
    



'''class ServerFunctions(ServerFunctionsBase):
    def __init__(self):
        self.client_clusters = {}  # Dictionary to track client-cluster assignments
        self.cluster_models = {}   # Stores models for each cluster
        self.client_data = {}      # Stores latest updates from clients
        self.num_clusters = 0
        self.client_cluster_history = {}  # Dictionary to track client cluster history across rounds
        self.current_round = 0  # Track the current round number
        

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
    
    #this is also creating new clusters for each round isn't it?
    
    # add an idea of having a parameter that says which rolling mean to use
    # like a parameter that dictates/says "use the rolling mean of the next 14 days"
    def aggregate(self, previous_global, client_updates):
        try:
            logger.info("--------------------------------")
            logger.info("[Success] ===== starting aggregate in ServerFunctions.py")
            logger.info("--------------------------------")
            client_ids = list(client_updates.keys())
            cluster_clients = {}
            
            # Increment round counter
            self.current_round += 1
            print(f"[DEBUG] ===== Starting round {self.current_round} =====")

            # Extract driving behavior features from metadata for clustering
            for cid in client_ids:
                try:
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
                    print(f"[DEBUG] ===== Client {cid} assigned to Cluster ID: {cluster_id}")
                    
                    # Update current cluster assignment
                    self.client_clusters[cid] = cluster_id
                    
                    # Update cluster history
                    if cid not in self.client_cluster_history:
                        self.client_cluster_history[cid] = {}
                    self.client_cluster_history[cid][self.current_round] = cluster_id
                    
                    # Add to cluster clients list
                    cluster_clients.setdefault(cluster_id, []).append(cid)
                    
                except Exception as e:
                    print(f"[ERROR] ===== Failed to extract driving behavior features for client {cid}: {e}")
                    continue

            # --- Cluster-wise FedAvg with model persistence ---
            # Define a decay factor for historical models (how much to weight previous models)
            # This can be adjusted based on how much you want to preserve historical learning
            historical_weight = 0.3  # 30% weight to historical models, 70% to new updates
            
            # Initialize new cluster models dictionary
            new_cluster_models = {}
            
            # Process each cluster
            for cluster_id, clients in cluster_clients.items():
                try:
                    # Calculate total weight for this cluster's clients
                    total_weight = sum(client_updates[cid][1].get("num_examples", 1) for cid in clients)
                    
                    # Initialize aggregated model for this round
                    aggregated_model = [np.zeros_like(param) for param in previous_global]
                    
                    # Aggregate client updates for this cluster
                    for cid in clients:
                        weight = client_updates[cid][1].get("num_examples", 1) / total_weight
                        for i, param in enumerate(client_updates[cid][0]):
                            aggregated_model[i] += param * weight
                    
                    # Check if we have a historical model for this cluster
                    if cluster_id in self.cluster_models:
                        print(f"[DEBUG] ===== Updating existing model for cluster {cluster_id}")
                        # Combine historical model with new aggregated model
                        historical_model = self.cluster_models[cluster_id]
                        combined_model = []
                        
                        for i, (hist_param, new_param) in enumerate(zip(historical_model, aggregated_model)):
                            # Weighted combination of historical and new parameters
                            combined_param = hist_param * historical_weight + new_param * (1 - historical_weight)
                            combined_model.append(combined_param)
                        
                        new_cluster_models[cluster_id] = combined_model
                    else:
                        print(f"[DEBUG] ===== Creating new model for cluster {cluster_id}")
                        # If no historical model exists, just use the new aggregated model
                        new_cluster_models[cluster_id] = aggregated_model
                        
                except Exception as e:
                    print(f"[ERROR] ===== Failed aggregating cluster {cluster_id}: {e}")
                    raise

            # Update the cluster models dictionary with the new models
            self.cluster_models = new_cluster_models

            # Logging cluster summary
            print(f"✅ Created/Updated {len(new_cluster_models)} behavior-temperature clusters.")
            for cluster_id, clients in cluster_clients.items():
                print(f" - Cluster '{cluster_id}' has clients: {clients}")
                
            # Log client cluster history
            print("\n[INFO] Client Cluster History:")
            for cid, history in self.client_cluster_history.items():
                history_str = ", ".join([f"Round {round_num}: {cluster}" for round_num, cluster in history.items()])
                print(f" - Client {cid}: {history_str}")

            # Return largest cluster model (fallback model)
            largest_cluster = max(new_cluster_models.keys(), key=lambda x: len(cluster_clients[x]))
            return new_cluster_models[largest_cluster]

        except Exception as e:
            print(f"[ERROR] ===== Failed in aggregate(): {e}")
            #print(f"[ERROR] ===== ")
            raise

    def get_client_cluster_history(self, client_id):
        """
        Retrieve the cluster history for a specific client.
        Args:
            client_id: The ID of the client
        Returns:
            A dictionary mapping round numbers to cluster IDs, or None if the client has no history
        """
        if client_id in self.client_cluster_history:
            return self.client_cluster_history[client_id]
        return None
        
    def get_all_clients_history(self):
        """
        Retrieve the cluster history for all clients.
        Returns:
            A dictionary mapping client IDs to their cluster history
        """
        return self.client_cluster_history

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
        "current_round": self.round,
        "window_offset": self.round,      # Example: current round number as a shift in days
        "orchestrate": True, 
        "number_of_cycles_to_compare": 10 # A flag to indicate that data orchestration is active
        #"window_length": 28,              # commenting our window lenght as that is arbitrary
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
'''


    