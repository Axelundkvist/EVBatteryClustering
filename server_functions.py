# --- Helper functions for hybrid clustering ---

import logging
from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random
from fedn.common.log_config import logger


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
        self.silhouette_threshold = 0.8  # ε from the paper

        # adding from Niclas script
        self.mode = "vanilla"
        self.mode_schedule = {
            0:  "log",
            1:  "train",
            10: "validate",
            # add more overrides here
        }
        

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

        # At round 0, send seed parameters; otherwise, send cluster-specific models and data instructions.
        if self.round == 0:
            print("[INFO] Sending seed model to all clients.")
            #print("check")
            for client_id in self.client_clusters:
                settings[client_id] = {
                    "model":           global_model,
                    "in_model_path": "seed.npz",
                    "learning_rate": self.lr, 
                    "mode": self.mode
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
                    "learning_rate": self.lr,
                    "mode": self.mode
                }
        self.round += 1
        print(f"self.client_clusters : {self.client_clusters}")
        
        print(list(settings.keys()))
        print(type(settings))
        return settings


    # här är den riktiga clusteringsalgoritmen
    def aggregate(self, previous_global: List[np.ndarray],
                  client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        # 1. Extract and flatten client parameter updates into a data matrix X
        print(f"Function Activated! running aggregatate function")
        print(f"round = {self.round}")
        
        
        print("******"*10)
        #print(f"client_updates: {client_updates}")
        client_ids = list(client_updates.keys())
        #print(f"client_ids : {client_ids}")
        
        # fråga till gpt: vad betyder X? 
        X = []
        weights = []
        # för varje client
        for i ,cid in enumerate(client_updates):

            # print(f"----"*10)
            # print(f"cid : {cid}")
            params, meta = client_updates[cid]
            
            # print(f"params = {params}")
            # print(f"meta = {meta}")
            
            # print(f"----"*10)
            # 1) flatten the model weights using your helper
            
            
            '''this way you only take a subset of the weights layer,
            you also achieve the private layer update part
            '''
            last_W = params[-2].ravel()   # assuming params[-2] is fc3.weight
            last_b = params[-1].ravel()   # assuming params[-1] is fc3.bias
            weight_vec = np.concatenate([last_W, last_b])

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
        silhouette_score = self._silhouette_scores(X, labels_km)
        print(f"{silhouette_score}")
        print(f"mean silhouette_score after K_means : {np.mean(silhouette_score)}")
        # 2) silhouette _on those_:
        if np.mean(self._silhouette_scores(X, labels_km)) < self.silhouette_threshold:
            # fallback to AP
            print(f" activating AP as the silhouette score was to low: {np.mean(silhouette_score)}")
            labels_ap, centers_ap = self._affinity_propagation(X)
            sil_ap   = self._silhouette_scores(X, labels_ap)
            mean_ap  = np.mean(sil_ap)
            print(f"silhouette scores for AP: {sil_ap}")
            print(f"mean silhouette_score after AP: {mean_ap:.4f}")
            
            labels, centers = labels_ap, centers_ap
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
        print(f"X.shape : {X.shape}")
        N = len(X)
        # similarity matrix
        S = -np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        R = np.zeros((N, N))
        A = np.zeros((N, N))
        for _ in range(50):  # max iterations
            # update responsibilities
            for i in range(N):
                for j in range(N):
                    mask = np.arange(N) != j
                    R[i, j] = S[i, j] - np.max(A[i, mask] + S[i, mask])
            # update availabilities
            for i in range(N):
                for j in range(N):
                    if i == j:
                        # diagonal case stays the same
                        A[j, j] = np.sum(np.maximum(0, R[:, j])) - R[j, j]
                    else:
                        # off‐diagonal: slice out row i
                        mask = np.arange(N) != i
                        avail = R[j, j] + np.sum(np.maximum(0, R[mask, j]))
                        A[i, j] = min(0, avail)

        # exemplars where R + A > 0
        exemplars = [j for j in range(N) if R[j, j] + A[j, j] > 0]
        # assign each point to nearest exemplar
        labels = [int(np.argmin([np.linalg.norm(X[i] - X[e]) for e in exemplars])) for i in range(N)]
        centers = [X[e] for e in exemplars]
        return labels, centers

#debugging purposes at the start of aggregate function
'''vec = self._flatten(params)         # flatten list of arrays → 1D vector
            weights.append(meta.get("num_examples", 1))
            
            
            recent = meta.get("recent_stats", {})
            # assume recent is a dict of numeric values; pick a consistent key order
            stat_keys = sorted(recent.keys())
            stat_vec = np.array([recent[k] for k in stat_keys], dtype=float)  # → shape (M,)
            
 
            # Log shapes of the incoming parameter arrays
            shapes = [p.shape for p in params]
            X.append(vec)
            logger.info(f"Client {cid} sent {len(params)} arrays, shapes: {shapes}")

            # Dump the metadata dict (this is where your recent_stats will live)
            logger.info(f"Metadata for {cid}: {meta}")

            # If you just want to see your recent_stats entry:
            stats = meta.get("recent_stats", None)
            logger.info(f"recent_stats for {cid}: {stats}")
'''
    