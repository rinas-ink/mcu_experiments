from sklearn.cluster import KMeans


class PointClouds:
    def __init__(self, point_clouds, params, cloud_dists, cluster_cnt = None):
        self.n = point_clouds.shape[0]
        self.point_clouds = point_clouds.reshape(self.n, -1, 3)
        self.params = params
        self.cluster_cnt = cluster_cnt
        self.average_dists_in_clusters = []
        if cluster_cnt is None:
            self.cluster_cnt = int(self.n ** 0.5)

        kmeans = KMeans(n_clusters=self.cluster_cnt)
        clusters = kmeans.fit_predict(self.params)
        self.clustered_clouds = {i: [] for i in range(kmeans.n_clusters)}
        for pc_idx, cluster_idx in enumerate(clusters):
            self.clustered_clouds[cluster_idx].append(pc_idx)

        self.medoids = {}
        for cluster_idx, pc_indexes in self.clustered_clouds.items():
            min_distance_sum = float('inf')
            medoid = None
            avg_dist = None
            for i in pc_indexes:
                distance_sum = sum(cloud_dists[i, j] for j in pc_indexes)
                if distance_sum < min_distance_sum:
                    min_distance_sum = distance_sum
                    medoid = i
                    avg_dist = distance_sum / len(pc_indexes)
            self.medoids[cluster_idx] = medoid
            self.average_dists_in_clusters[cluster_idx] = avg_dist

    def get_k_neighbors(self, point_cloud, k, distance_func):
        medoid_dists = []
        for cluster_idx, medoid_idx in self.medoids.items():
            medoid_dists.append((distance_func(point_cloud, self.point_clouds[medoid_idx]), cluster_idx))
        medoid_dists.sort()
        closest_distance = medoid_dists[0][0]
        closest_cluster_indices = [medoid_dists[0][1]]
        threshold = 0.1
        for distance, cluster_idx in medoid_dists[1:]:
            if (distance - closest_distance) / closest_distance <= threshold:
                closest_cluster_indices.append(cluster_idx)
            else:
                break

        nearest_pcs = []
        for i in closest_cluster_indices:
            for pc_idx in self.clustered_clouds[i]:
                nearest_pcs.append((distance_func(point_cloud, self.point_clouds[pc_idx]), self.point_clouds[pc_idx]))
        nearest_pcs.sort()
        return nearest_pcs[:k]
