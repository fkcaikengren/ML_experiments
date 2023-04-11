import numpy as np

class KMeans:
    def __init__(self, data, n_clusters,  max_iter=1, random_state=None):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        self.inertia = 0
        self.n_samples = data.shape[0]
        self.n_features = data.shape[1]


    def train(self):
        # 1.随机初始化簇中心
        rng =  np.random.default_rng(seed=self.random_state) 
        data_index = rng.permutation(self.n_samples)
        self.centroids = self.data[data_index[:self.n_clusters]] #从data中随机取n_clusters行

        # print(self.centroids)
        for _ in range(self.max_iter):
            # 对样本划分簇
            closest_centroid_ids, inertia = KMeans.find_closest_centroids(self.data, self.centroids)
            if inertia == self.inertia:
                break
            else:
                self.inertia = inertia
            # print(closest_centroid_ids)
            # 计算簇中心
            self.centroids = KMeans.compute_centroids(self.data, closest_centroid_ids, self.n_clusters)
            # print(self.inertia)
        return self.centroids,closest_centroid_ids

    def predict(self, data):
        closest_centroid_ids,_ = KMeans.find_closest_centroids(data, self.centroids)
        return closest_centroid_ids

    @staticmethod
    def find_closest_centroids(data, centroids):
        n_samples = data.shape[0]
        closest_centroid_ids = np.zeros(n_samples,dtype=int)
        inertia = 0 
        for i, cur_data in enumerate(data):
            min_dist = np.inf
             # 计算距离该样本最近的簇中心索引
            for j,centroid in enumerate(centroids):
                cur_dist = np.sum((cur_data - centroid)**2)
                if cur_dist<=min_dist:
                    min_dist = cur_dist
                    closest_centroid_ids[i] = j
            inertia = inertia+min_dist
        return closest_centroid_ids,inertia
    
    
    @staticmethod
    def compute_centroids(data, closest_centroid_ids, n_clusters):
        centroids = np.zeros((n_clusters, data.shape[1]))
        for idx  in range(n_clusters): 
            cluster_data = data[closest_centroid_ids == idx] #过滤得到当前簇的样本点
            centroids[idx] = np.mean(cluster_data, axis=0)
        return centroids
    

