
class WaveletClusteringModel(nn.Module):
    def __init__(self, scale=50, wavelet='db2', threshold=0.5):
        super(WaveletClusteringModel, self).__init__()  # 调用父类的构造函数
        self.scale = scale
        self.wavelet = wavelet
        self.threshold = threshold

    def forward(self, data):
        """Wavelet Clustering with Multi-scale"""
        # Step 1: Normalize data
        normData = self.scale_01_data(data)

        # Step 2: Map data to scale domain
        dataDic = self.map2ScaleDomain(normData, self.scale)

        # Step 3: Perform wavelet transform
        dwtResult = self.ndWT(dataDic, normData.shape[1], self.scale, self.wavelet)

        # Step 4: Thresholding and clustering
        result = self.thresholding(dwtResult, self.threshold, self.scale, normData.shape[1])

        # Step 5: Return the results (centroids and labels)
        return result


    def scale_01_data(self, rawData):
        """归一化数据"""
        # 如果 rawData 是 PyTorch Tensor，则转换为 NumPy 数组
        if isinstance(rawData, torch.Tensor):
            rawData = rawData.detach().cpu().numpy()

        dim = rawData.shape[1]
        minList = [np.amin(rawData[:, x]) for x in range(dim)]
        maxList = [np.amax(rawData[:, x]) + 0.001 for x in range(dim)]
        toZero = rawData - np.array(minList)
        normData = toZero / (np.array(maxList) - np.array(minList))
        return normData

    def map2ScaleDomain(self, dataset, scale=128):
        """Map data to scale domain"""
        if scale <= 0 or not isinstance(scale, int):
            raise ValueError('scale must be a positive integer')
        dim = dataset.shape[1]
        length = dataset.shape[0]
        sd_data = {}
        for i in range(0, length):
            num = 0
            for j in reversed(range(0, dim)):
                num += (dataset[i, j] // (1 / scale)) * pow(scale, j)
            num = int(num)
            if sd_data.get(num, 'N/A') == 'N/A':
                sd_data[num] = 1
            else:
                sd_data[num] += 1
        return sd_data

    def ndWT(self, data, dim, scale, wave):
        """Perform n-dimensional wavelet transform"""
        wavelets = {'db1': [0.707, 0.707], 'bior1.3': [-0.09, 0.09, 0.707, 0.707, 0.09, -0.09],
                    'db2': [-0.13, 0.224, 0.836, 0.483]}
        lowFreq = {}
        convolutionLen = len(wavelets.get(wave)) - 1
        lineLen = ceil(scale / 2) + ceil((convolutionLen - 2) / 2)

        for inDim in range(0, dim):
            for key in data.keys():
                coordinate = []
                tempkey = key
                for i in range(0, dim):
                    if i <= dim - inDim - 1:
                        coordinate.append(tempkey // pow(scale, (dim - 1 - i)))
                        tempkey = tempkey % pow(scale, (dim - 1 - i))
                    else:
                        coordinate.append(tempkey // pow(lineLen, (dim - 1 - i)))
                        tempkey = tempkey % pow(lineLen, (dim - 1 - i))
                coordinate.reverse()
                startCoord = ceil((coordinate[inDim] + 1) / 2) - 1
                startNum = 0
                for i in range(0, dim):
                    if i <= inDim:
                        if i == inDim:
                            startNum += startCoord * pow(lineLen, i)
                        else:
                            startNum += coordinate[i] * pow(lineLen, i)
                    else:
                        startNum += coordinate[i] * pow(scale, i)
                wavelet = wavelets.get(wave)
                for i in range(0, convolutionLen // 2 + 1):
                    if startCoord + i >= lineLen:
                        break
                    if lowFreq.get(int(startNum + pow(lineLen, inDim) * i), 'N/A') == 'N/A':
                        lowFreq[int(startNum + pow(lineLen, inDim) * i)] = data[key] * wavelet[
                            int((startCoord + 1 + i) * 2 - (coordinate[inDim] + 1))]
                    else:
                        lowFreq[int(startNum + pow(lineLen, inDim) * i)] += data[key] * wavelet[
                            int((startCoord + 1 + i) * 2 - (coordinate[inDim] + 1))]
            data = lowFreq
            lowFreq = {}
        return data

    def thresholding(self, data, threshold, scale, dim):
        """Thresholding and clustering"""

        nodes = {}
        result = {}
        startNode = node(0)
        avg = 0
        if len(nodes) == 0:
            print("Warning: nodes is empty. Skipping division to avoid ZeroDivisionError.")
            return 0  # 或者返回一个合理的默认值
        for key, value in data.items():
            if value >= threshold:
                nodes[key] = node(key, value)
                avg += value
                if value > startNode.value:
                    startNode = node(key, value)
        cutMiniCluster = avg / len(nodes)
        clusters = self.clustering(nodes, scale, dim, cutMiniCluster)
        return clusters

    def clustering(self, data, scale, dim, cutMiniCluster):
        """Perform clustering"""
        equal_pair = []
        cluster_flag = 1
        for point in data.values():
            point.process = True
            for around in point.around(scale, dim):
                if not (data.get(around, 'N/A') == 'N/A'):
                    around = data.get(around)
                    if around.cluster is not None:
                        if point.cluster is None:
                            point.cluster = around.cluster
                        elif point.cluster != around.cluster:
                            mincluster = min(point.cluster, around.cluster)
                            maxcluster = max(point.cluster, around.cluster)
                            equal_pair += [(mincluster, maxcluster)]
            if point.cluster is None:
                point.cluster = cluster_flag
                cluster_flag += 1
        equal_pair = set(equal_pair)
        equal_list =self.bfs(equal_pair, cluster_flag)
        result = self.build_key_cluster(data, equal_list, cutMiniCluster)
        return result

    def bfs(self, equal_pair, maxQueue):
        if equal_pair == []:
            return (equal_pair)
        group = {x: [] for x in range(1, maxQueue)}
        result = []
        for x, y in equal_pair:
            group[x].append(y)
            group[y].append(x)
        for i in range(1, maxQueue):
            if i in group:
                if group[i] == []:
                    del group[i]
                else:
                    queue = [i]
                    for j in queue:
                        if j in group:
                            queue += group[j]
                            del group[j]
                    record = list(set(queue))
                    record.sort()
                    result.append(record)
        return (result)

    def build_key_cluster(self, nodes, equal_list, cutMiniCluster):
        cluster_key = {}
        for point in nodes.values():
            flag = 0
            for cluster in equal_list:
                if point.cluster in cluster:
                    point.cluster = cluster[0]
                    if cluster_key.get(cluster[0], 'N/A') == 'N/A':
                        cluster_key[cluster[0]] = [point]
                        flag = 1
                    else:
                        cluster_key[cluster[0]].append(point)
                        flag = 1
                    break
            if flag == 0:
                if cluster_key.get(point.cluster, 'N/A') == 'N/A':
                    cluster_key[point.cluster] = [point]
                else:
                    cluster_key[point.cluster].append(point)
        count = 1
        result = {}
        # ipdb.set_trace()
        for cluster in cluster_key.keys():
            # ipdb.set_trace()
            if len(cluster_key[cluster]) == 1:
                if cluster_key[cluster][0].value < cutMiniCluster:
                    continue
            for p in cluster_key[cluster]:
                result[p.key] = count
            count += 1
        return (result)


    def thresholding(self, data, threshold, scale, dim):
        """Apply Threshold to the Data and Perform Clustering"""
        nodes = {}
        avg = 0
        for key, value in data.items():
            if value >= threshold:
                nodes[key] = node(key, value)
                avg += value
        cutMiniCluster = avg / len(nodes)
        clusters = self.clustering(nodes, scale, dim, cutMiniCluster)
        return clusters

    def markData(self, normData, cluster, scale):
        """Mark Data According to the Clusters"""
        tags = []
        for point in range(normData.shape[0]):
            number = sum((normData[point, inDim] // (1 / scale)) * (scale ** inDim) for inDim in range(normData.shape[1]))
            tags.append(cluster.get(int(number), 0))
        return tags
