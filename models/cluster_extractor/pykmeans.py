import math


def category_hamming_dist(x, y):
    """Counts number of mismatches as distance between two lists"""

    dist = 0
    for i in range(0, len(x)):
        if x[i] != y[i]:
            dist += 1

    return dist


def category_hamming_centroid_factory(data):
    """Create a centroid from a list of categories

    The centroid is a data point for which
    sum(map(lambda x: category_hamming_dist(centroid, x),data)) is minimized
    """

    cat_dict = {}
    centroid = []
    dim = len(data[0])

    for i in range(0, dim):

        for dp in data:

            if dp[i] in cat_dict:
                cat_dict[dp[i]] += 1
            else:
                cat_dict[dp[i]] = 1

        centroid.append(max(cat_dict, key=cat_dict.get))

        cat_dict = {}

    return centroid


def category_hamming_medoid_factory(data):
    """Create a medoid from a list of categories"""

    distance_list = []

    # compute the distance from each dp to the other
    for candidate in data:
        dist_sum = 0
        for dp in data:
            dist_sum += category_hamming_dist(candidate, dp)
        distance_list.append(dist_sum)

    return data[distance_list.index(min(distance_list))]




def _popcnt(v):
    c = 0
    while v:
        v &= v - 1
        c += 1
    return c


def _bit_length(x):
    return len(bin(x)) - 2;


try:
    int(0).bit_length()
    _bit_length = lambda x: x.bit_length()
except:
    pass


def hamming_dist(x, y):
    """Return the hamming distance of the two numbers"""
    return _popcnt(x ^ y)


def hamming_centroid_factory(data):
    """Create a centroid from a list of numbers
	
	The centroid is a number for which
	sum(map(lambda x: hamming_dist(centroid,x),data)) is minimized.
	"""
    l = max((_bit_length(d) for d in data))
    buckets = [0] * l
    bits = map(lambda i: 1 << i, range(l))
    for d in data:
        for i, two_to_the_i in enumerate(bits):
            if d & two_to_the_i:
                buckets[i] += 1
            else:
                buckets[i] -= 1
    return reduce(
        lambda accum, x: accum | (x[1] > 0) * bits[x[0]],
        enumerate(buckets),
        0
    )


def euclid_dist(x, y):
    """Return the euclidean distance between two lists of numbers"""
    return sum((math.pow(x[i] - y[i], 2) for i in range(len(x))))


def euclid_centroid_factory(data):
    """Return the geometric center of a list of list of numbers"""
    dim = len(data[0])
    n = len(data)
    sums = [sum((data[j][i] for j in range(n))) for i in range(dim)]
    return [float(s) / n for s in sums]


def kmeans(data, centroids, distf, centroidf, cutoff):
    """Apply the kmeans clustering algorithm
	
	Parameters
	
	data		is any type of data
	centroids	is a number of centroids compatible to the data.
				ex.: if the data are vectors in 2 dimensions the such
				should also be the centroids. The number of the
				centroids implies the number of the clusters we want.
	distf		a function that takes any two items from the data list
				and returns a comparable measure of distance to use for
				the clustering
	centroidf	a function that is used to update the new centroids. it
				receives a list of data items and should return a data
				item that minimizes the sum(map(lambda x: distf(x,centroid),data))
	cutoff		a value used to stop iterating. Its type is the return
				type of distf. When the maximum centroid change is less
				than the cutoff then we stop iterating.
	
	Returns
	
	centroids			the centroids list updated to be the clusters' centroids.
	data_to_centroids	the centroid that each data item belongs to.
						This is a list that for each position i the value
						v is the cluster index that the data[i] belongs to
	distances			the distances of each data item to each centroid
						as a list of lists
	
	
	"""
    k = len(centroids)
    while True:
        """
        distances = [map(lambda x: distf(x, y), centroids) for y in data]
        """

        centroid_buckets = []
        corresponding_centroid = []
        distances = []

        for i in range(len(centroids)):
            centroid_buckets.append([])

        # assign each datapoint to the closest centroid where centroid_buckets[centroid_index] = [datapoint, ...]
        for dp in data:
            dp_dist = []
            for ct in centroids:
                dp_dist.append(distf(dp, ct))
            centroid_buckets[dp_dist.index(min(dp_dist))].append(dp)
            corresponding_centroid.append(dp_dist.index(min(dp_dist)))
            distances.append(dp_dist)

        # compute centroids
        new_centroids = []
        for bucket in centroid_buckets:
            new_centroids.append(centroidf(bucket))

        """
        new_centroids = map(
            centroidf,  # return a centroid given a list of points
            [
                map(
                    lambda x: data[x[0]],  # get the actual data point
                    filter(
                        lambda x: x[1] == y,
                        enumerate(data_to_centroids)
                    )  # get the points of cluster y as tuples (data_idx,cluster)
                )
                for y in range(k)  # for each temporary cluster
            ]
        )
        """

        changes = [distf(list(new_centroids)[i], list(centroids)[i]) for i in range(k)]

        if max(changes) <= cutoff:
            return centroids, corresponding_centroid, distances

        centroids = new_centroids


def kmeans_euclid(data, centroids, cutoff):
    """Shortcut for kmeans clustering using euclidean distance"""
    return kmeans(data, centroids, euclid_dist, euclid_centroid_factory, cutoff)


def kmeans_hamming(data, centroids, cutoff):
    """Shortcut for kmeans clustering using hamming distance"""
    return kmeans(data, centroids, hamming_dist, hamming_centroid_factory, cutoff)


def kmeans_category_hamming(data, centroids, cutoff):
    """Shortcut for kmeans clustering using category hamming distance"""
    return kmeans(data, centroids, category_hamming_dist, category_hamming_centroid_factory, cutoff)

def kmeans_category_hamming_medoids(data, centroids, cutoff):
    """Shortcut for kmeans clustering using category hamming distance generating medoids as centroids"""
    return kmeans(data, centroids, category_hamming_dist, category_hamming_medoid_factory, cutoff)
