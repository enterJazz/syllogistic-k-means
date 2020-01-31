import ccobra
import pandas
import pykmeans
import numpy
import math
import principleextractor
import collections


class SyllogisticKMeans:
    """
    Class to model a syllogistic kmean solver using a category hamming distance
    Input: Syllogistic Data (see e.g. Ragni2016), number of clusters to form (k)
    Output: k Clusters, which correspond to 'cognitive' clusters
    """

    def __init__(self, k):

        self.data = []
        self.subj_data_dict = {}

        self.resp_to_num = {
            'Aac': 0,
            'Aca': 1,
            'Iac': 2,
            'Ica': 3,
            'Eac': 4,
            'Eca': 5,
            'Oac': 6,
            'Oca': 7,
            'NVC': 8
        }

        self.num_to_syllogism = {}
        data = pandas.read_csv("./syllogisms.csv")

        for index, syllogism in data.iterrows():
            self.num_to_syllogism[index] = syllogism['Syllogism']

        self.num_to_resp = {v: k for k, v in self.resp_to_num.items()}
        self.syllogism_to_num = {v: k for k, v in self.num_to_syllogism.items()}

        self.k = k

        self.final_clusters_num = []
        self.final_clusters_syll_list = []  # [{Syllogism : Response, ...}, ...]
        self.corresponding_centroids = []
        self.distances = []

    def add_syllogistic_data(self, data):
        # enc_task = ccobra.syllogistic.encode_task(item.task)
        # enc_resp = ccobra.syllogistic.encode_response(truth, item.task)
        subj_data_dict = {}

        for index, subj_data in data.iterrows():

            if subj_data['id'] not in subj_data_dict:
                subj_data_dict[subj_data['id']] = {}

            task = parse(subj_data['task'])
            response = ccobra.syllogistic.encode_response(parse(subj_data['response']), task)

            subj_data_dict[subj_data['id']][self.syllogism_to_num[ccobra.syllogistic.encode_task(task)]] =\
                self.resp_to_num[response]
            # subj_data_dict[subj_data['id']][ccobra.syllogistic.encode_task(task)] = response

        self.subj_data_dict = subj_data_dict  # {id : [1, 6, 8, 0, ...], ...}

        ordered_dict = collections.OrderedDict(sorted(subj_data_dict.items()))

        for key, val in ordered_dict.items():
            add_list = []
            for i in range(64):
                add_list.append(val[i])
            self.data.append(add_list)

    def generate_clusters(self, cutoff):

        # TODO PROBLEM data is not ordered, even though it is assumed to be ordered
        initial_centroids = generate_centroids(self.data, self.k)

        kme = pykmeans.kmeans_category_hamming(data=self.data, centroids=initial_centroids, cutoff=cutoff)
        self.final_clusters_num = kme[0]
        self.corresponding_centroids = kme[1]
        self.distances = kme[2]

        for cluster in self.final_clusters_num:
            temp_d = {}
            i = 0
            for val in cluster:
                temp_d[self.num_to_syllogism[i]] = (self.num_to_resp[val])
                i += 1
            self.final_clusters_syll_list.append(temp_d)

    def generate_score(self):
        """
        Generates a quality measurement for each final cluster
        """
        # TODO : don't generate score to select initial cluster, create initial cluster with kmeans k=1
        center_scores = []
        for i in range(self.k):
            center_scores.append(0)

        for dp in self.data:
            i = 0
            for center in self.final_clusters_num:
                center_scores[i] += math.pow(math.e, -1 * category_hamming_dist(dp, center))
                i += 1

        return center_scores


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


def parse(item):
    """
    Parses a task or response from a syllogism data csv
    :param item:
    :return:
    """
    l = item.split('/')
    new_l = []
    for entity in l:
        new_l.append(entity.split(';'))

    return new_l


'''
def generate_centroids(k):
    l = []
    val_l = []
    for i in range(0, int(k)):
        for i in range(0, 64):
            val_l.append(randrange(10))
        l.append(val_l)
        val_l = []
    return l
'''


def generate_centroids(X, K):
    """
    A kmeans++ implementation of generating K centroids
    :param X: data, list of lists
    :param K: number of centroids to specify
    :return:
    """
    C = [X[0]]
    for k in range(1, K):
        D2 = numpy.array(
            [min([numpy.inner(category_hamming_dist(c, x), category_hamming_dist(c, x)) for c in C]) for x in X])
        probs = D2 / D2.sum()
        cumprobs = probs.cumsum()
        r = numpy.random.rand()
        for j, p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C



