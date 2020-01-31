import os
import pandas
import pykmeans


class PrincipleExtractor:
    """
    Class to create principle extractor objects; read all principles.csv in ./principles and then, given a list of
    clusters, can generate a list of all corresponding principles for each response in each cluster
    """

    def __init__(self):
        self.principle_csv_dict = {}
        self.get_principle_csvs()

        self.principle_list_dict_dict = {}  # {Principle : {  Syllogism : [response1, response2, ...], ...}}
        # e.g. {'basic' : {'AA1' : ['Oac', 'NVC']}}
        self.generate_principle_list_dict()

    def generate_principle_list_dict(self):

        for principle, dp in self.principle_csv_dict.items():
            for index, data_row in dp.iterrows():
                if principle in self.principle_list_dict_dict:
                    self.principle_list_dict_dict[principle][data_row['Syllogism']] = parse(data_row['Prediction'])
                else:
                    self.principle_list_dict_dict[principle] = {data_row['Syllogism']: parse(data_row['Prediction'])}

    def extract_principles_from_k_clusters(self, k_clusters):
        """
        Extracts the principles from a given list of syllogistic kclusters (list of list) in syllogistic form
        :param k_clusters: list of dict [{syllogism : response, ...}]
        :return: list of dict with the corresponding principles for each kcluster [{syllogism : [principle1, ...]}]
        """
        # k_clusters : [{Syllogism : Response, ...}, ...]
        k_principle_dict_dict = {}  # { cluster_num : k_principle_dict (see below), ...}
        k_cluster_num = 0
        for cluster in k_clusters:  # for each cluster : ['Aac', 'NVC', ...]
            k_principle_dict = {}  # {syllogism : [principle1, ...]}
            for syllogism, response in cluster.items():  # for each response : 'Aac'
                for principle, syllogism_response_dict in self.principle_list_dict_dict.items():  # key: principle,
                    # val = {Syllogism : [response1, response2, ...]}
                    if response in syllogism_response_dict[syllogism]:
                        if syllogism in k_principle_dict:
                            k_principle_dict[syllogism].append(principle)
                        else:
                            k_principle_dict[syllogism] = [principle]
                    # else:
                    #   k_principle_dict[syllogism] = []

            k_principle_dict_dict[k_cluster_num] = k_principle_dict
            k_cluster_num += 1

        coverage = compute_coverage(k_principle_dict_dict)

        return k_principle_dict_dict, coverage

    def get_principle_csvs(self):

        for entry in os.scandir("./principles"):
            self.principle_csv_dict[entry.path.split('/')[-1].split('.')[0]] = pandas.read_csv(entry.path)
            #  e.g. d[basic] = pd.csv


def parse(unparsed_responses):
    return unparsed_responses.split(';')


def compute_coverage(k_cluster_principles):
    k_cluster_coverage_dict = {}  # {k_cluster_num : {principle : coverage, ...}, ...}

    for k_cluster_num, k_cluster_dict in k_cluster_principles.items():
        coverage_dict = {}  # {principle : coverage, ...}
        for syllogism, corresponding_principle_list in k_cluster_dict.items():
            for principle in corresponding_principle_list:
                if principle in coverage_dict:
                    coverage_dict[principle] += 1
                else:
                    coverage_dict[principle] = 1

        for principle, num in coverage_dict.items():
            coverage_dict[principle] = num / 64

        k_cluster_coverage_dict[k_cluster_num] = coverage_dict
        k_cluster_num += 1

    return k_cluster_coverage_dict


def compute_participant_correspondence(data, k_clusters):
    # data : {id : [1, 6, 8, 0, ...], ...}
    # k_clusters : [[0, 2, 5, ...], ...]

    result_dict = {}
    result_dict['participant_correspondence'] = {}
    result_dict['k_cluster_correspondence'] = {}
    result_dict['total_correspondence'] = 0

    participants_per_cluster = {}
    distance_per_cluster = {}
    for i in range(len(k_clusters)):
        participants_per_cluster[i] = 0
        distance_per_cluster[i] = 0
    # calculate participant correspondence
    for id, responses in data.items():
        matching_cluster = 0
        match_distance = 64
        for cluster in k_clusters:
            if pykmeans.category_hamming_dist(responses, cluster) < match_distance:
                match_distance = pykmeans.category_hamming_dist(responses, cluster)
                matching_cluster = k_clusters.index(cluster)
        result_dict['participant_correspondence'][id] = {'k-cluster': matching_cluster, 'distance': match_distance,
                                                         'percentage match': 1 - match_distance / 64}

        participants_per_cluster[matching_cluster] += 1
        distance_per_cluster[matching_cluster] += match_distance

    for i in range(len(k_clusters)):
        result_dict['k_cluster_correspondence'][i] = {'# Participants': participants_per_cluster[i],
                                                      'Mean Matching': 1-distance_per_cluster[i]/participants_per_cluster[i]/64}

    total_match = 0
    for key, val in result_dict['k_cluster_correspondence'].items():
        total_match += val['Mean Matching']
    result_dict['total_correspondence'] = total_match / len(result_dict['k_cluster_correspondence'])

    return result_dict


def write_results(k_cluster_result_dict, coverage_dict, k_cluster_principles, participant_correspondence):
    pandas.DataFrame(k_cluster_result_dict).to_csv('./results/k_cluster_corresponding_principles.csv')
    pandas.DataFrame(coverage_dict).to_csv('./results/coverage.csv')
    pandas.DataFrame(k_cluster_principles).to_csv('./results/k_cluster_corresponding_answer.csv')
    pandas.DataFrame(participant_correspondence).to_csv('./results/participant_correspondence.csv')
