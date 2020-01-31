import collections

import numpy as np

import ccobra
import syllogistickmeans
import pandas
import principleextractor


class SyllogisticKMeansModel(ccobra.CCobraModel):
    def __init__(self, k=6, initial_iterations=1, cutoff=5, name='SyllogisticKMeansModel'):
        super(SyllogisticKMeansModel, self).__init__(name, ['syllogistic'], ['single-choice'])

        # Parameters
        self.k = k  # k is number of clusters
        self.skm = syllogistickmeans.SyllogisticKMeans(k)
        self.cutoff = cutoff  # a value used to stop iterating
        # Its type is the return type of distf (category_hamming_distance -> pykmeans)
        # When the maximum centroid change is less than the cutoff then we stop iterating
        self.current_iteration = 0
        self.initial_iterations = initial_iterations

        # k clusters (computed in pre train)
        self.k_clusters = []  # [{Syllogism : Response, ...}, ...]
        self.current_k_cluster = []  # the k cluster used to currently make predictions

        # k cluster scores (deciding what k-cluster fits to individual)
        self.k_cluster_scores = {}
        for i in range(k):
            self.k_cluster_scores[i] = 0

    def pre_train(self, dataset):
        """
        the dataset is given to compute the k clusters and the initial general k cluster
        :param dataset:
        :return:
        """
        pd_dataset = pandas.read_csv('../../data/Ragni2016.csv')
        pe = principleextractor.PrincipleExtractor()
        # NOTE this does not use CCOBRA's dataset; if different dataset is to be used, must be specified here and
        # not in the .json; or with argument given TODO
        self.skm.add_syllogistic_data(data=pd_dataset)

        self.skm.generate_clusters(cutoff=self.cutoff)

        results = pe.extract_principles_from_k_clusters(self.skm.final_clusters_syll_list)

        participant_correspondence = principleextractor.compute_participant_correspondence(self.skm.subj_data_dict,
                                                                                           self.skm.final_clusters_num)
        principleextractor.write_results(results[0], results[1], self.skm.final_clusters_syll_list,
                                         participant_correspondence)

        self.k_clusters = self.skm.final_clusters_syll_list

        gen_skm = syllogistickmeans.SyllogisticKMeans(1)
        gen_skm.add_syllogistic_data(data=pd_dataset)

        gen_skm.generate_clusters(cutoff=self.cutoff)

        self.current_k_cluster = gen_skm.final_clusters_syll_list[0]

    def predict(self, item, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)

        return ccobra.syllogistic.decode_response(self.current_k_cluster[enc_task],
                                                  item.task)

    def adapt(self, item, response, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_resp = ccobra.syllogistic.encode_response(response, item.task)

        # if a participants answer corresponds to an answer given by some k cluster, increment score
        for cluster_num in range(self.k):
            if enc_resp == self.k_clusters[cluster_num][enc_task]:
                self.k_cluster_scores[cluster_num] += 1

        # for the first initial_iterations, use the initial general cluster
        if self.initial_iterations < self.current_iteration:
            # set the the k cluster to be used as the one with the highest score
            if self.current_iteration % 5 == 0:
                self.current_k_cluster = self.k_clusters[max(self.k_cluster_scores, key=self.k_cluster_scores.get)]

        self.current_iteration += 1


