import pandas
import syllogistickmeans 
import principleextractor
import sys

skm = syllogistickmeans.SyllogisticKMeans(int(sys.argv[1]))

df = pandas.read_csv('../../data/Ragni2016.csv')

skm.add_syllogistic_data(df)

skm.generate_clusters(cutoff=0)

pe = principleextractor.PrincipleExtractor()

results = pe.extract_principles_from_k_clusters(skm.final_clusters_syll_list)

participant_correspondence = principleextractor.compute_participant_correspondence(skm.subj_data_dict, skm.final_clusters_num)

principleextractor.write_results(results[0], results[1], skm.final_clusters_syll_list, participant_correspondence)
