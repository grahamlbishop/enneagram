
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

embeddings_dict = {}
test = []
with open(os.path.expanduser("~/Downloads/glove.twitter.27B/glove.twitter.27B.50d.txt"), 'r') as f:
    for line in f:
        values = line.split()
        test.append(len(values))
        if len(values)==51:
            word = values[0]
            vector = np.asarray(values[1:], "float32")
        elif len(values)==50:
            word = " "
            vector = np.asarray(values, "float32")
        embeddings_dict[word] = vector
        
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


tsne = TSNE(n_components=2, random_state=0, perplexity=15)

T1P = ["principled", "disciplined", "diplomatic", "ethical", "conscientious"]
T1N = ["perfectionist", "critical", "judgmental", "rigid", "angry"]
T2P = ["supportive", "generous", "helpful", "loving", "empathic"]
T2N = ["manipulative", "needy", "smothering", "codependent", "prideful"]
T3P = ["accomplished", "successful", "productive", "effective", "popular"]
T3N = ["workaholic", "pretender", "overachiever", "self-promoting", "deceitful"]
T4P = ["original", "sensitive", "romantic", "aesthetic", "creative"]
T4N = ["melancholic", "dramatic", "misunderstood", "envious", "self-absorbed"]
T5P = ["insightful", "wise", "observant", "curious", "cerebral"]
T5N = ["distant", "withholding", "greedy", "isolated", "condescending"]
T6P = ["reliable", "loyal", "honorable", "prepared", "trustworthy"]
T6N = ["fearful", "worrier", "pessimistic", "projecting", "paranoid"]
T7P = ["entertaining", "joyful", "optimistic", "wonder", "planner"]
T7N = ["impulsive", "self-indulgent", "dabbling", "spacey", "gluttonous"]
T8P = ["strong", "independent", "leader", "powerful", "straightforward"]
T8N = ["tough", "lustful", "bossy", "vindictive", "controlling"]
T9P = ["contented", "harmonious", "peaceful", "easygoing", "universal"]
T9N = ["procrastinating", "indecisive", "oblivious", "lazy", "apathetic"]     


words = ["principled", "disciplined", "diplomatic", "ethical", "conscientious", "perfectionist", "critical", "judgmental", "rigid", "angry", "supportive", "generous", "helpful", "loving", "empathic", "manipulative", "needy", "smothering", "codependent", "prideful", "accomplished", "successful", "productive", "effective", "popular", "workaholic", "pretender", "overachiever", "self-promoting", "deceitful", "original", "sensitive", "romantic", "aesthetic", "creative", "melancholic", "dramatic", "misunderstood", "envious", "self-absorbed", "insightful", "wise", "observant", "curious", "cerebral", "distant", "withholding", "greedy", "isolated", "condescending", "reliable", "loyal", "honorable", "prepared", "trustworthy", "fearful", "worrier", "pessimistic", "projecting", "paranoid", "entertaining", "joyful", "optimistic", "wonder", "planner", "impulsive", "self-indulgent", "dabbling", "spacey", "gluttonous", "strong", "independent", "leader", "powerful", "straightforward", "tough", "lustful", "bossy", "vindictive", "controlling", "contented", "harmonious", "peaceful", "easygoing", "universal", "procrastinating", "indecisive", "oblivious", "lazy", "apathetic"] 
vectors = [embeddings_dict[word] for word in words]
my_array = np.array(vectors)
Y = tsne.fit_transform(my_array)
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()

typelist = [T1P, T1N, T2P, T2N, T3P, T3N, T4P, T4N, T5P, T5N, T6P, T6N, T7P, T7N, T8P, T8N, T9P, T9N]
datapoints = []


for i in typelist:
    datapoints2 = []
    for j in i:
        datapoints2.append(embeddings_dict[j])
    datapoints.append(sum(datapoints2)/5)

datapoints_array = np.array(datapoints, dtype=object)
color_list = ["#bf6171",
"#6db842",
"#aa5bca",
"#54c081",
"#d04596",
"#3f8041",
"#6367cf",
"#c7a83a",
"#7485ca",
"#d15e29",
"#46aed7",
"#d24250",
"#4ab29c",
"#b971ae",
"#9eb165",
"#e09466",
"#687428",
"#95662e"]
Y2 = tsne.fit_transform(datapoints_array)
plt.scatter(Y2[:, 0], Y2[:, 1], c=color_list)

label_list = ["T1P", "T1N", "T2P", "T2N", "T3P", "T3N", "T4P", "T4N", "T5P", "T5N", "T6P", "T6N", "T7P", "T7N", "T8P", "T8N", "T9P", "T9N"]

for label, x, y in zip(label_list, Y2[:, 0], Y2[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()
