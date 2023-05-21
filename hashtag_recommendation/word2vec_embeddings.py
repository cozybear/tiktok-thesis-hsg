from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import pprint
import multiprocessing
import string
from gensim.models import Word2Vec
from gensim import utils
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import regex as re
import emoji
import unicodedata
from umap import UMAP
import warnings
from gensim.test.utils import datapath
from ast import literal_eval
import csv



warnings.filterwarnings("ignore")






path_test = '/Volumes/My_Passport/tiktokdata/metadata_test'
path = '/Volumes/My_Passport/tiktokdata/total'



prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
plt.rcParams['font.family'] = prop.get_family()



def split_emojis(s):
    return ''.join((' ' + c + ' ') if c in emoji.EMOJI_DATA else c for c in s)

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

def get_dataframe(path):
    #get all descriptions per author
    hashtags_all = []
    users = []
    # for every folder in directory
    for folder in os.listdir(path):

        hashtags = []
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:


                    #this contains hashtags only
                    hashtag = item['video_stats']['hashtags']
                    # filter empty hashtags from list
                    hashtag = list(filter(None, hashtag))
                    #convert to string
                    hashtag = ' '.join(hashtag)



                    # some basic cleaning
                    # Make lower
                    hashtag = hashtag.lower()


                    # Remove numbers
                    hashtag = ''.join([i for i in hashtag if not i.isdigit()])
                    # add whitespace between emojis
                    hashtag = split_emojis(hashtag)
                    # remove emojis
                    hashtag = give_emoji_free_text(hashtag)
                    # remove short words with less than 3 letters
                    hashtag = ' '.join(word for word in hashtag.split() if len(word) > 2)
                    # add whitespace before hashtags and filter ><
                    hashtag = hashtag.replace("#", " ").replace('<', '').replace('>', '')
                    # Remove punctuation
                    hashtag = re.sub(r"[^\P{P}]+", "", hashtag)
                    # remove special characters & numbers
                    hashtag = re.sub('[^A-Za-zäöüÄÖÜß0-9]+', ' ', hashtag)
                    # remove links
                    hashtag = re.sub(r'http\S+', '', hashtag)
                    # remove single alphabetical letters
                    hashtag = re.sub(r"\b[a-zA-Z]\b", "", hashtag)
                    # covert to normal font
                    hashtag = unicodedata.normalize('NFKD', hashtag).encode('ascii', 'ignore').decode('utf-8')




                    # split sentence to tokens in list
                    hashtag = hashtag.strip().split()
                    print(hashtag)

                    if len(hashtag) > 0:
                        hashtags_all.extend([hashtag])



    df = pd.DataFrame({'hashtags': hashtags_all})

    return df



def tsne_word2vec(model):
    labels = []
    tokens = []

    for word in model.wv.key_to_index:
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2,  random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 14))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='dodgerblue', s=2)

        """"
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        """

    #plt.title("t-SNE Word2Vec Token Embeddings")
    plt.xlabel("t-SNE_1")
    plt.ylabel("t-SNE_2")
    plt.savefig('word2vec/tsneplot.png', dpi=400)
    plt.show()

    ################################################################
    # add annotated plot for different examples
    """
    most_sim_words = [i[0] for i in model.wv.most_similar(positive='#fyp', topn=10)]

    plt.figure(figsize=(16, 14))

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='lightblue', s=2)

        if labels[i] in most_sim_words:
            plt.scatter(x[i], y[i], label=labels[i], s=5)

    plt.legend(title="Top 10 most similar tokens to ""#fyp"")
    
    plt.title("t-SNE Word2Vec Token Embeddings")
    plt.xlabel("Dimension-1")
    plt.ylabel("Dimension-2")
    plt.show()

    ###############################################################
    # add annotated plot for different examples
    most_sim_words = [i[0] for i in model.wv.most_similar(positive='happy', topn=10)]

    plt.figure(figsize=(16, 14))

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='lightblue', s=2)

        if labels[i] in most_sim_words:
            plt.scatter(x[i], y[i], label=labels[i], s=5)

    plt.legend(title="Top 10 most similar tokens to ""happy"")
    plt.title("t-SNE Word2Vec Token Embeddings")
    plt.xlabel("Dimension-1")
    plt.ylabel("Dimension-2")
    plt.show()
    """




def tsne_user(df):
    "Creates and TSNE model and plots it"
    users = df['user'].tolist()
    embeddings = df['mean_embedding'].tolist()


    tsne_model = TSNE(perplexity=40, n_components=2, init='pca',  random_state=23)
    new_values = tsne_model.fit_transform(embeddings)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='black', s=1)
        """"
        plt.annotate(users[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        """
    plt.title("t-SNE User Level Topic Embeddings")
    plt.xlabel("Dimension-1")
    plt.ylabel("Dimension-2")
    plt.show()


def pca_word2vec(model):

    labels = []
    tokens = []

    for word in model.wv.key_to_index:
        tokens.append(model.wv[word])
        labels.append(word)

    # first reduce dimensionality before feeding to t-sne
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(tokens)


    x = []
    y = []

    for value in X_pca:
        x.append(value[0])
        y.append(value[1])


    plt.figure(figsize=(16, 14))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='dodgerblue', s=2)
    plt.title("PCA Word2Vec Token Embeddings")
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.show()


    ################################################################
    #add annotated plot for different examples
    most_sim_words = [i[0] for i in model.wv.most_similar(positive='#fyp', topn=10)]

    plt.figure(figsize=(16, 14))

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='lightblue', s=2)

        if labels[i] in most_sim_words:
            plt.scatter(x[i], y[i], label=labels[i],  s=5)


    plt.legend(title="Top 10 most similar tokens to ""#fyp""")
    plt.title("PCA Word2Vec Token Embeddings")
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.show()

    ###############################################################
    # add annotated plot for different examples
    most_sim_words = [i[0] for i in model.wv.most_similar(positive='happy', topn=10)]

    plt.figure(figsize=(16, 14))

    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='lightblue', s=2)

        if labels[i] in most_sim_words:
            plt.scatter(x[i], y[i], label=labels[i], s=5)

    plt.legend(title="Top 10 most similar tokens to ""happy""")
    plt.title("PCA Word2Vec Token Embeddings")
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.show()


def pca_user(df):
    "Creates and TSNE model and plots it"
    users = df['user'].tolist()
    embeddings = df['mean_embedding'].tolist()

    # first reduce dimensionality before feeding to t-sne
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)

    x = []
    y = []

    for value in X_pca:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='black', s=1)
    """"
        plt.annotate(users[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        """

    plt.title("User Level Topic Embeddings")
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.show()


    ''''
    # make a TSNE scatterplot of encodings
    X = df['mean_embedding'].tolist()

    model = TSNE(n_components=2, init='random', perplexity=2)

    tsne_data = model.fit_transform(X)

    df_tsne = pd.DataFrame()
    df_tsne["Component-1"] = tsne_data[:, 0]
    df_tsne["Component-2"] = tsne_data[:, 1]

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.scatterplot(x=df_tsne["Component-1"], y=df_tsne["Component-2"],
                    data=df_tsne, legend='full').set(
        title="")
    plt.show()
    '''


#TODO: set markersize; plot different examples of tsne

def umap_word2vec(model):
    labels = []
    tokens = []

    for word in model.wv.index_to_key:
        tokens.append(model.wv[word])
        labels.append(word)


    reduced_embeddings = UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine').fit_transform(tokens)

    x = []
    y = []

    for value in reduced_embeddings:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 14))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='dodgerblue', s=2)
    plt.title("UMAP: Word2Vec Token Embeddings")
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.show()


def umap_user(df):
    users = df['user'].tolist()
    embeddings = df['mean_embedding'].tolist()

    reduced_embeddings = UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    x = []
    y = []

    for value in reduced_embeddings:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 14))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c='dodgerblue', s=2)
        """"
        plt.annotate(users[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        """
    plt.title("UMAP: Word2Vec User Embeddings")
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.show()



def word2vec(sentences):
    #init and train model
    model = Word2Vec(sentences, min_count=1, workers=3,  sg=1)

    return model


def test(model, word):
    try:
        sim = model.wv.most_similar(positive=str(word), topn=10)
        print(f'\nMost similar hashtags for #{str(word)}:')
        pprint.pprint(sim)
        return sim

    except KeyError:
        print(f'#{word} not present in vocabulary!')



#per user descriptions: df from above
def compute_mean_embedding(model, listoflists):
    words=[]
    for listofwords in listoflists:
        for word in listofwords:
            #append weach word to list
            words.append(word)

    # for every word in list, get embedding and then take mean of all embeddings
    mean_embedding = np.mean([model.wv[w] for w in words], axis=0)
    print(mean_embedding)

    return mean_embedding


def load_model(path):
    # Load pre-trained Word2Vec model.
    model = Word2Vec.load(path)
    return model


#save model
def save_model(model):
    if not os.path.isdir('../word2vec'):
        os.makedirs('../word2vec')
    model.save("word2vec/word2vec_hashtags_4_100.model")



#save topic embedding
def save_embeddings(df):
    if not os.path.isdir('../word2vec'):
        os.makedirs('../word2vec')
    df.drop('descriptions', axis=1).to_csv('word2vec/topic_embedding.csv', index=False)







#get df
#df = get_dataframe(path)
#df.to_csv("word2vec/hashtags_2.csv", index=False)

#df = pd.read_csv("word2vec/hashtags_2.csv")

#df['hashtags'] = df.hashtags.apply(lambda x: literal_eval(str(x)))



"""
#only hashtags where len > 2
df = df.loc[df.hashtags.str.len() > 2].reset_index(drop=True)

#total and unique # hashtags
print(len(df.index))
print(len(set(df.hashtags.explode().tolist())))


#train/test-split: 1% test size

train, test = train_test_split(df, test_size=0.01, random_state=32)
"""


#list of lists of strings for every video and user
#sentences_list = train.tolist()



#init and train model
#model = Word2Vec(sentences=sentences_list, vector_size=100, window=5, min_count=3, workers=4, sg=1)
#model2 = Word2Vec(sentences=sentences_list, vector_size=200, window=5, min_count=3, workers=4, sg=1)
#model3 = Word2Vec(sentences=sentences_list, vector_size=300, window=5, min_count=3, workers=4, sg=1)


#save model
#save_model(model)

#load model from file
model = load_model("../word2vec/word2vec_hashtags_4_100.model")
#model2 = load_model("word2vec/word2vec_hashtags_4_200.model")
#model3 = load_model("word2vec/word2vec_hashtags_4_300.model")

#tsne_word2vec(model)



#print vocab size
#print(len(model.wv.index_to_key))

"""
#save to tensors
with open('tensors.tsv', 'w') as tensors:
    with open('metadata.tsv', 'w') as metadata:
        for word in model.wv.index_to_key:
            metadata.write(word + '\n')
            vector_row = '\t'.join(map(str, model.wv[word]))
            tensors.write(vector_row + '\n')

"""
#test hashtags recommendations (most similar)

test(model, 'fyp')
test(model, 'dance')
test(model, 'love')
test(model, 'parati')
test(model, 'cats')
test(model, 'duet')
test(model, 'trending')
test(model, 'funny')
test(model, 'travel')
test(model, 'comedy')
test(model, 'art')
test(model, 'gucci')
test(model, 'football')
test(model, 'beach')
test(model, 'music')
test(model, 'doityourself')
test(model, 'fashion')
test(model, 'germany')
test(model, 'furdich')
data = test(model, 'schweiz')
test(model, 'stgallen')
test(model, 'horses')
test(model, 'books')


with open('../word2vec/schweiz.csv', 'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Hashtag', 'Cosine Similarity'])
    for row in data:
        csv_out.writerow(row)


"""

#similarity correlations
similarities_wordsim = model1.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
similarities_simlex = model1.wv.evaluate_word_pairs(datapath('simlex999.txt'))



print(similarities_wordsim)
print(similarities_simlex)

"""

"""


#plot
#tsne_word2vec(model)
#umap_word2vec(model)
#pca_word2vec(model)
#tsne_user(df)
#umap_user(df)
#pca_user(df)



"""


"""


#quantitative evaluation






#print(X_test)

def atleast1in(a, b):
    return not set(a).isdisjoint(b)

def multiplein(a, b):
    return len(set(a).intersection(b))


def atleastonecorrect(model, X_test):
    hit = 0
    tries = 0
    preds = []
    for index, row in X_test.iterrows():
        r = row['hashtags']
        print(r)
        for word in r:
            try:
                print('Word: ', word)
                #pred = list(zip(*model.wv.most_similar(positive=word, topn=5)))[0]
                pred = [model.wv.most_similar(positive=word, topn=1)[0][0]]
                print('Prediction: ', pred, '\n')
                #preds.extend(pred)

                if atleast1in(pred, r):
                    hit += 1
                    tries += 1

                else:
                    tries += 1

            except KeyError:
                continue



    aloc = hit/tries
    return aloc

metric1 = atleastonecorrect(model=model1, X_test=test)
metric2 = atleastonecorrect(model=model2, X_test=test)
metric3 = atleastonecorrect(model=model3, X_test=test)




def multiplecorrect(model, X_test):
    hit = 0
    tries = 0
    preds = []
    for index, row in X_test.iterrows():
        r = row['hashtags']
        print(r)
        K = len(r)-1
        for word in r:
            try:
                print('Word: ', word)
                pred = list(zip(*model.wv.most_similar(positive=word, topn=K)))[0]
                #pred = model.wv.most_similar(positive=word, topn=1)[0][0]
                print('Prediction: ', pred, '\n')
                #preds.extend(pred)

                if multiplein(pred, r) >= 2:
                    hit += 1
                    tries += 1

                else:
                    tries += 1

            except KeyError:
                continue



    muc = hit/tries
    return muc

metric_1_muc = multiplecorrect(model=model1, X_test=test)
metric_2_muc = multiplecorrect(model=model2, X_test=test)
metric_3_muc = multiplecorrect(model=model3, X_test=test)


#show results
print(metric1, metric2, metric3)
print(metric_1_muc, metric_2_muc, metric_3_muc)

"""
