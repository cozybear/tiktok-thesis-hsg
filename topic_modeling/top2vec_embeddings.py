from top2vec import Top2Vec
import os
import numpy as np
import regex as re
import pandas as pd
import emoji
import string
import pprint
import json
import matplotlib.pyplot as plt
import warnings
import unicodedata
import umap
import umap.plot
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from nltk import FreqDist
warnings.filterwarnings("ignore")


path_test = '/.../metadata_test'
path = '/.../total'



def split_emojis(s):
    return ''.join((' ' + c + ' ') if c in emoji.EMOJI_DATA else c for c in s)


def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

# save model
def save_model(top2vec_model):
    if not os.path.isdir('../top2vec'):
        os.makedirs('../top2vec')
    top2vec_model.save("top2vec/top2vec_descriptions_2.model")
    print('Model saved to file')

#save topics
def save_user_topics(dataframe):
    if not os.path.isdir('../top2vec'):
        os.makedirs('../top2vec')
    dataframe.drop('descriptions', axis=1).to_csv('top2vec/user_topics_descriptions.csv', index=False)
    print('DataFrame saved to file')






def get_descs(pathname):

    # get all descriptions per author
    df = pd.DataFrame(columns=['user', 'descriptions'])
    # for every folder in directory
    for folder in os.listdir(pathname):
        # get all descriptions from all authors
        descriptions = ""
        # if it is a directory
        if os.path.isdir(os.path.join(pathname, folder)):
            with open(os.path.join(pathname, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    #this contains video description and hashtags
                    description = item['video_stats']['description']
                    # this contains signature text
                    signature = item['author_stats']['signature']
                    # add both  so for every video we get description + sticker words
                    description += ' '+signature


                    # some basic cleaning
                    # Make lowercase
                    description = description.lower()

                    # Remove numbers
                    description = ''.join([i for i in description if not i.isdigit()])
                    # add whitespace between emojis
                    description = split_emojis(description)
                    # remove emojis
                    description = give_emoji_free_text(description)

                    # add whitespace before hashtags and filter ><
                    description = description.replace("#", " ").replace('<', '').replace('>', '')
                    # Remove punctuation
                    description = re.sub(r"[^\P{P}]+", "", description)
                    # remove special characters
                    description = re.sub('[^A-Za-z0-9]+', ' ', description).strip()
                    # remove links
                    description = re.sub(r'http\S+', '', description)
                    # remove single alphabetical letters
                    description = re.sub(r"\b[a-zA-Z]\b", "", description)
                    # covert to normal font
                    description = unicodedata.normalize('NFKD', description).encode('ascii', 'ignore').decode('utf-8')

                    # remove short words with less than 3 letters
                    description = ' '.join(word for word in description.split() if len(word) > 2)

                    if len(description) > 0:
                        #descriptions.append(description)
                        #append string to string
                        descriptions += (description+". ")
                        print(description)
            if descriptions:


                df = df.append({'user': folder, 'descriptions': filtered_text}, ignore_index=True)
    return df


#df = get_descs(path)
df.to_csv("top2vec/descriptions_2.csv", index=False)

df = pd.read_csv("../top2vec/descriptions_2.csv")


#model input document is a list of strings
all_descriptions = df['descriptions'].explode().reset_index(drop=True).tolist()
print(all_descriptions)


#init and train model
#TODO: use bert sentence transormer
model = Top2Vec(all_descriptions, embedding_model='universal-sentence-encoder-multilingual')

save_model(model)

#load trained model from file
model = Top2Vec.load("../top2vec/top2vec.model")



topic_sizes, _ = model.get_topic_sizes()
#print(topic_sizes, topic_nums)

topic_words, word_scores, topic_nums = model.get_topics()
#print(topic_words, topic_nums)



words=[]
for l in topic_words:
    words.append(l[:10])

topics = pd.DataFrame({'topic': topic_nums, 'words': words, 'size': topic_sizes})
topics['words'] = [', '.join(map(str, l)) for l in topics['words']]

#print(topics)
topics.to_csv('top2vec/topics_descriptions.csv', index=False)








umap_args = {
    "n_neighbors": 15,
    "min_dist": 0.2,
    "n_components": 2, # 5 -> 2 for plotting
    "metric": "cosine",
}

labels = model.doc_top
#print(labels)

umap_model = umap.UMAP(**umap_args).fit(model.document_vectors)
p = umap.plot.points(umap_model,  show_legend=False, labels=labels,  color_key_cmap='hsv', width=1410, height=1000)

plt.savefig('../top2vec/documents_labeled.png', bbox_inches='tight', dpi=400)
#umap.plot.plt.show()



#get topics wordcloud
#topic_words, word_scores, topic_nums = model.get_topics()




topic_counter = 0
for topic in topic_nums:
    fig = plt.figure(clear=True)
    model.generate_topic_wordcloud(topic, background_color="white")
    plt.savefig(f'top2vec/topic_{topic_counter}_descriptions_wordcloud.png', dpi=400, bbox_inches="tight")
    topic_counter += 1




#reassign topics to df
df["topic"] = model.doc_top



#save_user_topics(df)

