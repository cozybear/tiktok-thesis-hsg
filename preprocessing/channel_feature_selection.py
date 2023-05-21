import os
import pandas as pd
import pprint
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


path = '/.../tiktokdata/total'


def min_max_normalizer(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

#import images, user, encodings
li = []
noface_counter = 0
# for every folder in directory
for folder in os.listdir(path):
    # if it is a directory
    if os.path.isdir(os.path.join(path, folder)):
        try:
            df = pd.read_csv(os.path.join(path, folder, 'videos', 'dbscan_inandout_total'), index_col=None)
            #get channel owner entries
            df_cleaned = df[df['group'] == 'Channel Owner'].copy()
            #add username
            df_cleaned['user'] = folder
            #add face image path
            df_cleaned['path_to_image'] = df_cleaned.apply(
                lambda row: os.path.join(path, folder, 'videos', 'faces_cleaned', row['name']), axis=1)

            li.append(df_cleaned[['user', 'path_to_image', 'encodings']])

        except Exception as e:
            print(e)
            noface_counter +=1
            continue


print(noface_counter)

path_encoding_df = pd.concat(li, axis=0, ignore_index=True)




#import tabular features
df_post_frequency = pd.read_csv('../data_analysis/post_frequency.csv', index_col=None)
df_channel_age = pd.read_csv('../data_analysis/channel_age.csv', index_col=None)
df_verified = pd.read_csv('../data_analysis/verified.csv', index_col=None)
df_topics = pd.read_csv('../top2vec/user_topics.csv', index_col=None)
df_languages = pd.read_csv('../languages/languages.csv', index_col=None)



#encode features to categories
df_languages["language_cat"] = df_languages["language"].astype('category').cat.codes
df_verified["verified"] = df_verified["verified"].astype('category').cat.codes
df_topics["topic_cat"] = df_topics["topic"].astype('category').cat.codes



#convert to int columns
df_topics.topic = df_topics.topic.astype(int)
df_languages.language = df_languages.language_cat.astype(int)

#normalize continous columns
df_channel_age.channel_age = min_max_normalizer(df_channel_age.channel_age)
df_post_frequency.post_frequency = min_max_normalizer(df_post_frequency.post_frequency)


#import engagement rates target labels
df_labels = pd.read_csv('../data_analysis/author_engagement_rates.csv', index_col=None)




#edit/remove outliers (get only channels where engagemt rate is under 25 %
df_labels['average_engagement_impressions'] = np.where(df_labels['average_engagement_impressions'] >=25.439,
                                                       np.nan,
                                                       df_labels['average_engagement_impressions'])


#merge all dfs on user column
data_frames = [path_encoding_df,
               df_labels, df_post_frequency, df_channel_age,
               df_verified, df_topics, df_languages]

df_final = reduce(lambda left, right: pd.merge(left, right, on=['user'],
                                            how='inner'), data_frames)


#drop nan rows
df_final = df_final.dropna(how='any')
print(df_final.user.nunique())



#save final file
if not os.path.isdir('../final'):
    os.makedirs('../final')
df_final.to_csv(os.path.join('../final', 'df_final_channel.csv'), index=False)

#save tabular feature dataset
df_tabular = df_final.groupby("user").sample(n=1, random_state=23)
df_tabular.to_csv("final/df_tabular.csv")