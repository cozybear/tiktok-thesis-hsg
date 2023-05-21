import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import random
import seaborn as sns


#Building the input pipeline for our regression model

# Converting the values into features

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  #if isinstance(value, type(tf.constant(0))):
    #value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0,
                      feature1,
                      feature2,
                      label):
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  features = {
      'image': _bytes_feature(feature0),
      'user': _bytes_feature(feature1),
      'emb': _float_array_feature(feature2),
      'label': _float_feature(label),

  }

  return tf.train.Example(features=tf.train.Features(feature=features))


def serialize_celeba_example(feature0, feature1, feature2, label):


  features = {
      'image': _bytes_feature(feature0),
      'user': _bytes_feature(feature1),
      'label': _float_feature(label)
  }

  return tf.train.Example(features=tf.train.Features(feature=features))



#read features and labels
df_final = pd.read_csv('../final/df_final_channel.csv')



#plot counts
x = df_final.user.value_counts()

print(max(x))

#plot faces per user
sns.displot(x, height=5, aspect=1.7)
plt.tight_layout()
plt.margins(x=0)
plt.xlabel('# Face Images')
plt.ylabel('User Count')
path_image = os.path.join('final', 'face_count.png')
plt.savefig(path_image, dpi=400, bbox_inches="tight")
plt.close()




#get only users where more than 31 faces exist
n = 31
v = df_final.user.value_counts()
df_final = df_final[df_final.user.isin(v.index[v.gt(n)])]


print(df_final.user.nunique())


#edit columns
df_final.encodings = df_final.encodings.apply(lambda x: x.replace('[', '').replace(']', ''))
df_final.encodings = df_final.encodings.apply(lambda x: np.fromstring(x, dtype=float, sep=' '))



#set face cap for dataset and sample n faces for each user
cap = 32
df_final = df_final.groupby("user").sample(n=32, random_state=7)

print(df_final.average_engagement_impressions.describe())


#split df in train/test/val
splitter = GroupShuffleSplit(test_size=.1, n_splits=1, random_state=23)
split = splitter.split(df_final, groups=df_final['user'])
train_inds, test_inds = next(split)

train = df_final.iloc[train_inds]
test = df_final.iloc[test_inds]

#split train set again in train and validation
splitter2 = GroupShuffleSplit(test_size=.2, n_splits=1, random_state=21)
split2 = splitter2.split(train, groups=train['user'])
train2_inds, val_inds = next(split2)

train_final = train.iloc[train2_inds]
validation_final = train.iloc[val_inds]


#get 1face test dataset
test_1face = test.groupby("user").sample(n=1, random_state=67)


#see example and shape
print(train_final, validation_final, test, test_1face)




#choose engagement_metric (target variable)
engagement_metric = 'average_engagement_impressions'




# FaceTagger dataset
file = 'final/tiktok_dataset.tfrecord'

#write
with tf.io.TFRecordWriter(file) as writer:
  for index, row in df_final.iterrows():
    image_string = open(row['path_to_image'], 'rb').read()
    user = row['user'].encode()
    label = row[engagement_metric]
    tf_example = serialize_celeba_example(image_string, user, label)
    writer.write(tf_example.SerializeToString())




#write tf train dataset
train_file = 'final/train_32face.tfrecord'
with tf.io.TFRecordWriter(train_file) as writer:
  for index, row in train_final.iterrows():

    image_string = open(row['path_to_image'], 'rb').read()
    user = row['user'].encode()
    emb = row['encodings']
    label = row[engagement_metric]

    tf_example = serialize_example(image_string,
                                   user,
                                   emb,
                                   label)
    writer.write(tf_example.SerializeToString())

#write tf validation dataset
train_file = 'final/validation_1face.tfrecord'



with tf.io.TFRecordWriter(train_file) as writer:
  for index, row in validation_final.iterrows():

    image_string = open(row['path_to_image'], 'rb').read()
    user = row['user'].encode()
    emb = row['encodings']
    label = row[engagement_metric]

    tf_example = serialize_example(image_string,
                                   user,
                                   emb,
                                   label)
    writer.write(tf_example.SerializeToString())


#write tf test dataset
test_file = 'final/test_32face.tfrecord'

with tf.io.TFRecordWriter(test_file) as writer:
  for index, row in test.iterrows():

    image_string = open(row['path_to_image'], 'rb').read()
    user = row['user'].encode()
    emb = row['encodings']
    label = row[engagement_metric]

    tf_example = serialize_example(image_string,
                                   user,
                                   emb,
                                   label)
    writer.write(tf_example.SerializeToString())



#write tf test dataset
test_1face_file = 'final/test_1_32face.tfrecord'

with tf.io.TFRecordWriter(test_1face_file) as writer:
  for index, row in test_1face.iterrows():

    image_string = open(row['path_to_image'], 'rb').read()
    user = row['user'].encode()
    emb = row['encodings']
    label = row[engagement_metric]

    tf_example = serialize_example(image_string,
                                   user,
                                   emb,
                                   label)
    writer.write(tf_example.SerializeToString())
