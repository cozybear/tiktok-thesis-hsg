import pandas as pd
import os
import json
from langdetect import detect
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator, constants
import re
import numpy as np

import warnings
warnings.filterwarnings("ignore")


path = '/.../total'
path_test = '/.../metadata_test'



class language_detector:
    def __init__(self, path):
        self.path = path

    def Most_Common(self, lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]


    def language_detection(self):
        # for every folder in directory
        df = pd.DataFrame(columns=['user', 'language'])
        for folder in os.listdir(self.path):
            langs = []

            # if it is a directory
            if os.path.isdir(os.path.join(self.path, folder)):
                with open(os.path.join(self.path, folder, 'metadata/meta_2.json')) as jsonf:
                    data = json.load(jsonf)
                    for item in data:
                        description = item['video_stats']['description']

                        try:
                            lan = detect(str(description))
                            langs.append(lan)

                        except:
                            #no features in video description
                            pass

                if langs:
                    df = df.append({'user': folder, 'language': self.Most_Common(langs)}, ignore_index=True)
                else:
                    df = df.append({'user': folder, 'language': np.nan}, ignore_index=True)

            print(f'Done for {folder}')


        plt.style.use('ggplot')
        fig = plt.figure(clear=True)

        sns.countplot(x=df["language"])

        # save
        if not os.path.isdir('../languages'):
            os.makedirs('../languages')
        plt.savefig(f'languages/languages.png', dpi=400, bbox_inches="tight")

        plt.show()

        return df


languages = language_detector(path)
df_language = languages.language_detection()

#save languages
if not os.path.isdir('languages'):
    os.makedirs('languages')
df_language.to_csv('languages/languages.csv', index=False)


LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'he': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'or': 'odia',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'ug': 'uyghur',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
}

df = pd.read_csv('../languages/languages.csv')

df.language = df.language.map(LANGUAGES)



plt.style.use('ggplot')
fig = plt.figure(clear=True, figsize=(11, 8))
sns.countplot(y=df["language"], order=df["language"].value_counts().index)
plt.show()
fig.savefig(f'languages/languages.png', dpi=400, bbox_inches="tight")
