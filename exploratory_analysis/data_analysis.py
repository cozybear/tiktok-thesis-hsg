
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import pprint
import os
import json
import seaborn as sns
import pandas as pd
import numpy as np
import emoji
from collections import Counter
from datetime import datetime
from datetime import date
import regex as re
from wordcloud import WordCloud
import string
from matplotlib.font_manager import FontProperties
import stopwordsiso as stopwords
import json
import unicodedata
import emoji
import subprocess


import warnings
warnings.filterwarnings("ignore")

path = '/.../tiktokdata/total'
path_test = '/.../tiktokdata/metadata_test'

##helper functions

#save
def save_dataframe(df, name):
    if not os.path.isdir('../data_analysis'):
        os.makedirs('../data_analysis')
    df.to_csv(os.path.join('../data_analysis', f'{name}.csv'), index=False)

#create table from df
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax



# PER AUTHOR:


def account_settings():
    comments = []
    downloads = []
    duets = []
    stitches = []

    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)
                comment = data['author']['commentSetting']
                downlaod = data['author']['downloadSetting']
                duet = data['author']['duetSetting']
                stitch = data['author']['stitchSetting']

                comments.append(comment)
                downloads.append(downlaod)
                duets.append(duet)
                stitches.append(stitch)

    #comments = [w.replace(str(1), 'Enabled').replace(str(0), 'Disabled') for w in comments]
    #downloads = [w.replace(str(1), 'Enabled').replace(str(0), 'Disabled') for w in downloads]
    #duets = [w.replace(str(1), 'Enabled').replace(str(0), 'Disabled') for w in duets]
    #stitches = [w.replace(str(1), 'Enabled').replace(str(0), 'Disabled') for w in stitches]


    # Combine all words together
    count_comments = Counter(comments)
    percentages_comments = []
    for user, value in count_comments.items():
        percentages_comments.append(value)

    count_downloads = Counter(downloads)
    percentages_downloads = []
    for user, value in count_downloads.items():
        percentages_downloads.append(value)

    count_duets = Counter(duets)
    percentages_duets = []
    for user, value in count_duets.items():
        percentages_duets.append(value)

    count_stitches = Counter(stitches)
    percentages_stitches = []
    for user, value in count_stitches.items():
        percentages_stitches.append(value)

    #group = ['Disabled', 'Enabled']
    colors = ['firebrick', 'orange', 'lightblue']

    #explode = (0, 0.04)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.pie(percentages_comments, explode=(0, 0.1, 0.2),
            #labels=['No one', 'Friends', 'Everyone'],
            autopct=None, colors=colors, radius=1.1, pctdistance= 0.7, textprops={'fontsize': 7})
    plt.title('Comment Settings', fontsize=9)

    plt.subplot(2, 2, 2)
    plt.pie(percentages_downloads, explode=(0, 0.1),
            labels=['Disabled', 'Enabled'],  autopct='%1.1f%%', colors=['orange', 'lightblue'],
            radius=1.1, pctdistance= 0.7, textprops={'fontsize': 7})
    plt.title('Download Settings', fontsize=9)

    plt.subplot(2, 2, 3)
    plt.pie(percentages_duets, autopct='%1.1f%%', explode=(0, 0.1, 0.2),
            labels=['No one', 'Friends', 'Everyone'], colors=colors,
            radius=1.1, pctdistance= 0.7, textprops={'fontsize': 7})
    plt.title('Duet Settings', fontsize=9)

    plt.subplot(2, 2, 4)
    plt.pie(percentages_stitches, autopct='%1.1f%%', explode=(0, 0.1, 0.2),
            labels=['No one', 'Friends', 'Everyone'], colors=colors,
            radius=1.1, pctdistance= 0.7, textprops={'fontsize': 7})
    plt.title('Stitch Settings', fontsize=9)

    path_image = os.path.join('../data_analysis', 'account_settings.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#account_settings()



def no_faces():
    # for every folder in directory
    face_exists = 0
    no_face = 0
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder, 'videos', 'faces_cleaned')):
            face_exists += 1
        else:
            no_face += 1

    sizes = np.array([face_exists, no_face])

    #print(face_exists, no_face)

    labels = ['Faces detected', 'No Faces']
    explode = (0, 0.02)
    colors = ['lightblue', 'orange']

    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors=colors)
    #plt.title('Profiles containing Faces')
    path_image = os.path.join('../data_analysis', 'faces.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#no_faces()



def plot_following_distribution():
    following_counts = []
    author = []
    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)
                following_count = data['authorStats']['followingCount']
                following_counts.append(following_count)
                author.append(folder)


    df = pd.DataFrame({'following': following_counts, 'user': author})
    #add +1 to be able to use log scale because log(0) = -infinity
    df['following'] += 1

    sns.displot(data=df.following, kde=True,  log_scale=True, legend=False)

    #remove spacing
    plt.margins(x=0)
    #label
    plt.xlabel('Following (log Scale)')
    path_image = os.path.join('../data_analysis', 'following.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_following = plot_following_distribution()
#save_dataframe(df_following, 'following')




def plot_follower_distribution():
    follower_counts = []
    author = []
    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)
                follower_count = data['authorStats']['followerCount']
                follower_counts.append(follower_count)
                author.append(folder)


    df = pd.DataFrame({'followers': follower_counts, 'user': author})
    #add +1 to be able to use log scale because log(0) = -infinity
    df['followers'] += 1

    sns.displot(data=df.followers, kde=True,  log_scale=True, legend=False)

    #remove spacing
    plt.margins(x=0)
    #label
    plt.xlabel('Followers (log Scale)')
    path_image = os.path.join('../data_analysis', 'followers.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_followers = plot_follower_distribution()
#save_dataframe(df_followers, 'followers')




def plot_total_likes_distribution():
    likes_counts = []
    author = []

    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)
                likes_count = data['authorStats']['heartCount']
                likes_counts.append(likes_count)
                author.append(folder)

    df = pd.DataFrame({'user': author, 'profile_likes': likes_counts})
    # add 1 to be able to use log scale because log(0) = -infinity
    df['profile_likes'] += 1

    sns.displot(data=df.profile_likes, kde=True, log_scale=True, legend=False)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Total Profile Likes (log Scale)')
    path_image = os.path.join('../data_analysis', 'likes.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_total_likes_distribution()


def plot_exact_total_likes_distribution():
    #digg_counts per author videos
    df = pd.DataFrame(columns=['user', 'likes'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            likes_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    likes = item['video_stats']['stats']['diggCount']
                    likes_counter += likes




            df = df.append({'user': folder, 'likes': float(likes_counter)}, ignore_index=True)
    # add 1 to be able to use log scale because log(0) = -infinity
    df['likes'] += 1
    #plot
    sns.displot(data=df.likes, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Exact Total Profile Likes (log scale)')
    path_image = os.path.join('../data_analysis', 'exact_likes.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_exact_likes = plot_exact_total_likes_distribution()
#save_dataframe(df_exact_likes, 'exact_likes')


def plot_plays_distribution():
    #playCount
    df = pd.DataFrame(columns=['user', 'total_plays', 'average_plays'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            plays = 0
            video_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    video_counter += 1
                    video_playcount = item['video_stats']['stats']['playCount']

                    plays += video_playcount


            df = df.append({'user': folder, 'total_plays': float(plays),
                            'average_plays': float(round((plays / video_counter), 2))}, ignore_index=True)

    df['total_plays'] += 1
    df['average_plays'] += 1

    # plot
    sns.displot(data=df.total_plays, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Total Channel Video Plays')
    path_image = os.path.join('../data_analysis', 'total_plays.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    # plot
    sns.displot(data=df.average_plays, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Mean Channel Video Plays')
    path_image = os.path.join('../data_analysis', 'average_plays.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#plot_plays_distribution()



def plot_likes_distribution():
    df = pd.DataFrame(columns=['user', 'total_likes', 'average_likes'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            likes = 0
            video_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    video_counter += 1
                    video_likecount = item['video_stats']['stats']['diggCount']

                    likes += video_likecount


            df = df.append({'user': folder, 'total_likes': float(likes),
                            'average_likes': float(round((likes / video_counter), 2))}, ignore_index=True)

    df['total_likes'] += 1
    df['average_likes'] += 1

    # plot
    sns.displot(data=df.total_likes, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Total Channel Video Likes')
    path_image = os.path.join('../data_analysis', 'total_likes.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    # plot
    sns.displot(data=df.average_likes, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Mean Channel Video Likes')
    path_image = os.path.join('../data_analysis', 'average_likes.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#plot_likes_distribution()



def plot_shares_distribution():
    #shareCount
    df = pd.DataFrame(columns=['user', 'total_shares', 'average_shares'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            shares = 0
            video_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    video_counter += 1
                    video_sharecount = item['video_stats']['stats']['shareCount']

                    shares += video_sharecount

            df = df.append({'user': folder, 'total_shares': float(shares),
                            'average_shares': float(round((shares / video_counter), 2))}, ignore_index=True)

    df['total_shares'] += 1
    df['average_shares'] += 1

    # plot
    sns.displot(data=df.total_shares, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Total Channel Video Shares')
    path_image = os.path.join('../data_analysis', 'total_shares.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    # plot
    sns.displot(data=df.average_shares, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Mean Channel Video Shares')
    path_image = os.path.join('../data_analysis', 'average_shares.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#plot_shares_distribution()



def plot_comments_distribution():
    #commentCount
    df = pd.DataFrame(columns=['user', 'total_comments', 'average_comments'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            comments = 0
            video_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    video_counter += 1
                    video_commentscount = item['video_stats']['stats']['commentCount']

                    comments += video_commentscount

            df = df.append({'user': folder, 'total_comments': float(comments),
                            'average_comments': float(round((comments / video_counter), 2))}, ignore_index=True)
    df.total_comments += 1
    df.average_comments += 1
    # plot
    sns.displot(data=df.total_comments, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Total Channel Video Comments')
    path_image = os.path.join('../data_analysis', 'total_comments.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    # plot
    sns.displot(data=df.average_comments, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Mean Channel Video Comments')
    path_image = os.path.join('../data_analysis', 'average_comments.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#plot_comments_distribution()


def plot_playtime_distribution():
    #playCount x duration
    df = pd.DataFrame(columns=['user', 'total_playtime_minutes', 'average_playtime_minutes'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            playtime = 0
            video_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    video_counter +=1
                    video_playcount = item['video_stats']['stats']['playCount']
                    video_duration = item['video_stats']['video_duration']

                    #per video playtime in seconds
                    video_playtime = video_playcount*video_duration
                    playtime += video_playtime


            
            df = df.append({'user': folder, 'total_playtime_minutes': float(round((playtime/60), 2)),
                            'average_playtime_seconds': float(round((playtime/video_counter), 2))}, ignore_index=True)
    df.total_playtime_minutes += 1
    df.average_playtime_seconds += 1

    #plot
    sns.displot(data=df.total_playtime_minutes, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Channel Total Playtime (in minutes)')
    path_image = os.path.join('../data_analysis', 'total_playtime.png')
    plt.savefig(path_image, dpi=400)
    plt.close()

    # plot
    sns.displot(data=df.average_playtime_seconds, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Average Video Playtime per Channel (in Seconds)')
    path_image = os.path.join('../data_analysis', 'average_playtime.png')
    plt.savefig(path_image, dpi=400)
    plt.close()

    return df

#plot_playtime_distribution()


def plot_digg_count():
    #
    digg_counter = []
    author = []
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):

            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)

                digg_count = data['authorStats']['diggCount']
                digg_counter.append(digg_count)
                author.append(folder)
    df = pd.DataFrame({'user': author, 'author_likes': digg_counter})

    df.author_likes += 1
    # plot
    sns.displot(data=df.author_likes, kde=True, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Author Likes (on other Posts)')
    path_image = os.path.join('../data_analysis', 'author_diggs.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_diggs = plot_digg_count()
#save_dataframe(df_diggs, 'author_diggs')



def plot_engagements_absolute_per_author():
    df = pd.DataFrame(columns=['user', 'average_video_engagements', 'total_engagements'])

    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            engagements = 1
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    # if the video was played at least once

                    posts_counter += 1

                    shares = item['video_stats']['stats']['shareCount']
                    likes = item['video_stats']['stats']['diggCount']
                    comments = item['video_stats']['stats']['commentCount']

                    video_engagements = sum([shares, likes, comments])
                    engagements += video_engagements
            try:
                df = df.append({'user': folder,
                                'average_video_engagements': float(engagements / posts_counter),
                                'total_engagements': float(engagements)}, ignore_index=True)
            except:
                # zero plays --> engagemnet = 0
                continue
                #df = df.append(
                    #{'user': folder, 'average_video_engagements': np.nan, 'total_engagements': np.nan}, ignore_index=True)

    sns.displot(data=df.average_video_engagements, log_scale=True)

    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Average Engagements per User Video')
    #plt.xlim(0, 40)
    path_image = os.path.join('../data_analysis', 'average_engagements.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()


    sns.displot(data=df.total_engagements, log_scale=True)

    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Total Engagements per User')
    # plt.xlim(0, 40)
    path_image = os.path.join('../data_analysis', 'total_engagements.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_engagements = plot_engagements_absolute_per_author()
#save_dataframe(df_engagements, 'engagements_absolute')


def average_engagements_detailed():
    # for every folder in directory
    share_rate = []
    like_rate = []
    comment_rate = []
    user = []
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            total_likes = []
            total_shares = []
            total_comments = []
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    posts_counter += 1
                    plays = item['video_stats']['stats']['playCount']

                    # if the video was played at least once
                    if plays > 0:


                        shares = item['video_stats']['stats']['shareCount']
                        likes = item['video_stats']['stats']['diggCount']
                        comments = item['video_stats']['stats']['commentCount']

                        total_likes.append(likes/plays)
                        total_shares.append(shares/plays)
                        total_comments.append(comments/plays)

            share_rate.append(sum(total_shares)/posts_counter)
            like_rate.append(sum(total_likes)/posts_counter)
            comment_rate.append(sum(total_comments)/posts_counter)
            user.append(folder)


    df = pd.DataFrame({'user': user,
                              'average_likes_rate': like_rate,
                              'average_shares_rate': share_rate,
                              'average_comments_rate': comment_rate
                              })
    return df

#df_engagements = average_engagements_detailed()
#save_dataframe(df_engagements, 'av_engagements_detailed')







def plot_engagementrate_by_impressions_distribution():
    #by impressions
    # per video (shares,likes,comments)/plays/# videos
    df = pd.DataFrame(columns=['user', 'average_engagement_rate'])
    total_engagements = []

    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            engagements = []
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    plays = item['video_stats']['stats']['playCount']

                    #if the video was played at least once
                    if plays > 0:
                        posts_counter += 1

                        shares = item['video_stats']['stats']['shareCount']
                        likes = item['video_stats']['stats']['diggCount']
                        comments = item['video_stats']['stats']['commentCount']

                        total_engagement = sum([shares, likes, comments])
                        per_video_engagement_rate = total_engagement/plays*100
                        engagements.append(per_video_engagement_rate)
                        total_engagements.append(per_video_engagement_rate)
            try:
                df = df.append({'user': folder, 'average_engagement_rate': round((sum(engagements)/posts_counter), 4)},
                               ignore_index=True)
            except:
                #zero plays --> engagemnet = 0
                df = df.append(
                    {'user': folder, 'average_engagement_rate': np.nan}, ignore_index=True)



    df_videos = pd.DataFrame({'total_engagement_rates': total_engagements})

    #df.sort_values(by='average_engagement_rate', ascending=False)
    #plt.figure(figsize=(11, 8))
    """
    sns.displot(data=df.average_engagement_rate)

    # remove spacing
    plt.margins(x=0)
    #plt.text(30, 500, f'Mean Channel\nEngagement Rate:\n{round(df.average_engagement_rate.mean(), 2)}%')
    plt.xlabel('Channel Engagement Rates by Impressions (in %)')
    plt.xlim(0, 40)
    path_image = os.path.join('data_analysis', 'channel_engagement_rates.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    #largest values
    print(df.nlargest(30, 'average_engagement_rate'))
    """
    #plt.figure(figsize=(11, 8))
    sns.displot(data=df_videos.total_engagement_rates)

    # remove spacing
    plt.margins(x=0)
    #plt.text(30, 500, f'Mean Video\nEngagement Rate:\n{round(df_videos.total_engagement_rates.mean(), 2)}%')
    plt.xlabel('Video Engagement Rates by Impressions (in %)')
    plt.xlim(0, 50)
    path_image = os.path.join('../data_analysis', 'video_engagement_rates.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    #largest values
    print(df_videos.nlargest(30, 'total_engagement_rates'))


    return df_videos

#df_e= plot_engagementrate_by_impressions_distribution()
#save_dataframe(df_e, 'total_engagement_rates')
#save_dataframe(df_videos, 'video_engagement_rates')
#print('Done for engagements')




def plot_amount_posted_videos():
    posted_videos = []
    author = []
    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)
                video_count = data['authorStats']['videoCount']
                posted_videos.append(video_count)
                author.append(folder)
    df = pd.DataFrame({'user': author, 'posted_videos': posted_videos})

    df.posted_videos += 1

    sns.displot(data=df.posted_videos, log_scale=True, height=5, aspect=1.7)
    # remove spacing

    plt.margins(x=0)
    plt.xlabel('Number of videos posted')
    path_image = os.path.join('../data_analysis', 'total_videos.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_amount_posted_videos()



"""
def plot_engagement_byimpressions_over_time():
"""



def plot_verified_accounts():
    df = pd.DataFrame(columns=['user', 'verified'])
    verified_accounts = []
    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)
                verified = str(data['author']['verified']).lower()
                verified_accounts.append(verified)
                #append to df
                df = df.append({'user': folder, 'verified': verified}, ignore_index=True)

    verified_accounts = [w.replace('true'
                                   , 'Verified').replace('false', 'Un-Verified') for w in verified_accounts]

    # Combine all words together
    count = Counter(verified_accounts)
    percentages = []
    for user, value in count.items():
        percentages.append(value)
    labels = ['Un-Verified', 'Verified']
    explode = (0, 0.1)

    colors = ['lightblue', 'orange']
    plt.pie(percentages, explode=explode, labels=labels, autopct='%1.1f%%', colors=colors)

    #plt.title('Account Verification Status')

    path_image = os.path.join('../data_analysis', 'is_verified_pie.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")

    plt.close()

    return df


#df_verified = plot_verified_accounts()
#save_dataframe(df_verified, "verified")



def get_emojis_signature():
    def extract_emojis(s):
      return ''.join(c for c in s if c in emoji.EMOJI_DATA)

    emojis = []

    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/author.json')) as jsonf:
                data = json.load(jsonf)
                signature = data['author']['signature']
                signature = extract_emojis(signature)
                emojis.extend(signature)
    c = Counter(emojis)
    #numer of uniqze emojis in list
    print(len(set(emojis)))
    # 30 most common emojis
    df = pd.DataFrame(c.most_common(30), columns=['Emoji', 'Value Count'])

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()

    path_image = os.path.join('../data_analysis', 'emojis.png')
    fig.savefig(path_image, dpi=400, bbox_inches="tight")

#get_emojis_signature()


def plot_channels_age():
    df = pd.DataFrame(columns=['user', 'channel_age'])
    today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

    # for every folder in directory
    for folder in os.listdir(path):
        timestamps = []

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    post_time = item['video_stats']['create_time']
                    post_time = datetime.utcfromtimestamp(post_time).strftime('%Y-%m-%d %H:%M:%S')
                    timestamps.append(post_time)

            if timestamps:
                first_post = pd.to_datetime(min(timestamps))
                difference_in_days = (today - first_post).days

                df = df.append({'user': folder, 'channel_age': difference_in_days, 'channel_date': first_post}, ignore_index=True)


    sns.displot(data=df.channel_date)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Channel Age')
    path_image = os.path.join('../data_analysis', 'channel_age.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_channel_age = plot_channels_age()
#save_dataframe(df_channel_age, "channel_age")



def plot_post_frequency():
    #divide number of posts by channel age in days
    df = pd.DataFrame(columns=['user', 'post_frequency'])
    today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            timestamps = []
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    posts_counter += 1
                    post_time = item['video_stats']['create_time']
                    post_time = datetime.utcfromtimestamp(post_time).strftime('%Y-%m-%d %H:%M:%S')
                    timestamps.append(post_time)

            if timestamps:
                first_post = pd.to_datetime(min(timestamps))
                total_weeks = (today - first_post).days/7
                df = df.append({'user': folder, 'post_frequency': round((posts_counter/total_weeks), 2)}, ignore_index=True)

    print(df.loc[df['post_frequency'].idxmax()])
    print(df.loc[df['post_frequency'].idxmin()])

    sns.displot(data=df.post_frequency)
    # remove spacing
    plt.margins(x=0)
    plt.xlim(0, 5)
    plt.xlabel('Mean Posts (per Week)')
    path_image = os.path.join('../data_analysis', 'post_frequency.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_post_frequency = plot_post_frequency()
#save_dataframe(df_post_frequency, "post_frequency_daily")



def plot_sticker_count():
    #average stickers used per user video
    df = pd.DataFrame(columns=['user', 'average_sticker_count'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            sticker_counter = 0
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    posts_counter += 1
                    stickers = item['video_stats']['stickers_text']
                    sticker_text = ''.join(str(x) for x in stickers).replace('[', '').replace(']', '')

                    if sticker_text == "":
                        sticker_counter += 0

                    else:
                        sticker_counter += len(sticker_text.split(','))



            df = df.append({'user': folder, 'average_sticker_count': round((sticker_counter/posts_counter*100), 2)}, ignore_index=True)


    sns.displot(data=df.average_sticker_count)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Average # of Stickers per Channel Video')
    plt.xlim(0, 1000)
    path_image = os.path.join('../data_analysis', 'sticker_count.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df


#df_sticker_count = plot_sticker_count()
#save_dataframe(df_sticker_count, "sticker_count")



def plot_sticker_ratio():
    #stickers used on user video (percentage)
    df = pd.DataFrame(columns=['user', 'sticker_ratio'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            sticker_counter = 0
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    posts_counter += 1
                    stickers = item['video_stats']['stickers_text']
                    sticker_text = ''.join(str(x) for x in stickers).replace('[', '').replace(']', '')

                    if sticker_text == "":
                        sticker_counter += 0

                    else:
                        sticker_counter += 1

            df = df.append({'user': folder, 'sticker_ratio': round((sticker_counter / posts_counter * 100), 2)}, ignore_index=True)


    sns.displot(data=df.sticker_ratio, kde=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Percentage of User Videos containing Stickers ')
    path_image = os.path.join('../data_analysis', 'sticker_ratio.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#plot_sticker_ratio()




def plot_description_ratio():
    #description used on user video (binary)
    df = pd.DataFrame(columns=['user', 'description_ratio'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            description_counter = 0
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    posts_counter += 1
                    description = item['video_stats']['description']

                    if description == "":
                        description_counter += 0

                    else:
                        description_counter += 1

            df = df.append({'user': folder, 'description_ratio': round((description_counter / posts_counter*100), 2)}, ignore_index=True)


    sns.displot(data=df.description_ratio)
    # remove spacing
    plt.margins(x=0)
    plt.xlim(50, 100)
    plt.xlabel('Percentage of User Videos with Description')
    path_image = os.path.join('../data_analysis', 'description_ratio.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#df_is_description = plot_description_ratio()
#save_dataframe(df_is_description, 'description_ratio')



def plot_hashtag_ratio():
    #hastags used per user video
    #description used on user video (binary)
    df = pd.DataFrame(columns=['user', 'hashtag_ratio'])
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            hashtag_counter = 0
            posts_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    posts_counter += 1
                    hashtags = item['video_stats']['hashtags']
                    hashtags = ' '.join(str(x) for x in hashtags).replace('[', '').replace(']', '')

                    if hashtags == "":
                        hashtag_counter += 0

                    else:
                        hashtag_counter += 1

            df = df.append({'user': folder, 'hashtag_ratio': round((hashtag_counter / posts_counter * 100), 2)}, ignore_index=True)


    sns.displot(data=df.hashtag_ratio, kde=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Percentage of User Videos containing Hashtags')
    path_image = os.path.join('../data_analysis', 'hashtag_ratio.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    return df

#plot_hashtag_ratio()



def plot_hashtag_count():
    #average hashtags per user video
    df = pd.DataFrame(columns=['user', 'hashtag_frequency'])

    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            video_counter = 0
            hashtag_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    hashtags = item['video_stats']['hashtags']
                    #filter empty hashtags from list
                    hashtags = list(filter(None, hashtags))

                    hashtag_counter += len(hashtags)
                    video_counter += 1

            df = df.append({'user': folder, 'hashtag_frequency': round((hashtag_counter / video_counter), 2)}, ignore_index=True)


    sns.displot(data=df.hashtag_frequency)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Average Amount of Video Hashtags per User')
    plt.xlim(0, 30)
    path_image = os.path.join('../data_analysis', 'hashtag_frequency.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()


    #max value
    #print(df.loc[df['hashtag_frequency'].idxmax()])

    return df

#df_hashtag_count = plot_hashtag_count()
#save_dataframe(df_hashtag_count, "hashtag_frequency")



def plot_duration_count():
    df = pd.DataFrame(columns=['user', 'average_duration'])

    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            duration_total = 0
            video_counter = 0
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    duration = item['video_stats']['video_duration']
                    duration_total += duration
                    video_counter += 1


            df = df.append({'user': folder, 'average_duration': round((duration_total / video_counter), 2)}, ignore_index=True)



    sns.displot(data=df.average_duration)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Average Video Duration per Author (in Seconds)')
    plt.xlim(0, 120)
    path_image = os.path.join('../data_analysis', 'average_video_duration.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()



    #print(df.loc[df['average_duration'].idxmax()])
    return df

#df_duration_count = plot_duration_count()
#save_dataframe(df_duration_count, "average_video_duration")




##############################
#PER VIDEO:

#create df for punchcard
def get_dataframe(pathname):
    timestamps = []
    folders = []
    # for every folder in directory
    for folder in os.listdir(pathname):
        # if it is a directory
        if os.path.isdir(os.path.join(pathname, folder)):
            with open(os.path.join(pathname, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    post_time = item['video_stats']['create_time']
                    post_time = datetime.utcfromtimestamp(post_time).strftime('%Y-%m-%d %H:%M:%S')
                    timestamps.append(post_time)
                    folders.append(folder)


    df = pd.DataFrame({'user': folders, 'timestamp': timestamps, 'body': 1})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    return df

##########################


def is_sticker():
    # stickers used per video
    #stickers used on user video (binary)
    sticker_counter = 0
    non_sticker_counter = 0
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):

            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    stickers = item['video_stats']['stickers_text']
                    sticker_text = ''.join(str(x) for x in stickers).replace('[', '').replace(']', '')

                    if sticker_text == "":
                        non_sticker_counter += 1

                    else:
                        sticker_counter += 1

    group = ['Stickers', 'No Stickers']
    plt.bar(group[0], sticker_counter, color='lightblue')
    plt.bar(group[1], non_sticker_counter, color='orange')

    plt.ylabel('Number of Videos')
    path_image = os.path.join('../data_analysis', 'is_sticker.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()



    explode = (0, 0.1)
    sizes = np.array([sticker_counter, non_sticker_counter])
    colors = ['lightblue', 'orange']
    plt.pie(sizes, explode=explode, labels=group, autopct='%1.1f%%',  colors=colors)

    #plt.title('Stickers used in Video')

    path_image = os.path.join('../data_analysis', 'is_sticker_pie.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")

    plt.close()

#is_sticker()



def plot_video_duration():
    ids = []
    durations = []
    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    id = item['video_id']
                    video_duration = item['video_stats']['video_duration']
                    if video_duration == 0:
                        video_duration += 1

                    ids.append(id)
                    durations.append(video_duration)

    df = pd.DataFrame({'id': ids, 'duration': durations})

    # plot
    #sns.kdeplot(data=df.duration, log_scale=True, bw_adjust=4)
    sns.displot(data=df.duration, kde=True, height=5, aspect=2)

    # remove spacing
    plt.margins(x=0)
    plt.xlim(0, 80)
    plt.xlabel('Video Durations (seconds)')
    path_image = os.path.join('../data_analysis', 'video_duration.png')
    plt.savefig(path_image, dpi=400)
    plt.close()

    return df


#df_duration = plot_video_duration()
#save_dataframe(df_duration, 'video_duration')




def is_description():
    #description used per video yes/no
    description_counter = 0
    non_description_counter = 0
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):

            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    description = item['video_stats']['description']

                    if description == "":
                        non_description_counter += 1

                    else:
                        description_counter += 1

    group = ['Description', 'No Description']
    plt.bar(group[0], description_counter, color='lightblue')
    plt.bar(group[1], non_description_counter, color='orange')

    plt.xlabel('Group')
    plt.ylabel('Number of Videos')
    path_image = os.path.join('../data_analysis', 'is_description.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()



    explode = (0, 0.1)
    sizes = np.array([description_counter, non_description_counter])
    colors = ['lightblue', 'orange']
    plt.pie(sizes, explode=explode, labels=group, autopct='%1.1f%%', colors=colors)

    #plt.title('Description used in Video')

    path_image = os.path.join('../data_analysis', 'is_description_pie.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")

    plt.close()

#is_description()




def is_hashtag():
    # description used per video yes/no
    hashtag_counter = 0
    non_hashtag_counter = 0
    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):

            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    hashtags = item['video_stats']['hashtags']
                    hashtags = ' '.join(str(x) for x in hashtags).replace('[', '').replace(']', '')

                    if hashtags == "":
                        non_hashtag_counter += 1

                    else:
                        hashtag_counter += 1

    group = ['Hashtags', 'No Hashtags']
    plt.bar(group[0], hashtag_counter, color='lightblue')
    plt.bar(group[1], non_hashtag_counter, color='orange')

    plt.xlabel('Group')
    plt.ylabel('Number of Videos')
    path_image = os.path.join('../data_analysis', 'is_hashtag.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    explode = (0, 0.1)
    sizes = np.array([hashtag_counter, non_hashtag_counter])
    colors = ['lightblue', 'orange']
    plt.pie(sizes, explode=explode, labels=group, autopct='%1.1f%%', colors=colors)

    #plt.title('Hashtags used in Video')

    path_image = os.path.join('../data_analysis', 'is_hashtag_pie.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()


#is_hashtag()



def plot_share_enabled():
    shares = []

    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    share = str(item['video_stats']['shareEnabled']).lower()
                    shares.append(share)

    shares = [w.replace('true', 'Enabled').replace('false', 'Disabled') for w in shares]
    # Combine all words together
    count = Counter(shares)
    percentages = []
    for user, value in count.items():
        percentages.append(value)

    colors = ['lightblue', 'orange']

    sns.displot(data=shares, palette=colors)
    # remove spacing
    plt.margins(x=0)
    #plt.title('Total Share Settings')
    plt.ylabel('Count')
    plt.xlabel('Class')
    path_image = os.path.join('../data_analysis', 'is_share.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

    group = ['Enabled']

    #explode = (0, 0.1)
    plt.pie(percentages, labels=group, autopct='%1.1f%%', colors=colors)

    #plt.title('Share Settings')

    path_image = os.path.join('../data_analysis', 'is_share_pie.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_share_enabled()
# #all videos seem to have shareEnabled = True



def plot_duet_enabled():
    duets = []

    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    duet = str(item['video_stats']['duetEnabled']).lower()
                    duets.append(duet)

    duets = [w.replace('true', 'Enabled').replace('false', 'Disabled') for w in duets]

    # Combine all words together
    count = Counter(duets)
    percentages = []
    for user, value in count.items():
        percentages.append(value)

    colors = ['lightblue', 'orange']
    group = ['Enabled', 'Disabled']


    sns.displot(data=duets, palette=sns.color_palette(colors, len(colors)))
    # remove spacing
    plt.margins(x=0)
    #plt.title('Total Duet Settings')
    plt.ylabel('Count')
    plt.xlabel('Class')
    path_image = os.path.join('../data_analysis', 'is_duet.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()


    explode = (0, 0.1)
    plt.pie(percentages, explode=explode, labels=group, autopct='%1.1f%%', colors=colors)

    #plt.title('Duet Settings')

    path_image = os.path.join('../data_analysis', 'is_duet_pie.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_duet_enabled()


def plot_stitch_enabled():
    stitches = []

    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    stitch = str(item['video_stats']['stitchEnabled']).lower()
                    stitches.append(stitch)

    stitches = [w.replace('true', 'Enabled').replace('false', 'Disabled') for w in stitches]

    # Combine all words together
    count = Counter(stitches)
    percentages = []
    for user, value in count.items():
        percentages.append(value)

    colors = ['lightblue', 'orange']
    group = ['Enabled', 'Disabled']

    sns.displot(data=stitches, palette=sns.color_palette(colors, len(colors)))
    # remove spacing
    plt.margins(x=0)
    #plt.title('Total Stitch Settings')
    plt.ylabel('Count')
    plt.xlabel('Class')
    path_image = os.path.join('../data_analysis', 'is_stitch.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()



    explode = (0, 0.1)
    plt.pie(percentages, explode=explode, labels=group, autopct='%1.1f%%', colors=colors)

    #plt.title('Stitch Settings')

    path_image = os.path.join('../data_analysis', 'is_stitch_pie.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_stitch_enabled()


def get_emojis_videos():
    def extract_emojis(s):
      return ''.join(c for c in s if c in emoji.EMOJI_DATA)

    emojis = []
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    description = item['video_stats']['description']

                    description = extract_emojis(description)
                    emojis.extend(description)



    c = Counter(emojis)
    #numer of unique emojis in list
    print(len(set(emojis)))
    # 30 most common emojis
    pprint.pprint(c.most_common(30))
    #30 least common emojis
    #pprint.pprint(c.most_common()[:-30 - 1:-1])

    emojis, counts = zip(*c.most_common(30))

    df = pd.DataFrame({'Emoji': emojis,
                       'Count': counts
                       })
    return df

#emojis = get_emojis_videos()
#save_dataframe(emojis, 'emojis_table')



def hashtag_distribution():
    #average hashtags per user video
    total_hashtag_counter = 0
    hashtags_list = []
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):

            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    hashtag_counter = 0

                    hashtags = item['video_stats']['hashtags']
                    if hashtags != '[]':
                        #filter empty hashtags from list
                        hashtags = list(filter(None, hashtags))
                        hashtags_length = len(hashtags)
                        hashtag_counter += hashtags_length
                        total_hashtag_counter += hashtag_counter
                        hashtags_list.append(hashtag_counter)
                    else:
                        hashtags_list.append(0)

    print(max(hashtags_list))

    counter = Counter(hashtags_list)
    lists = sorted(counter.items())  # sorted by key, return a list of tuples


    print(lists)

    x, y = zip(*lists)  # unpack a list of pairs into two tuples




    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10,6))
    plt.bar(x, y)
    #plt.text(.95, .95,
        #f'Total Hashtags in Dataset:\n{total_hashtag_counter}', horizontalalignment='right',
        #verticalalignment='top',
        #transform = ax.transAxes)
    #fig.tight_layout()

    plt.xlabel('Hashtags')
    plt.ylabel('Count')
    #plt.xticks(np.arange(len(x)), np.arange(1, len(x) + 1))

    plt.xlim(-1, 25)
    path_image = os.path.join('../data_analysis', 'hashtag_distribution.png')
    fig.savefig(path_image, dpi=400, bbox_inches="tight")
    #plt.show()
    plt.close()

#hashtag_distribution()



def get_hashtags():
    total_hashtags = []

    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    hashtags = item['video_stats']['hashtags']
                    total_hashtags.extend(hashtags)

    c = Counter(total_hashtags)
    # numer of unique hashtags in list
    print(len(set(total_hashtags)))
    # 50 most common hashtags
    pprint.pprint(c.most_common(50))
    # 30 least common hashtags
    pprint.pprint(c.most_common()[:-30 - 1:-1])

    hashtags, counts = zip(*c.most_common(31))
    df = pd.DataFrame({'Hashtag': hashtags,
                       'Count': counts
                       }).replace('', np.nan).dropna(subset=['Hashtag'])
    return df

#hashtags = get_hashtags()
#save_dataframe(hashtags, 'hashtags_table')


def get_sounds():
    total_sounds = []
    ids = []
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                #for video in meta
                for item in data:
                    #get music title
                    sound = str(list(item['video_stats']['music'][1].values())[0]).title()
                    artist = str(list(item['video_stats']['music'][0].values())[0]).title()
                    # extend to list
                    total_sounds.append(sound +' '+ artist)
                    id = item['video_id']
                    ids.append(id)
    df = pd.DataFrame({'id': ids, 'sound': total_sounds
                       }).replace('', np.nan).dropna(subset=['sound'])
    """
    c = Counter(total_sounds)
    # number of unique sounds in list
    print(len(set(total_sounds)))
    # 50 most common sounds
    pprint.pprint(c.most_common(50))

    song, counts = zip(*c.most_common(31))
    sounds, artists = zip(*song)
    df = pd.DataFrame({'Title': sounds, 'Artist': artists,
                       'Count': counts
                       }).replace('', np.nan).dropna(subset=['Title'])
    """
    return df

#sounds = get_sounds()
#save_dataframe(sounds, 'sounds_table_id')



def get_sound_authors():
    total_sound_author = []

    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                # for video in meta
                for item in data:

                    # get music title author
                    sound_author = str(list(item["video_stats"]["music"][0].values())[0])
                    #sound_author = sound_author.replace('{}', '')
                    print(sound_author)
                    # extend to list
                    total_sound_author.append(sound_author)
    print(total_sound_author)
    c = Counter(total_sound_author)
    # numer of unique sound author in list
    print(len(set(total_sound_author)))
    # 50 most common sound author
    pprint.pprint(c.most_common(50))

#get_sound_authors()



def plot_playcount_distribution():
    play_counts = []
    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    plays = item['video_stats']['stats']['playCount']
                    play_counts.append(plays)

    df = pd.DataFrame({'plays': play_counts})
    # add 1 to be able to use log scale because log(0) = -infinity
    df['plays'] += 1

    sns.displot(data=df, kde=True, log_scale=True, legend=False)
    plt.xlabel('Video Plays (log scale)')
    #plt.xlim(0, 10000000)
    plt.show()
    plt.close()

#plot_playcount_distribution()



def plot_comments_stats():
    comment_counts = []

    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    comment_count = item['video_stats']['stats']['commentCount']
                    comment_counts.append(comment_count)



    #df = pd.DataFrame({'comments': comment_counts})

    # add 1 to be able to use log scale because log(0) = -infinity
    #df['comments'] += 1

    zeros = comment_counts.count(0)
    non_zeros = sum(x > 0 for x in comment_counts)
    percentages_comments = [non_zeros, zeros]
    # plot
    #sns.displot(data=df.comments, kde=True, log_scale=True)
    # remove spacing
    #plt.margins(x=0)
    #plt.xlabel('Video Comments')
    #path_image = os.path.join('data_analysis', 'video_comment_stats.png')
    #plt.savefig(path_image, dpi=400, bbox_inches="tight")
    #plt.close()
    colors = ['lightblue', 'orange']
    plt.pie(percentages_comments, explode=(0, 0.1), autopct='%1.1f%%',
            labels=['Comments', 'No comments'], colors=colors)


    path_image = os.path.join('../data_analysis', 'video_comment_pie.png')

    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_comments_stats()

def plot_comments1_stats():
    likes_counts = []


    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    likes_count = item['video_stats']['stats']['commentCount']
                    likes_counts.append(likes_count)



    df = pd.DataFrame({'comments': likes_counts
                       })

    # add 1 to be able to use log scale because log(0) = -infinity
    df['comments'] += 1

    # plot
    sns.displot(data=df.comments, kde=False, bins=80, log_scale=True, height=5, aspect=1.5, color='olive')
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Video Comments')
    plt.xlim(0, 100000)
    path_image = os.path.join('../data_analysis', 'video_comment_stats.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_comments1_stats()


def plot_likes_stats():
    likes_counts = []


    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    likes_count = item['video_stats']['stats']['diggCount']
                    likes_counts.append(likes_count)



    df = pd.DataFrame({'likes': likes_counts
                       })

    # add 1 to be able to use log scale because log(0) = -infinity
    df['likes'] += 1

    # plot
    sns.displot(data=df.likes, kde=False, log_scale=True, bins=80, height=5, aspect=1.5, color='orchid')
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Video Likes')
    path_image = os.path.join('../data_analysis', 'video_likes_stats.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_likes_stats()



def plot_plays_stats():
    play_counts = []


    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    play_count = item['video_stats']['stats']['playCount']
                    play_counts.append(play_count)




    df = pd.DataFrame({'plays': play_counts,
                       })

    # add 1 to be able to use log scale because log(0) = -infinity
    df['plays'] += 1
    #fig = plt.figure(figsize=(10, 5))
    # plot
    sns.displot(data=df.plays, kde=False, log_scale=True, bins=80, height=5, aspect=1.5, color='lightblue')
    #ax = sns.kdeplot(data=df.plays, bw_adjust=.25, log_scale=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Video Plays')
    path_image = os.path.join('../data_analysis', 'video_play_stats.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_plays_stats()



def plot_shares_stats():
    share_counts = []

    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    share_count = str(item['video_stats']['shareEnabled']).lower()
                    share_counts.append(share_count)

    shares = [w.replace('true', 'Enabled').replace('false', 'Disabled') for w in share_counts]
    # Combine all words together
    count = Counter(shares)
    percentages = []
    for user, value in count.items():
        percentages.append(value)

    """
    df = pd.DataFrame({'shares': share_counts
                       })

    # add 1 to be able to use log scale because log(0) = -infinity
    df['shares'] += 1

    # plot
    sns.displot(data=df.shares, kde=True)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Video Shares')
    path_image = os.path.join('data_analysis', 'video_shares_stats.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()
    """

    colors = ['lightblue', 'orange']
    plt.pie(percentages, autopct='%1.1f%%',
            labels=['Enabled'], colors=colors)

    path_image = os.path.join('../data_analysis', 'video_share_pie.png')

    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()


#plot_shares_stats()


def plot_shares1_stats():
    likes_counts = []


    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    likes_count = item['video_stats']['stats']['shareCount']
                    likes_counts.append(likes_count)



    df = pd.DataFrame({'likes': likes_counts
                       })

    # add 1 to be able to use log scale because log(0) = -infinity
    df['likes'] += 1

    # plot
    sns.displot(data=df.likes, kde=False, log_scale=True, bins=80, height=5, aspect=1.5, color='crimson')
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Video Shares')
    path_image = os.path.join('../data_analysis', 'video_share_stats.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_shares1_stats()

sns.set(rc={'figure.figsize':(14, 6)})

def plot_posttimes_total(time_interval):
    ''''
    post  count Time interval can be:
    'd' for daily
    'm' for monthly
    'y' for yearly

    '''

    df = get_dataframe(path)


    daily_count = df.resample(time_interval, on='timestamp').count()

    #fig, ax = plt.figure(figsize=(14, 6))
    sns.lineplot(data=daily_count.body, linewidth=0.4, legend=False, ax=ax)

    plt.xlabel('Year')
    plt.xlim([date(2014, 6, 1), date(2022, 9, 1)])
    plt.ylabel('Posts per Day')
    path_image = os.path.join('../data_analysis', 'posttimes_daily.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_posttimes_total('d')





def postimes_video():
    # for every folder in directory
    #5199730 videos total
    ids = []
    hours = []
    weekdays = []
    video_ages = []

    today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    id = item['video_id']
                    post_time = item['video_stats']['create_time']
                    post_time = datetime.utcfromtimestamp(post_time)
                    hour = post_time.hour
                    weekday = post_time.weekday()
                    video_age = (today - pd.to_datetime(post_time.strftime('%Y-%m-%d %H:%M:%S'))).days

                    ids.append(id)
                    hours.append(hour)
                    weekdays.append(weekday)
                    video_ages.append(video_age)

    df = pd.DataFrame({'id': ids, 'hour_of_day': hours, 'weekday': weekdays, 'video_age': video_ages})

    return df

#df_postimes_video = postimes_video()
#save_dataframe(df_postimes_video, "posttimes_video")



stopwords_multilingual = stopwords.stopwords(['he', 'bg', 'fr', 'zh', 'fi', 'pt',
                                              'ur', 'hu', 'gu', 'sl', 'tl', 'br',
                                              'eu', 'so', 'hy', 'de', 'ha', 'uk',
                                              'lt', 'sw', 'ko', 'ca', 'bn', 'ro',
                                              'af', 'et', 'en', 'nl', 'el', 'sk',
                                              'st', 'lv', 'ms', 'cs', 'la', 'eo',
                                              'fa', 'ga', 'sv', 'ru', 'hr', 'id',
                                              'hi', 'mr', 'yo', 'no', 'es', 'ja',
                                              'da', 'tr', 'vi', 'pl', 'ku', 'gl',
                                              'it', 'ar', 'th', 'zu'])


def split_emojis(s):
    return ''.join((' ' + c + ' ') if c in emoji.EMOJI_DATA else c for c in s)

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

def clean_text(text):
    # some basic cleaning
    # Make lower
    text = text.lower()
    # add whitespace before hashtags and filter ><
    text = text.replace("#", " ").replace('<', '').replace('>', '')
    # Remove punctuation
    text = re.sub(r"[^\P{P}]+", "", text)
    # remove single alphabetical letters
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # add whitespace between emojis
    text = split_emojis(text)
    # remove emojis
    text = give_emoji_free_text(text)
    # remove short words with less than 3 letters
    text = ' '.join(word for word in text.split() if len(word) > 2)
    # covert to normal font
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # remove stopwords multilingual
    text = ' '.join(
        [word for word in text.split() if word not in stopwords_multilingual])

    # TODO:REMOVE NEWLINES, add new features

    # split sentence to tokens in list
    text = text.split()

    return text

def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.EMOJI_DATA)


def get_videodata(path):
    #get all descriptions per author
    ids = []
    total_hashtags = []
    hashtag_amounts = []
    caption_amounts = []
    descriptions = []
    description_amounts = []
    total_emojis = []
    engagements = []
    engagement_rates = []

    # for every folder in directory
    for folder in os.listdir(path):

        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_2.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:

                    #video id
                    id = item['video_id']
                    ids. append(id)

                    #this contains video description and hashtags
                    description = item['video_stats']['description']

                    #extract emojis
                    emojis = extract_emojis(description)
                    total_emojis.append([emojis])



                    #this contains hashtags only
                    hashtags = item['video_stats']['hashtags']

                    # filter empty hashtags from list
                    hashtags = list(filter(None, hashtags))
                    hashtags_length = len(hashtags)
                    hashtag_amounts.append(hashtags_length)

                    for word in emojis:
                        description = description.replace(word, "")

                    total_length = len(description.split())
                    description_amounts.append(total_length)

                    #fliter hashtags
                    tags = ["#"+x for x in hashtags]

                    for word in tags:
                        description = description.replace(word, "")



                    description_length = len(description.split())
                    caption_amounts.append(description_length)


                    #convert to string
                    hashtags = ' '.join(hashtags)

                    #clean text data
                    hashtags = clean_text(hashtags)
                    description = clean_text(description)
                    total_hashtags.append(hashtags)
                    descriptions.append(description)

                    #engagements
                    shares = item['video_stats']['stats']['shareCount']
                    likes = item['video_stats']['stats']['diggCount']
                    comments = item['video_stats']['stats']['commentCount']
                    plays = item['video_stats']['stats']['playCount']

                    per_video_engagement = sum([shares, likes, comments])
                    engagements.append(per_video_engagement)
                    try:
                        engagement_rate = per_video_engagement/plays*100
                        engagement_rates.append(engagement_rate)
                    except:
                        engagement_rates.append(np.nan)
                        continue
        #print(total_hashtags)

    df = pd.DataFrame({'id': ids, 'hashtags': total_hashtags,
                        'hashtag_word_amount': hashtag_amounts,
                        'caption': descriptions, 'caption_word_amount': caption_amounts,
                        'description_word_amount': description_amounts,
                        'emojis': total_emojis, 'engagements': engagements,
                       'engagement_rate': engagement_rates
                       })

    return df



#get df of hashtags and engagemnt rate of video
#df_videodata = get_videodata(path)

#drop all rows that have any NaN values
#df_videodata = df_videodata.dropna()

#save df
#save_dataframe(df_videodata, 'text_video_engagement')




def plot_loudness():
    ids= []
    loudness_total = []

    # for every folder in directory
    for folder in os.listdir(path):
        # if it is a directory
        if os.path.isdir(os.path.join(path, folder)):
            with open(os.path.join(path, folder, 'metadata/meta_3.json')) as jsonf:
                data = json.load(jsonf)
                for item in data:
                    id = item['video_id']
                    loudness = item['video_stats']['loudness']

                    ids.append(id)
                    try:
                        loudness_total.append(float(loudness))
                    except:
                        loudness_total.append(np.nan)

    df = pd.DataFrame({'id': ids, 'loudness': loudness_total})

    # plot
    sns.displot(data=df.loudness.dropna())
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Video Loudness')
    path_image = os.path.join('../data_analysis', 'loudness.png')
    plt.savefig(path_image, dpi=400)
    plt.close()
    return df


#df_loudness = plot_loudness()
#save_dataframe(df_loudness, 'loudness')


def plot_weekly():
    df = get_dataframe(path)
    #fig, ax = plt.figure(figsize=(14, 6))
    sns.lineplot(data=(df.groupby(df['timestamp'].dt.isocalendar().week)['body'].count()), legend=False, ax=ax)
    plt.xlabel('Week of Year')
    plt.ylabel('Count')
    path_image = os.path.join('../data_analysis', 'weekly.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

plot_weekly()



def plot_monthly():
    df = get_dataframe(path)
    sns.lineplot(data=(df.groupby(df['timestamp'].dt.month)['body'].count()), legend=False)
    ticks = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plt.xticks(list(range(1, 13)), ticks)
    # remove spacing
    plt.margins(x=0)
    plt.xlabel('Month of Year')
    plt.ylabel('Count')
    path_image = os.path.join('../data_analysis', 'monthly.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_monthly()



def plot_day_of_week():
    df = get_dataframe(path)
    sns.lineplot(data=(df.groupby(df['timestamp'].dt.dayofweek)['body'].count()), legend=False)
    ticks = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # remove spacing
    plt.margins(x=0)
    locs, labels = plt.xticks()
    plt.xticks(locs, ticks)
    plt.xlabel('Day Of Week')
    plt.ylabel('Count')
    path_image = os.path.join('../data_analysis', 'dayofweek.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#plot_day_of_week()



# create punchcard visualisation function
def draw_punchcard(infos,
                   ax1=range(7),
                   ax2=range(24),
                   ax1_ticks=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                   ax2_ticks=range(24),
                   ax1_label='Day Of Week',
                   ax2_label='Hour Of Day'
                   ):

    # build the array which contains the values
    data = np.zeros((len(ax1), len(ax2)))
    for key in infos:
        data[key[0], key[1]] = infos[key]
    data = data / float(np.max(data))

    # shape ratio
    r = float(data.shape[1]) / data.shape[0]

    cmap = plt.get_cmap('hot_r')

    # Draw the punchcard (create one circle per element)
    # Ugly normalisation allows to obtain perfect circles instead of ovals....
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            rectangle = plt.Rectangle((x / float(data.shape[1] - 1) * (data.shape[0]) - (
                        data[y][x] / float(data.shape[1]) * data.shape[0]) / 1.1 / 2,
                                       y / r - (data[y][x] / float(data.shape[1]) * data.shape[0]) / 1.1 / 2),
                                      data[y][x] / float(data.shape[1]) * data.shape[0] / 1.1,
                                      data[y][x] / float(data.shape[1]) * data.shape[0] / 1.1,
                                      color=cmap(data[y][x]))
            plt.gca().add_artist(rectangle)

    plt.ylim(0 - 0.5, data.shape[0] - 0.5)
    plt.xlim(0, data.shape[0])
    plt.yticks(np.arange(0, len(ax1) / r - .1, 1 / r), ax1_ticks)
    xt = np.linspace(0, len(ax1), len(ax2))
    plt.xticks(xt, ax2_ticks)
    plt.ylabel(ax1_label)
    plt.xlabel(ax2_label)
    plt.gca().invert_yaxis()

    # make sure the axes are equal, and resize the canvas to fit the plot
    plt.axis('equal')
    plt.axis([-.1, 7.15, 7 / r, -.3])
    scale = 0.7
    plt.gcf().set_size_inches(data.shape[1] * scale, data.shape[0] * scale, forward=True)
    path_image = os.path.join('../data_analysis', 'punchcard.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()

#df = get_dataframe(path)
#draw_punchcard(dict(df.groupby([df["timestamp"].dt.dayofweek, df["timestamp"].dt.hour])['body'].count()))

#TODO: join languages with time posted df and add timezone delta to calculate time posted

# total occurrences of each hour for each respective day of the week
def post_hourly_week_distribution():
    print('Calculating Hourly Week')
    df = get_dataframe(path)

    fig, ax = plt.subplots(ncols=7, figsize=(30, 5))
    plt.subplots_adjust(wspace=0.05)  #Remove some whitespace between subplots

    for idx, gp in df['timestamp'].groupby(df['timestamp'].dt.dayofweek):
        ax[idx].set_title(gp.dt.day_name().iloc[0])  #Set title to the weekday

        (gp.groupby(gp.dt.hour).size().rename_axis('Hour').to_frame('')
            .reindex(np.arange(0, 24, 1)).fillna(0)
            .plot(kind='bar', ax=ax[idx], rot=0, ec='k', legend=False))

        # Ticks and labels on leftmost only
        if idx == 0:
            _ = ax[idx].set_ylabel('Counts', fontsize=11)

        _ = ax[idx].tick_params(axis='both', which='major', labelsize=7,
                                labelleft=(idx == 0), left=(idx == 0))

    # Consistent bounds between subplots.
    lb, ub = list(zip(*[axis.get_ylim() for axis in ax]))
    for axis in ax:
        axis.set_ylim(min(lb), max(ub))

    path_image = os.path.join('../data_analysis', 'hourly_week.png')
    plt.savefig(path_image, dpi=400, bbox_inches="tight")
    plt.close()


#post_hourly_week_distribution()


#TODO: join languages with time posted df and add timezone delta to calculate time posted
