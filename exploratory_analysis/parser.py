import os
import json
import pprint
import sys
import numpy as np


path = '/Volumes/My_Passport/tiktokdata/total'


while True:
    try:
        i = 0
        # for every folder in directory
        for folder in os.listdir(path):
            # if it is a directory
            if os.path.isdir(os.path.join(path, folder)):
                perall = []
                # if it is a directory
                if os.path.isdir(os.path.join(path, folder)) \
                        and not folder.startswith('.'):  # exclude hidden folders of external drive

                    perfile = []
                    print(f'Editing {folder}')
                    # for every file in folder
                    for file in os.listdir(os.path.join(path, folder)):

                        if not file.startswith('.') and os.path.isfile(
                                os.path.join(path, folder, file)):

                            # open file
                            # data = pd.read_json(os.path.join('metadata_test', folder, file))
                            with open(os.path.join(path, folder, file)) as jsonf:
                                data = json.load(jsonf)

                                # for every entry in itemList in file/for every video in file
                                for item in data.get("itemList", {}):
                                    pervideo = {}
                                    stats = {}

                                    author = {}

                                    videoid = item['id']
                                    stats['stats'] = item['stats']
                                    stats['create_time'] = item['createTime']
                                    stats['description'] = item['desc']
                                    try:
                                        stats['stickers_text'] = [sticker['stickerText'] for sticker in item.get('stickersOnItem', {})]
                                    except:
                                        stats['stickers_text'] = []
                                        pass
                                    stats['duetEnabled'] = item['duetEnabled']
                                    stats['shareEnabled'] = item['shareEnabled']
                                    stats['stitchEnabled'] = item['stitchEnabled']

                                    keys = ['authorName', 'title', 'duration']
                                    stats['music'] = [{key: item.get('music', {}).get(key, {})} for key in keys]
                                    # print(stats['music'])

                                    stats['hashtags'] = [hashtag['hashtagName'] for hashtag in item.get('textExtra', {})]
                                    stats['duetdisplay'] = item['duetDisplay']
                                    try:
                                        stats['loudness'] = item['video']['volumeInfo']['Loudness']
                                    except:
                                        stats['loudness'] = ""
                                        pass

                                    try:
                                        stats['video_duration'] = item['video']['duration']
                                    except:
                                        stats['video_duration'] = ""
                                        pass


                                    pervideo['video_id'] = videoid
                                    pervideo['video_stats'] = stats

                                    author['author'] = item['author']
                                    author['authorStats'] = item['authorStats']

                                    # add pervideo to perfile
                                    perfile.append(pervideo)


                    # create directory and write to json files
                    directory = os.path.join(path, folder, 'metadata')

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    with open(os.path.join(directory, 'meta_3.json'), 'w') as outfile:
                        json.dump(perfile, outfile)



                print(f'Done for {i, folder}')
                i += 1

    except Exception:
        print(sys.exc_info())
        break
    break
