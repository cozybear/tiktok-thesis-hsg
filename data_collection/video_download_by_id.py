import json
from TikTokApi import TikTokApi
import os
import random
import sys
import time
import random



def get_cookies_from_file():
    with open('../cookies.json') as f:
        cookies = json.load(f)

    cookies_kv = {}
    for cookie in cookies:
        cookies_kv[cookie['Name raw']] = cookie['Content raw']

    return cookies_kv


cookies = get_cookies_from_file()

def get_cookies(**kwargs):
    return cookies


##############

def download_ids(path):
    with TikTokApi(custom_device_id='xxx',
                   custom_verify_fp='xxx',
                   use_test_endpoints=True) as api:
        api._get_cookies = get_cookies  # This fixes issues the api was having
        pathy = path
        # for folder in metadata json
        for directory in next(os.walk(pathy))[1]:
            folder = ''.join(directory)
            # create new folder
            if not os.path.exists(os.path.join(pathy, folder, 'videos')):
                os.makedirs(os.path.join(pathy, folder, 'videos'))

            try:
                with open(os.path.join(pathy, folder, 'metadata/meta.json')) as f:
                    json_meta = json.load(f)
                    videoids = [item.get('video_id') for item in json_meta]
                    # set max video amount limit per user
                    limit = 2

                    #randomize and limit
                    samples = random.sample(videoids, limit)
                    # init counter
                    count = 1
                    # for id in random video ids of user
                    for video_id in samples:
                        try:
                            # if file was already downloaded
                            if os.path.exists(os.path.join(pathy, folder, f'videos/{video_id}.mp4')):
                                print(f'File {video_id} already exists. Skipping...\n')
                                # continue for other files in loop
                                continue
                            video = api.video(id=video_id)
                            video_data = video.bytes()
                            print(f'Downloading ID: {video_id} / No. {count} of {limit}...')


                            with open(os.path.join(pathy, folder, f'videos/{video_id}.mp4'), "wb") as out_file:
                                print(f'Writing file to : /{folder}/videos...\n')
                                out_file.write(video_data)

                            count += 1
                            time.sleep(random.uniform(4.02, 8.135))
                        except Exception as e:
                            print(f'An error occurred for ID {video_id}:', e)
                            print('Continuing...\n')
                            continue

            except KeyboardInterrupt:
                sys.exit(0)

            except Exception as e:
                print('Error:', e)
                continue



#download_ids('metadata_test')
download_ids('../xxx/tiktokdata/metadata_videos')
