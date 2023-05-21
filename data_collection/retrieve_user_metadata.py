import json
import requests
from requests.adapters import HTTPAdapter, Retry
import os
from urllib.parse import urlencode
from base64 import b64decode, b64encode
from urllib.parse import parse_qsl, urlencode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from fake_useragent import UserAgent
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import traceback
import pandas as pd
import time
import lxml.html
from stem import Signal
from stem.control import Controller
import shutil
import sys
import random
import time
from hyper.contrib import HTTP20Adapter
from hyper.http20.exceptions import StreamResetError
import cloudscraper
import httpx
import asyncio

def get_cookies_from_file():
    with open('../cookies.json') as f:
        cookies = json.load(f)

    cookies_kv = {}
    for cookie in cookies:
        cookies_kv[cookie['Name raw']] = cookie['Content raw']

    return cookies_kv


cookies = get_cookies_from_file()

def encrypt(r):
    s = urlencode(r, doseq=True, quote_via=lambda s, *_: s)
    key = "webapp1.0+202106".encode("utf-8")
    cipher = AES.new(key, AES.MODE_CBC, key)
    ct_bytes = cipher.encrypt(pad(s.encode("utf-8"), AES.block_size))
    return b64encode(ct_bytes).decode("utf-8")


def decrypt(s):
    key = "webapp1.0+202106".encode("utf-8")
    cipher = AES.new(key, AES.MODE_CBC, key)
    ct = b64decode(s)
    s = unpad(cipher.decrypt(ct), AES.block_size)
    return dict(parse_qsl(s.decode("utf-8"), keep_blank_values=True))




def get_tor_session():
    session = requests.Session()
    # Tor uses the 9050 port as the default socks port
    proxy = {
        'http': 'socks5://127.0.0.1:9050',
        "https": "socks5://127.0.0.1:9050"
    }
    session.proxies.update(proxy)
    return session


# signal TOR for a new connection
def renew_connection():
    print(f'Renewing connection...\n')
    with Controller.from_port(port=9051) as controller:
        controller.authenticate(password="16:226D33A6C87CE5BA601C9A52AA4301B42C54DE1FC94738DE3D68E743FC")
        controller.signal(Signal.NEWNYM)



# copy & paste params from browser session
msToken='xxx'
device_id = 'xxx'
X_Bogus = 'xxx'
_signature = 'xxx'
verifyFp = 'xxx'

referer = [
    'https://www.stackoverflow.com/',
    'https://www.twitter.com/',
    'https://www.google.com/',
    'https://www.tiktok.com/',
    'https://www.youtube.de/',
    'https://www.twitch.com/',
    'https://www.microsoft.com/',
    'https://www.cloudflare.com/',
    'https://www.linkedin.com/',
    'https://www.facebook.com/',
    'https://www.instagtam.com/',
    'https://www.wordpress.org/',
    'https://www.en.wikipedia.org/',
    'https://www.adobe.com/',
    'https://www.vk.com/',
    'https://www.vimeo.com/',
    'https://www.bbc.co.uk/',
    'https://www.nytimes.com/',
    'https://www.google.de/',
    'https://www.cnn.com/',
    'https://www.amazon.co.jp/',
    'https://www.google.fr',
    'https://www.google.pl/',
    'https://www.tiktok.de'
]

import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util import ssl_




# retrieve user metadata by authorSecId
def retrieve_user_metadata(authorSecId, session):
    json_data = None
    while json_data is None:
        try:
            # check ip address
            #response = session.get("https://mylocation.org/")
            response = session.get('https://httpbin.org/ip')

            print(response.json())

            i = 0
            cursor = 0
            dir = os.path.join('../metadata', authorSecId)
            # if path doesnt exist, create it
            if not os.path.exists(dir):
                print('Creating folder....\n')
                # make new directory for each user
                os.makedirs(dir)
                pass
            else:
                print('Folder already exists, continuing...')




            while True:
                # random useragent generator
                user_agent = UserAgent()
                #print(user_agent.random)
                #referer
                ref = random.choice(referer)
                #print(ref)
                # params which will be encoded into the call url (Most important: device id, msToken, verifyFp)
                params = {
                    'device_id': device_id,
                    'referer': ref,
                    'verifyFp': verifyFp,
                    'msToken': msToken,
                    'X-Bogus': X_Bogus,
                    '_signature': _signature,
                }
                # http get request containing the headers and params
                payload = {
                    'aid': '1988',
                    'app_language': 'de-DE',
                    'app_name': 'tiktok_web',
                    'browser_language': 'en',
                    'browser_name': 'Mozilla',
                    'browser_online': 'true',
                    'browser_platform': 'MacIntel',
                    'browser_version': '5.0 (Macintosh)',
                    'channel': 'tiktok_web',
                    'cookie_enabled': 'true',
                    'count': '30',
                    'cursor': cursor,
                    'device_id': device_id,
                    'device_platform': 'web_pc',
                    'focus_state': 'false',
                    'from_page': 'user',
                    'history_len': '31',
                    'is_encryption': '1',
                    'is_fullscreen': 'false',
                    'is_page_visible': 'true',
                    'language': 'de-DE',
                    'os': 'mac',
                    'priority_region': 'DE',
                    'referer': 'https://www.tiktok.com',
                    'region': 'DE',
                    'root_referer': 'https://www.tiktok.com',
                    'screen_height': '1080',
                    'screen_width': '1920',
                    'secUid': authorSecId,
                    'tz_name': 'Europe/Berlin',
                    'userId': 'undefined',
                    'verifyFp': verifyFp,
                    'webcast_language': 'de-DE'

                }

                # encrypt payload containing authorSecId
                xttparams = encrypt(payload)

                # payload is now in the x-tt-params header, in the form of base64 of AES-encrypted JSON, with the hardcoded
                # encryption key ("webapp1.0+202106")
                headers = {
                    'Accept': '*/*',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'de,en-US;q=0.7,en;q=0.3',
                    'Connection': 'keep-alive',
                    'Host': 'www.tiktok.com',
                    'Referer': 'https://www.tiktok.com',
                    'sec-ch-ua': '"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"macOS"',
                    'sec-fetch-dest': 'empty',
                    'sec-fetch-mode': 'cors',
                    'sec-fetch-site': 'same-origin',
                    'User-Agent': user_agent,
                    'x-tt-params': xttparams
                }




                call='https://m.tiktok.com/api/post/item_list/?aid=1988'
                response = session.get(call, headers=headers, params=params)

                # sleep random time between requests
                time.sleep(random.uniform(2, 4))
                print(f'Status Code: {response.status_code}')
                print(response.text)

                #print(response.reason)
                #print(response.http_version)

                json_data = response.json()
                # print(json_data.text)

                # write json response to json
                with open(f'../metadata/{authorSecId}/{i}.json', 'w') as outfile:
                    json.dump(json_data, outfile)

                print(f'Executing loop {i} for cursor timestamp {cursor} / user: {authorSecId}')
                cursor = json_data['cursor']
                i += 1

                # repeat until hasMore value in response equals False
                if not json_data['hasMore']:
                    print(f'\nScrape completed for user: {authorSecId}\n')
                    break

        except KeyboardInterrupt:
            print('\nScrape stopped by user')
            sys.exit(0)

        except:
            # raise error message
            print('\n' + traceback.format_exc())
            #print(f'Status Code: {response.status_code}')
            # request new ip
            #session = get_tor_session()

            #renew_connection()
            #session.mount("https://www.tiktok.com", HTTPAdapter())
            time.sleep(random.uniform(2, 5))
            continue




# inititate tor session and retrieve user metadata
def get_data(authorSecId, session):
    while True:
        try:
            # retrieve data
            retrieve_user_metadata(authorSecId, session)

        except Exception as e:
            print('Connection error: ', e)
            # go for next loop iteration
            pass
        break




# initiate tor session
#s = get_tor_session()
#renew_connection()
#s.mount("https://www.tiktok.com", HTTPAdapter())



s = requests.Session()
s.mount("https://www.tiktok.com", HTTPAdapter())



# read authorSecIds from file
tiktok_bothnames = pd.read_csv('../final_both_names2.csv')

# iterate over authorSecIds rows
tiktok_bothnames['authorSecId'].apply(lambda x: get_data(x, s))


# stop the gateways
#gateway1.shutdown()