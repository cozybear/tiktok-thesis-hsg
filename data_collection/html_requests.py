import requests
import lxml.html
import pandas as pd
import json
import time
from numpy import random
import traceback
from fake_useragent import UserAgent
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS


def get_cookies_from_file():
    with open('../cookies.json') as f:
        cookies = json.load(f)

    cookies_kv = {}
    for cookie in cookies:
        cookies_kv[cookie['Name raw']] = cookie['Content raw']

    return cookies_kv

# cookies from file
cookies = get_cookies_from_file()

# random useragent generator
user_agent = UserAgent()

# use amazon api gateway to rotate ip
gateway = ApiGateway(site="https://www.tiktok.com",
                     access_key_id="xxx",
                     access_key_secret="xxx")
gateway.start()

# create session
session = requests.Session()
session.mount("https://www.tiktok.com", gateway)


def retrieve_user_sec_id(user):
    try:
        # Note: It is important to send the right headers (user agent and cookie)
        headers = {
            'User-Agent': user_agent.random,
            'referer': 'https://www.google.com'
        }

        response = session.get("https://www.tiktok.com/@" + user, headers=headers, cookies=cookies)

        test = lxml.html.fromstring(response.text)

        # get the "authorSecId" (second author ID) value from test output
        authorSecId = \
            test.xpath('//script[contains(text(), "authorSecId")]/text()')[0].split('authorSecId":"')[1].split('"')[0]

        # random sleep time between requests
        time.sleep(random.uniform(0.003, 0.2))

        print(f'user {user}: Success')
        return authorSecId

    except:
        # raise error message
        print('\n'+traceback.format_exc())
        pass


tiktokusernames = pd.read_csv('../test_username.csv')

# randomly shuffle the usernames and apply the retrieve_user_sec_id function to each row
tiktokusernames['authorSecId'] = tiktokusernames['username'].sample(frac=1).apply(lambda x: retrieve_user_sec_id(x))

# write to file
tiktokusernames.to_csv('test_both_names3.csv')

# stop the gateways
gateway.shutdown()
