import datetime
import pandas as pd
import json
import time
import re
import os
import schedule
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from requests_oauthlib import OAuth1Session


bearer_token = os.environ.get("BEARER_TOKEN")


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(keyword, start_date, end_date, max_results=10):
    
    search_url = "https://api.twitter.com/2/tweets/search/all" 

    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'text,author_id,geo,conversation_id,created_at,lang,public_metrics',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def date_range(start, stop, step=relativedelta(months=1)):
    current = start
    while current < stop:
        yield current
        current += step


max_results = 500


def gettwitterdata(target_query, start, end):

    count = 0
    posts = []

    headers = create_headers(bearer_token)
    
    if True:
        since = start.strftime('%Y-%m-%dT%H:%M:%SZ')
        until = end.strftime('%Y-%m-%dT%H:%M:%SZ')
        print(since, until)
    
        count = 0 # Counting tweets per time period
        flag = True
        next_token = None
        while flag:
        
            print("-------------------")

            if target_query=="vaccine":
                keyword = target_query + " covid lang:en -is:nullcast -is:retweet -is:reply -has:videos -has:links" 
            elif target_query=="BlackLivesMatter":
                keyword = "#" + target_query + " lang:en -is:nullcast -is:retweet -is:reply -has:links"
            elif target_query=="Abortion":
                keyword = target_query + " lang:en -is:nullcast -is:retweet -is:reply -has:links"

            url = create_url(keyword, since, until, max_results)
            json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
            result_count = json_response['meta']['result_count']
            print("Token: ", next_token, result_count)
        
            if 'next_token' in json_response['meta']:
                # Save the token to use for next call
                next_token = json_response['meta']['next_token']
                if result_count is not None and result_count > 0 and next_token is not None:
                    print("Start Date: ", since)
                    for tweet in json_response['data']:
                        text = re.sub('[ 　]https://t\.co/[a-zA-Z0-9]+', '', tweet['text'])
                        text = re.sub(r"[\u3000\t\n]", "", text)
                        #user_id = user['id']
                        user_id = tweet['author_id']
                        date = tweet['created_at']
                        retweet = tweet['public_metrics']['retweet_count']
                        favorite = tweet['public_metrics']['like_count']
                        posts.append([date,user_id,text,retweet,favorite])
                    count += result_count
                    print("-------------------")
                    time.sleep(5)          
            # If no next token exists
            else:
                if result_count is not None and result_count > 0:
                    print("-------------------")
                    print("Start Date: ", since)
                    for tweet in json_response['data']:
                        text = re.sub('[ 　]https://t\.co/[a-zA-Z0-9]+', '', tweet['text'])
                        text = re.sub(r"[\u3000\t\n]", "", text)
                        date = tweet['created_at']
                        user_id = tweet['author_id']
                        retweet = tweet['public_metrics']['retweet_count']
                        favorite = tweet['public_metrics']['like_count']
                        posts.append([date,user_id,text,retweet,favorite])
                    count += result_count
                    print("-------------------")
                    time.sleep(5)          

                flag = False
                next_token = None
            time.sleep(5)

    
    df = pd.DataFrame(posts,columns=['date','user','sentence','retweet','favorite'])
    print(df.sort_values("user"))
    print(f"Saved {count} tweets")

    return df

    
if __name__ == '__main__':
    
    keywords = ["BlackLivesMatter","Abortion"]

    for keyword in keywords:

        if keyword=="BlackLivesMatter":
            start = datetime(2020, 7, 1, 0, 0, 0)
            end = datetime(2020, 9, 1, 0, 0, 0)
        if keyword=="Abortion":
            start = datetime(2018, 6, 1, 0, 0, 0)
            end = datetime(2021, 5, 1, 0, 0, 0)

        for date in date_range(start, end):
            until = date + relativedelta(months=1)
            str_start = date.strftime('%Y%m%d')
            str_end = until.strftime('%Y%m%d')
            posts = gettwitterdata(keyword, date, until)

            posts.to_csv("input/twitter_"+keyword+"_"+str_start+"-"+str_end+".csv", index=False)


