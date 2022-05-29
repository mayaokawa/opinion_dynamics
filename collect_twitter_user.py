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

def create_url(ids, max_results=10):
    
    search_url = "https://api.twitter.com/2/users"

    query_params = {'ids': ids, 
                    'user.fields': 'id,description,location,public_metrics,verified'
                   } 
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params):
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


max_results = 500


def gettwitterdata(ids):

    posts = []

    headers = create_headers(bearer_token)
    count = 0 # Counting tweets per time period
    
    for user in ids:
        
        print("-------------------")
        url = create_url(user, max_results)
        json_response = connect_to_endpoint(url[0], headers, url[1])
        print(json_response.keys())
        
        if "data" in json_response: 
            profile = json_response['data'][0]
            description = profile['description']
            if "location" in profile:
                location = profile['location']
            else:
                location = None
            username = profile['username']
            public_metrics = profile["public_metrics"]
            followers_count = public_metrics["followers_count"]
            following_count = public_metrics["following_count"]
            tweet_count = public_metrics["tweet_count"]
        else:
            description = None
            location = None
            username = None
            followers_count = None
            following_count = None
            tweet_count = None

        posts.append([user,username,location,description,followers_count,following_count,tweet_count])
        count += 1 
        print("-------------------")
        time.sleep(5) 

    
    df = pd.DataFrame(posts,columns=['user','username','location','description','followers_count','following_count','tweet_count'])
    print(df.sort_values("user"))
    print(f"Saved {count} tweets")

    return df

    
if __name__ == '__main__':
    
    keywords = [
          "twitter_BlackLivesMatter", 
          "twitter_vaccination",
          "twitter_Abortion"
         ]

    for keyword in keywords:

        posts = pd.read_csv("working/posts_"+keyword+".tsv", dtype="object")
        posts["user"] = posts["user"].astype(str)
        users = posts["user"].unique()
        ids = users.tolist()
        profiles = gettwitterdata(ids)
        profiles.to_csv("input/profiles_"+keyword+".tsv", index=False, sep="\t")


