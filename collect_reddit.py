import requests
from datetime import datetime, timedelta
import traceback
import time
import pandas as pd
import numpy as np
import re

 

url = "https://api.pushshift.io/reddit/{}/search?&limit=1000&subreddit={}&before={}&after="

start = datetime(2020, 5, 1)
end = datetime(2020, 9, 1)
nday = 1 

def date_range(start, stop, step=timedelta(hours=nday)):
    current = start
    while current < stop:
        yield current
        current += step


def downloadFromUrl(subreddit="politics", user=None, object_type="submission"):
    
    count = 0
    posts = []

    for date in date_range(start, end):
        epoch = int((date+timedelta(hours=nday)).timestamp())
        previous_epoch = int(date.timestamp())
        new_url = url.format(object_type, subreddit, epoch)+str(previous_epoch)
        json = requests.get(new_url)
        time.sleep(0.2) 
        try:
            json_data = json.json()
        except Exception as err:
            print("Couldn't open json file")
            continue
        if 'data' not in json_data:
            continue
        objects = json_data['data']
        if len(objects) == 0:
            continue
    
        for object in objects:
            count += 1
            try:
                parsed_date = datetime.utcfromtimestamp(object['created_utc'])
                text = object['title']
                score = object['score']
                num_comments = object['num_comments']
                user = object['author']
                posts.append([parsed_date,user,subreddit,text,score,num_comments])
            except Exception as err:
                print(f"Couldn't print post: {object['subreddit']}")
                print(traceback.format_exc())
    
        print("Saved {} {}s through {}".format(count, object_type, datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")))

    df = pd.DataFrame(posts,columns=['date','user','subreddit','sentence','score','num_comments'])
    print(f"Saved {count} {object_type}s")
    
    return df



dfs = []

for subreddit in ["Conservative", "Libertarian"]:  
    df = downloadFromUrl(subreddit=subreddit)
    print(df)
    dfs.append(df)
posts = pd.concat(dfs)

str_start = start.strftime('%Y%m%d')
str_end = end.strftime('%Y%m%d')
posts.to_csv("input/reddit_politics_"+str_start+"-"+str_end+".tsv", index=False, sep="\t")

