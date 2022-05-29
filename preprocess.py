# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import glob



dataset_names = [
          "twitter_BlackLivesMatter", 
          "twitter_Abortion",
          "reddit_politics"
         ]


for dataset in dataset_names: 

    if dataset=="reddit_politics": 
        files = glob.glob("input/"+dataset+"*.tsv")
    else:
        files = glob.glob("input/"+dataset+"*.csv")

    dfs = []
    map_user = {}
    for file in files: 
        if dataset=="reddit_politics": 
            df_each_month = pd.read_csv(file, sep="\t", error_bad_lines=False, engine="python", dtype="object")
            df_each_month['date'] = pd.to_datetime(df_each_month['date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        else:
            df_each_month0 = pd.read_csv(file, sep=",", error_bad_lines=False, engine="python", dtype="object")
            df_each_month = pd.read_csv(file, sep=",", error_bad_lines=False, engine="python")
            tmpmap_user = dict(zip(df_each_month["user"].astype(str), df_each_month0["user"].astype(str)))
            map_user = map_user | tmpmap_user
            df_each_month['date'] = pd.to_datetime(df_each_month['date'], format="%Y-%m-%dT%H:%M:%S.%fZ", errors='coerce')
        df_each_month["sentence"] = df_each_month["sentence"].astype(str)
        df_each_month['sentence'] = df_each_month['sentence'].replace(['&amp;','&lt;','&gt;'],['&','<','>'], regex=True)
        df_each_month = df_each_month[~df_each_month.sentence.str.contains("#.*#.*#.*#")]
        df_each_month = df_each_month[~df_each_month.sentence.str.contains("@")]
        df_each_month = df_each_month[~df_each_month.sentence.str.contains("https")]
        df_each_month = df_each_month[~df_each_month.sentence.str.contains('News|NEWS|news| percent|% |Biden ')]
        df_each_month = df_each_month[~df_each_month.sentence.str.contains(' calls | Calls | Say| tells | say| said|announce')]
        df_each_month = df_each_month[~df_each_month.sentence.str.contains('RT |:| - | ― | — | – |-')]
        df_each_month = df_each_month[~df_each_month.sentence.str.contains(' state| State| country| Council| City| council| city')]
        dfs.append(df_each_month)
    df = pd.concat(dfs)
    

    if dataset=="twitter_Abortion": 
        df = df[(df["date"]>=datetime(2018, 6, 1, 0, 0, 0))&(df["date"]<datetime(2021, 5, 1, 0, 0, 0))]
        df = df[~df.sentence.str.contains(' Senate| so far today ')]
        df = df[df.sentence.str.contains('abortion', case=False)]
    if dataset=="twitter_vaccination": 
        df = df[df["date"]<datetime(2021, 3, 1, 0, 0, 0)]
        df = df[~df.sentence.str.contains(' center| centre|mass vaccination site| book|release| start|commence| began| begin| access| plan| NHS | test | CDC | launch| clinic| approve| prepare')]
        df = df[~df.sentence.str.contains('flu |Flu |influenza|RNA| Card|NHS|HPV| U.S| order')]
        df = df[df.sentence.str.contains('vaccination|vaccine', case=False)]
    if dataset=="twitter_BlackLivesMatter": 
        df = df[df["date"]<datetime(2021, 4, 1, 0, 0, 0)]
        df = df[~df.sentence.str.contains(' protest| Protest| march')]
        df = df[~df.sentence.str.contains(" ding| Ding| DING")]

    df = df.drop_duplicates(subset="sentence",keep=False)
    df['time'] = (df['date']-df['date'].min()) / timedelta(weeks=2)
    df['time'] /= df['time'].max()
    df['time'] = df['time'] - 0.5

    counts = df["user"].value_counts()

    user_order = counts[(counts>=5)].index
    user_dict = {user_name: user_id for user_id, user_name in enumerate(user_order)}
    df["user_id"] = df["user"].map(user_dict)
    df = df.sort_values('user')
    df = df.sort_values('date')
    df = df.dropna()
    df["user"] = df["user"].astype(str)
    if dataset=="reddit_politics": 
        df['opinion'] = df['subreddit'].map({'Libertarian': 1, 'Conservative': -1})
    if dataset!="reddit_politics": 
        df["user"] = df["user"].map(map_user)
    df.to_csv("working/posts_"+dataset+".tsv", index=False)
    
    if dataset!="reddit_politics": 
        df["sentence"].to_csv("working/texts_"+dataset+".tsv", header=False, sep="\t")

    print(dataset, "&", len(df), "&", len(df["user"].unique()), "&", df["date"].min().strftime('%B %d, %Y'), "-", df["date"].max().strftime('%B %d, %Y'))


