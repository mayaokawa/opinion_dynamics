import sys
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import torchtext
import re

sns.set(style="white",font_scale=0.9)
sns.set_context("paper", 1.5, {"lines.linewidth": 2})

datasets = ["synthetic_consensus","synthetic_clustering","synthetic_polarization","twitter_BlackLivesMatter","twitter_Abortion","reddit_politics"]
dataset_names = {"synthetic_consensus": "Consensus", "synthetic_clustering": "Clustering", "synthetic_polarization": "Polarization", 
                 "twitter_BlackLivesMatter":"Twitter BLM","twitter_Abortion":"Twitter Abortion","reddit_politics":"Reddit Politics"} 


if __name__ == '__main__':

    for data_type in datasets:
    
        if "synthetic" in data_type:
            df = pd.read_csv("working/"+data_type+".csv", delimiter=",", names=["user_id","opinion","time"], header=None)
            df["raw_opinion"] = df["opinion"].copy()
            df["opinion"] = pd.cut(df["opinion"], [-1.1, -0.6, -0.2, 0.2, 0.6, 1.1], labels=range(5)).astype(float)
        elif "reddit" in data_type:
            df = pd.read_csv("working/posts_"+data_type+".tsv", delimiter=",", usecols=["date", "user", "time", "sentence", "opinion"], #dtype="object") 
                             dtype={"date": str, "user": str, "time": np.float64, "sentence": str}, parse_dates=["date"]) 
            df["opinion"] = (df["opinion"] + 1.) / 2.
        elif "voteview" in data_type:
            df = pd.read_csv("working/posts_"+data_type+".tsv", delimiter=",", usecols=["user_id", "time", "opinion"])
            df["opinion"] = (df["opinion"] + 1.) / 2.
        else:
            df_score = pd.read_csv("working/rated_twitter_with_ID/rated_"+data_type+"_with_ID.tsv", delimiter="\t", usecols=["Sentiment Rating","Rater’s ID"], dtype="object")
    
            df = pd.read_csv("working/posts_"+data_type+".tsv", delimiter=",", usecols=["date", "user", "time", "sentence"], 
                             dtype={"date": str, "user": str, "time": np.float64, "sentence": str}, parse_dates=["date"]) 
            df["opinion"] = df_score['Sentiment Rating'].astype(float)
            df = df[~df["sentence"].str.contains('\"')]
            opinion_dict = {1:0, 2:1, 4:2, 5:3}
            df["opinion"] = df["opinion"].map(opinion_dict)
            df = df.dropna()
    
        if "twitter" in data_type:
            df_user = pd.read_csv("input/profiles_"+data_type+".tsv", delimiter="\t", usecols=["user","username","description","following_count","followers_count","tweet_count"], lineterminator='\n', dtype="object")
            map_username = dict(zip(df_user["user"].astype(str), df_user["username"].astype(str)))
            df["username"] = df["user"].map(map_username)
            df = df[~df["username"].str.contains("bot|news", case=False)] 
            map_description = dict(zip(df_user["user"].astype(str), df_user["description"].astype(str)))
            df["description"] = df["user"].map(map_description)
            df = df[~df["description"].str.contains("bot|news", case=False)] 
            if data_type=="twitter_Abortion" or data_type=="twitter_BlackLivesMatter":
                map_tweet_count = dict(zip(df_user["user"].astype(str), df_user["tweet_count"].astype(float)))
                df["tweet_count"] = df["user"].map(map_tweet_count)
                df = df[(df["tweet_count"]<300000)]
            counts = df["user"].value_counts()
            map_counts = dict(zip(counts.index.astype(str), counts.values.astype(float)))
            df["counts"] = df["user"].map(map_counts)
            df = df[(df["counts"]>=3)]
            df = df.dropna(subset=["opinion"])

            counts = df["user"].value_counts()
            user_order = counts[(counts>=5)].index
            user_dict = {user_name: user_id for user_id, user_name in enumerate(user_order)}
            df["user_id"] = df["user"].map(user_dict)
            df = df.dropna(subset=["user_id"])
            nclasses = len(df["opinion"].unique())

            df_profile = pd.read_csv("input/profiles_"+data_type+".tsv", delimiter="\t", usecols=["user","description"], lineterminator='\n')
            df_profile = df_profile.astype(str)
            df_profile["user_id"] = df_profile["user"].map(user_dict)
            df_profile = df_profile.dropna(subset=["user_id"])
            df_profile = df_profile.set_index("user_id").sort_index()
            #
            mat_profile = pd.DataFrame(df_profile["description"].to_list())
            mat_profile = mat_profile.fillna("nan")
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            encoded_inputs = tokenizer(mat_profile[0].to_list(), padding=True, truncation=True, max_length=25).input_ids 
            mat_profile[0].to_csv("working/input_text_profile_"+data_type+".tsv")
            lang_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=nclasses, output_hidden_states=True)
            profs = torch.from_numpy(np.array(encoded_inputs, dtype=np.float32)).clone().long() 
            lang_output = lang_model(profs)
            logits = lang_output.logits.detach().numpy() #.squeeze(1) #.hidden_states 
            hidden_state = lang_output.hidden_states[-1].detach().numpy()
            hidden_state = hidden_state.reshape(hidden_state.shape[0], (hidden_state.shape[1]*hidden_state.shape[2])) 
            np.savez_compressed("working/hidden_state_profile_"+data_type, logits=logits, hidden_state=hidden_state, fmt="%s")

            
    
        if "reddit" in data_type:
            counts = df["user"].value_counts()
            user_order = counts[(counts>=5)&(counts<500)].index
            user_dict = {user_name: user_id for user_id, user_name in enumerate(user_order)}
            df["user_id"] = df["user"].map(user_dict)
            df = df.dropna(subset=["user_id"])
    
        df["time"] = (df["time"]-df["time"].min()) / (df["time"].max()-df["time"].min())
        df = df.sort_values("time")
    
        if "synthetic" in data_type:
            df.to_csv("working/posts_final_"+data_type+".tsv", index=False, sep="\t", columns=["user_id","opinion","raw_opinion","time"])
        else:
            df.to_csv("working/posts_final_"+data_type+".tsv", index=False, sep="\t", columns=["user_id","opinion","time"])
    
        num_users = int(df["user_id"].max()) + 1
        initial_u = np.zeros(num_users)
        for iu in range(num_users): 
            tmpop = np.array(df[df["user_id"]==iu]["opinion"])
            if len(tmpop)>0:  
                initial_u[iu] = tmpop[0]
        initial_u = np.array(initial_u)
        np.savetxt("working/initial_"+data_type+".txt", initial_u)

    
        if not "synthetic" in data_type: 
            nclasses = len(df["opinion"].unique())
            mean = nclasses / 2.
            nusers = len(df["user_id"].unique())
            npos = len(df[df["opinion"]>=mean])
            nneg = len(df[df["opinion"]<mean])
            ndata = len(df)
            print(dataset_names[data_type], "&", "{:,}".format(nusers), "&", "{:,}".format(ndata), "&", "{:,}".format(npos), "&", "{:,}".format(nneg), "&", nclasses, "&", df["date"].min().strftime('%B %d, %Y'), "-", df["date"].max().strftime('%B %d, %Y'), "\\\\")


