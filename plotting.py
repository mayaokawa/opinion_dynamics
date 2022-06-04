import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from matplotlib.collections import LineCollection
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import glob
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from datetime import datetime
import matplotlib.dates as mdates


sns.set(style="white",font_scale=0.9)
sns.set_context("paper", 1.5, {"lines.linewidth": 2})
fig_format = ".png" #".eps"

size_users = [2500] 
flag_profiles = [True, False]

hist_labels = {2: ["Positive", "Negative"],
               4: ["Highly positive", "Positive", "Negative", "Highly negative"], 
               5: ["Highly positive", "Positive", "Neutral", "Negative", "Highly negative"]} 


def calc_label_error(pred_label, truth_label, nclasses):

    pred_label = pred_label.fillna(nclasses).clip(lower=0.,upper=nclasses+1)
    pred_label = np.round(pred_label).astype(np.int)
    truth_label = ((nclasses-1)*truth_label).astype(np.int)
    acc = accuracy_score(truth_label, pred_label)
    recall = recall_score(truth_label, pred_label, average='macro', zero_division=1)
    f1 = f1_score(truth_label, pred_label, average='macro')
    cm = confusion_matrix(truth_label, pred_label, labels=range(nclasses))
    return acc, f1, recall, cm 


def visualize_hist(test_res, dataset, nclasses):

    timeslices = [val_period, 0.75, 0.8, 0.85, 0.9, 1.0]
    for timeslice in timeslices:
        fig, axes = plt.subplots(1, 3, figsize=(7,2), sharey=True)
        for im, method in enumerate(["Ground truth","NN","SINN"]):
            if method=="Ground truth":
                test_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,"SINN")]+"/test_predicted_SINN.csv")
            else:
                test_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,method)]+"/test_predicted_"+method+".csv")
            test_res["ts_delt"] = (test_res["time"] - timeslice).abs()
            indx = test_res.groupby('user')['ts_delt'].idxmin() 
            subset_res = test_res.loc[indx]
            if method=="Ground truth":
                node_value = np.array(subset_res["gt"])
                node_color = (nclasses-1)*node_value
            else:
                node_color = np.array(subset_res["pred_label"].fillna(nclasses).clip(lower=0.,upper=nclasses+1))
            arr = np.arange(nclasses+1)
            N, bins, patches = axes[im].hist(node_color, arr, edgecolor='white', linewidth=0, alpha=0.7, rwidth=0.9)
            if method=="SINN": axes[im].set_xlabel("Our SINN")
            else: axes[im].set_xlabel(method)
            cmap = plt.cm.get_cmap('coolwarm', len(arr)-1)
            for ip, patch in zip(arr, patches):
                patch.set_color(cmap(ip))
            axes[im].set_xticks([])
        axes[0].set_ylabel("Counts")
        axes[0].legend(handles=patches, labels=hist_labels[nclasses][::-1], bbox_to_anchor=(-0.2,1.05), loc=3, 
                       borderaxespad=0., labelspacing=0.3, handletextpad=0.3, columnspacing=0.3, ncol=5)
        plt.savefig(outdir+dataset+"/hist_"+str(timeslice)+".png", bbox_inches='tight', pad_inches=0.05), plt.close()



def visualize_graph(test_res, dataset, nclass=5):

    timeslices = [val_period, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] #np.linspace(val_period, 1, 3)[:-1]
    df_gt = pd.read_csv("working/posts_final_"+dataset+".tsv", delimiter="\t")

    cmap = plt.cm.get_cmap('RdYlBu_r', nclass)

    fig, ax = plt.subplots(figsize=(8,1.5))
    for iu in df_gt["user_id"].unique():
        tmpop = np.array(df_gt["raw_opinion"][df_gt["user_id"]==iu])
        tmptime = np.array(df_gt["time"][df_gt["user_id"]==iu]) * 200
        points = np.array([tmptime, tmpop]).T.reshape(1, -1, 2)
        norm = plt.Normalize(df_gt["raw_opinion"].min(), df_gt["raw_opinion"].max())
        lc = LineCollection(points, cmap='RdYlBu_r', norm=norm, alpha=0.2)
        lc.set_array(tmpop)
        lc.set_linewidth(1.5)
        line = ax.add_collection(lc)
    for side in ['top','right','bottom','left']:
        ax.spines[side].set_color('grey') #.set_visible(False)
    handles_markers = []
    markers_labels = hist_labels[nclass]
    for ll, marker_name in enumerate(markers_labels):
        pts = plt.scatter([0], [0], marker="o", color=cmap(ll), label=marker_name, s=40, edgecolor="black")
        handles_markers.append(pts)
        pts.remove()
    plt.legend(handles_markers,markers_labels,bbox_to_anchor=(0.15,1.4,0.7,0),loc='upper center', #mode="expand",
               borderaxespad=0., labelspacing=0., handletextpad=0., columnspacing=0., ncol=5)
    plt.ylim(-1, 1)
    plt.xlim(0, 200)
    plt.savefig(outdir+dataset+"/raw_opinion.png", bbox_inches='tight', pad_inches=0.02), plt.close()


    for it, timeslice in enumerate(timeslices): 
        pos = None
        for im, method in enumerate(["Ground truth","NN","SINN"]): 
            fig, ax = plt.subplots(figsize=(3,3))
            if method=="Ground truth": 
                test_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,"SINN")]+"/test_predicted_SINN.csv")
                test_zu = pd.read_csv("working/synthetic_interaction_"+dataset.split("_")[1]+".csv", names=("u","v","time"))
                test_zu["time"] /= test_zu["time"].max()  
            else:
                test_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,method)]+"/test_predicted_"+method+".csv")
                files = glob.glob(outdir+dataset+"/*_SBCM/interaction_predicted_SINN.csv")
                test_zu = pd.read_csv(files[0])

            Nu = len(test_res["user"].unique())
            test_res = test_res[test_res["user"]<Nu]

            test_res["ts_delt"] = (test_res["time"] - timeslice).abs()
            indx = test_res.groupby('user')['ts_delt'].idxmin() 
            subset_res = test_res.loc[indx]
            if method=="Ground truth": 
                W = np.zeros([Nu,Nu])
                for _, row in test_zu.iterrows():
                    if abs(row["time"]-timeslice)<0.01: 
                        if int(row["u"])<Nu and int(row["v"])<Nu: W[int(row["u"]),int(row["v"])] = 1
            elif method=="SINN":
                subset_zu = test_zu.loc[indx]
                subset_zu = subset_zu.drop("user",axis=1)
                W = np.array(subset_zu)[:,:Nu]
            else:
                W = np.zeros([Nu,Nu])
            nodes = subset_res["user"].astype(np.int)

            if method=="Ground truth":
                node_value = np.array(subset_res["gt"])
                node_color = (nclass-1)*node_value
            else:
                node_color = np.array(subset_res["pred_label"].fillna(nclass).clip(lower=0.,upper=nclass+1))
                node_value = node_color / (nclass-1)
            G = nx.from_numpy_matrix(0.7*W)
            poss = np.random.rand(len(G.nodes),2)
            rad = np.random.rand(Nu)
            theta = 2 * np.pi * np.random.rand(Nu)
            poss = np.c_[rad * np.cos(theta), rad * np.sin(theta)]
            poss[node_color==0,:] += np.array([-1,-1])
            poss[node_color==1,:] += np.array([1,-1])
            poss[node_color==3,:] += np.array([-1,1])
            poss[node_color==4,:] += np.array([1,1])
            pos = dict(zip(G.nodes, poss))
            weights = nx.get_edge_attributes(G,'weight').values()
            nx.draw_networkx_nodes(G, pos, node_color=cmap(node_value), edgecolors='grey', linewidths=0.5, node_size=30, ax=ax, alpha=0.9)
            nx.draw_networkx_edges(G, pos, edge_color='grey', ax=ax, alpha=0.7, width=list(weights))
            for side in ['top','right','bottom','left']:
                ax.spines[side].set_visible(False)
            plt.savefig(outdir+dataset+"/network_estimated_"+method+"_"+str(timeslice)+".png", bbox_inches='tight', pad_inches=0.01, transparent=True), plt.close()


def date_linspace(start, end, steps):
  delta = (end - start) / steps
  increments = range(0, steps) * np.array([delta]*steps)
  return start + increments


datasets = ["synthetic_consensus","synthetic_clustering","synthetic_polarization"] #,"sample_twitter_Abortion"]
dataset_names = {"synthetic_consensus": "Consensus", "synthetic_clustering": "Clustering", "synthetic_polarization": "Polarization", 
                 "twitter_BlackLivesMatter":"Twitter BLM","twitter_Abortion":"Twitter Abortion",
                 "sample_twitter_Abortion":"Twitter Abortion","reddit_politics":"Reddit Politics"} 
nclasses = {"synthetic_consensus": 5, "synthetic_clustering": 5, "synthetic_polarization": 5, 
            "twitter_BlackLivesMatter": 4,"twitter_Abortion": 4,"reddit_politics": 2} 
methods = ["Voter", "DeGroot", "AsLM", "SLANT", "SLANT+", "NN", "SINN"]


if "synthetic" in datasets[0]:
    train_period = 0.5
    val_period = 0.7
else:
    train_period = 0.7
    val_period = 0.8

bins5 = np.linspace(val_period,1.,11,endpoint=True)
markers = ["x",".","*","d","P","^","x","o","+"]
outdir = "output/"

best_dir_acc = {}
for method in methods: 
    for dataset in datasets: 
        best_val_acc = -9999
        files = glob.glob(outdir+dataset+"/*/val_predicted_"+method+".csv")
        for fname in files: 
            tmpdf = pd.read_csv(fname)
            _, val_acc, _, _ = calc_label_error(tmpdf["pred_label"], tmpdf["gt"], nclasses[dataset])
            if best_val_acc<val_acc: 
                best_val_acc = val_acc 
                best_dir_acc[(dataset,method)] = fname.split("/")[-2] 
#print(best_dir_acc)


################################
###      Overall results     ###
################################

maes = {}
frs = {}
accs = {}
recalls = {}
f1s = {}
for dataset in datasets: 
    maes[dataset] = {}
    frs[dataset] = {}
    accs[dataset] = {}
    recalls[dataset] = {}
    f1s[dataset] = {}
    for method in methods: 
        test_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,method)]+"/test_predicted_"+method+".csv")
        acc, f1, recall, cm = calc_label_error(test_res["pred_label"], test_res["gt"], nclasses[dataset])
        accs[dataset][method] = acc
        recalls[dataset][method] = recall
        f1s[dataset][method] = f1

        sns.heatmap(cm)
        plt.xlabel("Predicted")
        plt.ylabel("Ground truth")
        plt.savefig(outdir+dataset+"/confusion_matrix_"+method+".png"), plt.close()


for dataset in datasets: 
    print("&", dataset_names[dataset], end=' ')
print("\\\\")
for method in methods: 
    print("\\textsf{",method.ljust(8), "}", end=' ')
    for dataset in datasets: 
  
        acc = accs[dataset]
        recall = recalls[dataset]
        f1 = f1s[dataset]
        fr = frs[dataset]
        mae = maes[dataset]
        if max(acc, key=acc.get)==method: print("& \\textbf{", "{:.3f}".format(accs[dataset][method]), end='} ')
        else: print("&", "{:.3f}".format(accs[dataset][method]), end=' ') 

        if max(f1, key=f1.get)==method: print("& \\textbf{", "{:.3f}".format(f1s[dataset][method]), end='} ')
        else: print("&", "{:.3f}".format(f1s[dataset][method]), end=' ') 

    print("\\\\")


################################
###   Graph visualization    ###
################################

att_cmap = {"NN":"Greens", "SINN":"Oranges"}
for dataset in datasets: 
    if not "twitter" in dataset: continue
    profile_df = pd.read_csv("working/input_text_profile_"+dataset+".tsv")
    profile_df = profile_df.dropna()
    for method in ["NN","SINN"]:
        tmpbest_dir = find_best_nll(str(size_users[-1])+"_True/*", nclasses[dataset], _method=method, return_dir=True)
        attention_df = pd.read_csv(outdir+dataset+"/"+tmpbest_dir+"/attention_predicted_"+method+".csv")
        attention_df = attention_df.set_index('user')
        word_list = []
        for uid in range(len(profile_df)):
            sentence = profile_df["0"].iloc[uid]
            words = sentence.split(" ")[:25]
            if len(words)<2: continue
            attention = np.array(attention_df.iloc[uid])[:len(words)]
            tmpword = np.array(words)[attention>0.1]
            word_list += tmpword.tolist() 
        str_text = " ".join(word_list) #.lower()
        str_text = re.sub(r"[^a-zA-Z0-9]"," ",str_text)
        str_text = str_text.replace(" t "," ")
        str_text = str_text.replace(" s "," ")
        wordcloud = WordCloud(background_color="white",width=800,height=400,colormap=att_cmap[method],min_font_size=16,relative_scaling=0.02).generate(str_text)
        plt.imshow(wordcloud) #, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(outdir+dataset+"/attention_"+method+".png", bbox_inches='tight', pad_inches=0.01), plt.close()



for dataset in datasets: 
    if "synthetic" in dataset: continue
    test_res = pd.read_csv(outdir+dataset+"/"+best_dir[(dataset,"NN")]+"/test_predicted_NN.csv")
    visualize_hist(test_res, dataset, nclasses[dataset])

"""   
for dataset in datasets: 
    if not "synthetic" in dataset: continue
    visualize_graph(test_res, dataset, nclasses[dataset])
"""   


