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

def calc_error(pred, truth):
    pred = pred.fillna(5.).clip(lower=-5.,upper=5.)
    maes = (pred-truth).abs()

    sign_truth = np.sign(truth-0.5)
    sign_pred = np.sign(truth-0.5)
    fr = np.mean( sign_pred != sign_truth )

    return maes.mean(), maes.var(), fr


def find_best_nll(search_dir, nclasses, _method="SINN", return_dir=False):
    files = glob.glob(outdir+dataset+"/"+search_dir+"/val_predicted_"+_method+".csv")
    best_val_nll = -9999
    for fname in files: 
        tmpdf = pd.read_csv(fname)
        _, val_nll, _, _ = calc_label_error(tmpdf["pred_label"], tmpdf["gt"], nclasses)
        if best_val_nll<val_nll: 
            best_val_nll = val_nll 
            tmpbest_dir = fname.split("/")[-3] + "/" + fname.split("/")[-2]
    tmpdf = pd.read_csv(outdir+dataset+"/"+tmpbest_dir+"/test_predicted_"+_method+".csv")
    test_acc, test_nll, _, _ = calc_label_error(tmpdf["pred_label"], tmpdf["gt"], nclasses)
    if return_dir:
        return tmpbest_dir
    else:
        return test_nll 


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
                files = glob.glob(outdir+dataset+"/*/*_Powerlaw/interaction_predicted_SINN.csv")
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


datasets = ["synthetic_consensus","synthetic_clustering","synthetic_polarization"]
#datasets = ["twitter_BlackLivesMatter","twitter_Abortion","reddit_politics"]
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

best_dir = {}
best_dir_acc = {}
for method in methods: 
    for dataset in datasets: 
        best_val_nll = 9999
        best_val_acc = -9999
        files = glob.glob(outdir+dataset+"/"+str(size_users[-1])+"_*/*/val_predicted_"+method+".csv")
        for fname in files: 
            tmpdf = pd.read_csv(fname)
            val_nll, _, _ = calc_error(tmpdf["pred"], tmpdf["gt"])
            if best_val_nll>val_nll: 
                best_val_nll = val_nll 
                best_dir[(dataset,method)] = fname.split("/")[-3] + "/" + fname.split("/")[-2] 

            _, val_acc, _, _ = calc_label_error(tmpdf["pred_label"], tmpdf["gt"], nclasses[dataset])
            if best_val_acc<val_acc: 
                best_val_acc = val_acc 
                best_dir_acc[(dataset,method)] = fname.split("/")[-3] + "/" + fname.split("/")[-2] 

print(best_dir)
print(best_dir_acc)

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
        test_res = pd.read_csv(outdir+dataset+"/"+best_dir[(dataset,method)]+"/test_predicted_"+method+".csv")
        mae, mae_var, fr = calc_error(test_res["pred"], test_res["gt"])
        maes[dataset][method] = mae
        frs[dataset][method] = fr

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
    if not "synthetic" in dataset: continue
    visualize_graph(test_res, dataset, nclasses[dataset])


for dataset in datasets: 
    if "synthetic" in dataset: continue
    test_res = pd.read_csv(outdir+dataset+"/"+best_dir[(dataset,"NN")]+"/test_predicted_NN.csv")
    visualize_hist(test_res, dataset, nclasses[dataset])


################################
###   Sensitivity analysis   ###
################################

data_markers = ["o","X","^","s",".","v"]
data_cmap = sns.color_palette("colorblind")
linestyles = ["-","--",":","-."]


size_units = [8,12,16]
size_layers = [3,5,7]
alphas = [0.1,1.0,5.0]
betas = [0.1,1.0,5.0]
latent_dimensions = [1,2,3]

plt.figure(figsize=(3.5,2.3))
plt.ylabel("F1")
for idataset, dataset in enumerate(datasets): 
    test_nlls = []
    for size_unit in size_units: 
        tmpbest_nll = find_best_nll(str(size_users[-1])+"_*/"+str(size_unit)+"_*_*_*_*_*_*", nclasses[dataset])
        test_nlls.append( tmpbest_nll )
    plt.plot(size_units, test_nlls, label=dataset_names[dataset], color=data_cmap[idataset], marker=data_markers[idataset], linestyle=linestyles[idataset], lw=2.5, ms=10)
plt.legend(ncol=1,loc='upper right',borderpad=0.1,labelspacing=0.1,borderaxespad=0.1,columnspacing=0.1) #,handlelength=3.)
plt.ylim(ymin=0,ymax=1.5)
plt.xticks(size_units)
plt.savefig("output/sensitivity_size_unit.png", bbox_inches='tight', pad_inches=0.03), plt.close()

plt.figure(figsize=(3.5,2.3))
plt.ylabel("F1")
for idataset, dataset in enumerate(datasets): 
    test_nlls = []
    for size_layer in size_layers: 
        tmpbest_nll = find_best_nll(str(size_users[-1])+"_*/*_"+str(size_layer)+"_*_*_*_*_*", nclasses[dataset])
        test_nlls.append( tmpbest_nll )
    plt.plot(size_layers, test_nlls, label=dataset_names[dataset], color=data_cmap[idataset], marker=data_markers[idataset], linestyle=linestyles[idataset], lw=2.5, ms=10)
plt.legend(ncol=1,loc='upper right',borderpad=0.1,labelspacing=0.1,borderaxespad=0.1,columnspacing=0.1) #,handlelength=3.)
plt.xticks(size_layers)
plt.ylim(ymin=0,ymax=1.5)
plt.savefig("output/sensitivity_size_layer.png", bbox_inches='tight', pad_inches=0.03), plt.close()


plt.figure(figsize=(3.5,2.3))
plt.ylabel("F1")
for idataset, dataset in enumerate(datasets): 
    test_nlls = []
    for alpha in alphas: 
        tmpbest_nll = find_best_nll(str(size_users[-1])+"_*/*_*_"+str(alpha)+"_*_*_*_*", nclasses[dataset])
        test_nlls.append( tmpbest_nll )
    plt.plot(alphas, test_nlls, label=dataset_names[dataset], color=data_cmap[idataset], marker=data_markers[idataset], linestyle=linestyles[idataset], lw=2.5, ms=10)
plt.legend(ncol=1,loc='upper right',borderpad=0.1,labelspacing=0.1,borderaxespad=0.1,columnspacing=0.1) #,handlelength=3.)
plt.ylim(ymin=0,ymax=1.5)
plt.savefig("output/sensitivity_alpha.png", bbox_inches='tight', pad_inches=0.03), plt.close()


plt.figure(figsize=(3.5,2.3))
plt.ylabel("F1")
for idataset, dataset in enumerate(datasets): 
    test_nlls = []
    for beta in betas: 
        tmpbest_nll = find_best_nll(str(size_users[-1])+"_*/*_*_*_"+str(beta)+"_*_*_*", nclasses[dataset])
        test_nlls.append( tmpbest_nll )
    plt.plot(betas, test_nlls, label=dataset_names[dataset], color=data_cmap[idataset], marker=data_markers[idataset], linestyle=linestyles[idataset], lw=2.5, ms=10)
plt.legend(ncol=1,loc='upper right',borderpad=0.1,labelspacing=0.1,borderaxespad=0.1,columnspacing=0.1) #,handlelength=3.)
plt.ylim(ymin=0,ymax=1.6)
plt.savefig("output/sensitivity_beta.png", bbox_inches='tight', pad_inches=0.03), plt.close()


plt.figure(figsize=(3.5,2.3))
plt.ylabel("F1")
for idataset, dataset in enumerate(datasets): 
    test_nlls = []
    for latent_dimension in latent_dimensions: 
        tmpbest_nll = find_best_nll(str(size_users[-1])+"_*/*_*_*_*_*_"+str(latent_dimension)+"_*", nclasses[dataset])
        test_nlls.append( tmpbest_nll )
    plt.plot(latent_dimensions, test_nlls, label=dataset_names[dataset], color=data_cmap[idataset], marker=data_markers[idataset], linestyle=linestyles[idataset], lw=2.5, ms=10)
plt.legend(ncol=1,loc='upper right',borderpad=0.1,labelspacing=0.1,borderaxespad=0.1,columnspacing=0.1) #,handlelength=3.)
plt.ylim(ymin=0,ymax=1.6)
plt.savefig("output/sensitivity_latent_dimension.png", bbox_inches='tight', pad_inches=0.03), plt.close()



flag_profiles = [True,False]
for idataset, dataset in enumerate(datasets): 
    if not "twitter" in dataset: continue
    for flag_profile in flag_profiles: 
        for method in ["NN","SINN"]:
            tmpbest_nll = find_best_nll(str(size_users[-1])+"_"+str(flag_profile)+"/*_*_*_*_*_*_*", nclasses[dataset], _method=method)
            print(method, tmpbest_nll)


type_odms = ["DeGroot", "Powerlaw", "BCM", "FJ"]
dummy_x = np.arange(len(type_odms)) 
plt.figure(figsize=(3.5,2.3))
plt.ylabel("F1")
for idataset, dataset in enumerate(datasets): 
    test_nlls = []
    for type_odm in type_odms: 
        tmpbest_nll = find_best_nll(str(size_users[-1])+"_*/*_*_*_*_*_*_"+type_odm, nclasses[dataset])
        test_nlls.append( tmpbest_nll )
    if "synthetic" in dataset:
        plt.plot(dummy_x, test_nlls, label=dataset_names[dataset], color=data_cmap[idataset+3], marker=data_markers[idataset+3], linestyle=linestyles[idataset], lw=2.5, ms=10)
    else:
        plt.plot(dummy_x, test_nlls, label=dataset_names[dataset], color=data_cmap[idataset], marker=data_markers[idataset], linestyle=linestyles[idataset], lw=2.5, ms=10)
type_odms[1] = "SBCM"
plt.xticks(dummy_x, type_odms)
plt.legend(ncol=1,borderpad=0.1,labelspacing=0.1,borderaxespad=0.1,columnspacing=0.1) #,handlelength=3.)
#plt.margins(y=0.)
plt.ylim(ymin=0,ymax=1.5)
if "synthetic" in datasets[0]:
    plt.savefig("output/sensitivity_odm_synthetic.png", bbox_inches='tight', pad_inches=0.03), plt.close()
else:
    plt.savefig("output/sensitivity_odm_real.png", bbox_inches='tight', pad_inches=0.03), plt.close()




################################
### Quantitative Evalutation ###
################################

mapes = {}
for dataset in datasets: 

    print()
    print(dataset, method)
    dfs = {}
    for method in methods:
        
        train_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,method)]+"/train_predicted_"+method+".csv")
        val_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,method)]+"/val_predicted_"+method+".csv")
        test_res = pd.read_csv(outdir+dataset+"/"+best_dir_acc[(dataset,method)]+"/test_predicted_"+method+".csv")
        test_res["pred"] = test_res["pred"].fillna(2).clip(lower=-2.,upper=2.)
        df_res = pd.concat([train_res, val_res, test_res]) 
        df_res = df_res.sort_values("time")

        plt.figure(figsize=(7,3.5))
        for tmpuid in range(20): 
            tmptime = df_res[df_res["user"]==tmpuid]["time"]
            tmpod = df_res[df_res["user"]==tmpuid]["pred"]
            plt.plot(np.array(tmptime), np.array(tmpod))
        plt.vlines(x=val_period, ymin=0, ymax=1, ls='--', lw=1.5, color="grey")
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        #plt.ylim(-1,1)
        plt.tight_layout()
        plt.savefig(outdir+dataset+"/estimated_"+method+".png"), plt.close()

        plt.figure(figsize=(7,3.5))
        for tmpuid in range(20): 
            tmptime = df_res[df_res["user"]==tmpuid]["time"]
            tmpod = df_res[df_res["user"]==tmpuid]["gt"]
            plt.plot(np.array(tmptime), np.array(tmpod))
        plt.vlines(x=val_period, ymin=0, ymax=1, ls='--', lw=1.5, color="grey")
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        #plt.ylim(-1,1)
        plt.tight_layout()
        plt.savefig(outdir+dataset+"/ground_truth.png"), plt.close()
 
        accs = []
        for ibin in range(len(bins5)-1): 
            subset_res = test_res[(test_res["time"]>=bins5[ibin])&(test_res["time"]<bins5[ibin+1])]
            _, acc, _, _ = calc_label_error(subset_res["pred_label"], subset_res["gt"], nclasses=nclasses[dataset])
            accs.append(acc)
        mapes[(dataset,method)] = accs  


        initial_dict = {}
        for iu in df_res["user"].unique():
            tmpop = np.array(df_res["gt"][df_res["user"]==iu])
            initial_dict[iu] = np.sign(tmpop)[0]
        df_res["initial"] = df_res["user"].map(initial_dict)

        dfs[method] = df_res

    
