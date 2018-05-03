import os
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

flatten = lambda l: [item for sublist in l for item in sublist]


def return_dict_data(folder):
    data_dict = dict()
    i = 0
    for name in os.listdir(folder):
        if name.endswith('.png') or name.endswith('.jpg') and not name.startswith('PRC') and not name.startswith('ROC'):
            filepath = os.path.join(folder, name)
            img = load_img(filepath, grayscale=True)
            img_array = img_to_array(img)
            img_list = img_array[:,:,0].tolist()
            img_flat_list = flatten(img_list)
        
            data_dict['image%s' %i] = img_flat_list
            i= i+1
    return data_dict
    
    
def normalize_to_df(dictionary):
    return pd.DataFrame.from_dict(dictionary)/255.0


def compute_f1_score(df_mask, df_result, path, threshold=0.5):
    i=0
    f1_scores = dict()
    for name in df_mask.columns:
        y_true = df_mask[name].values.astype(int)
        scores = df_result[name].values
        y_pred = (scores>threshold).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f1_scores[str(i)] = f1
        i=i+1
    
    return pd.DataFrame(data=[f1_scores], index=[path])

def compute_rand_index_adjusted(df_mask, df_result, path):
    i=0
    r_index_adj = dict()
    for name in df_mask.columns:
        y_true = df_mask[name].values.astype(int)
        ypred = df_result[name].values
        ria = adjusted_rand_score(y_true, ypred)
        r_index_adj[str(i)] = ria
        i=i+1
    return pd.DataFrame(data=[r_index_adj], index=[path])

def compute_precision_recal(df_mask, df_result, path):
    i=0
    precision_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    for name in df_result.columns:
        y = df_mask[name].values.astype(int)
        scores = df_result[name].values
        precision, recall, thresholds = precision_recall_curve(y, scores, pos_label=1)
        precision_df = pd.concat([precision_df,pd.DataFrame({str(i):precision.tolist()})], axis=1)
        recall_df = pd.concat([recall_df,pd.DataFrame({str(i):recall.tolist()})],axis=1) 
        i=i+1
    precision_df.to_csv(path_or_buf=path+'\\precision.csv')
    recall_df.to_csv(path_or_buf=path+'\\recall.csv')
    return precision_df, recall_df

def compute_ROC(df_mask, df_result, path):
    i=0
    fpr_df = pd.DataFrame()
    tpr_df = pd.DataFrame()
    
    for name in df_result.columns:
        y = df_mask[name].values.astype(int)
        scores = df_result[name].values
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        fpr_df = pd.concat([fpr_df, pd.DataFrame({str(i):fpr.tolist()})], axis=1)
        tpr_df = pd.concat([tpr_df, pd.DataFrame({str(i):tpr.tolist()})], axis=1) 
        i=i+1
    fpr_df.to_csv(path_or_buf=path+'\\FPR.csv')
    tpr_df.to_csv(path_or_buf=path+'\\TPR.csv')
    return fpr_df, tpr_df


def plot_curve_PRC(precision_df, recal_df, path):
    
    plt.figure(1, figsize=(16,8))
    
    for name in precision_df.columns:
        plt.plot(recal_df[name], precision_df[name], linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc='best')
    plt.legend(['Image 0', 'Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'], loc='best')
    plt.rc('font', size=22)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    
    plt.savefig(path+'\\PRC.jpg')
    return

def plot_curve_ROC(fpr_df, tpr_df, path):
    
    plt.figure(1, figsize=(16,8))
    
    for name in fpr_df.columns:
        plt.plot(fpr_df[name], tpr_df[name], linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('1 - Specificity (False positive rate)')
    plt.ylabel('Sensitivity (True positive rate)')
    plt.title('ROC')
    plt.legend(['Image 0', 'Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'], loc='best')
    plt.rc('font', size=22)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(path+'\\ROC.jpg')
    plt.close()
    return

def print_score(df_score, path):
    mean_f1 = df_score.values.mean()
    std2_f1 = 2*df_score.values.std()

    mean_f1 = ("%.2f" % mean_f1)
    std2_f1 = ("%.2f" % std2_f1)
    print(path + ' : ' + str(mean_f1) + ' +/- ' + str(std2_f1))
	
# directory paths

list_result_dir = glob.glob('*unet*')
original_mask_dir = glob.glob('OriginalMasks')

mask_dict = return_dict_data('OriginalMasks')
mask_df = normalize_to_df(mask_dict)
i=1
Ria_all = pd.DataFrame()
F1_all = pd.DataFrame()

for path in list_result_dir:
    prediction_dict = return_dict_data(path)
    prediction_df = normalize_to_df(prediction_dict)
        
    RIA_df = compute_rand_index_adjusted(mask_df, prediction_df, path)
    RIA_df.to_csv(path_or_buf=path+'\\RIA_score.csv')
    print_score(RIA_df, path)
    Ria_all = Ria_all.append(RIA_df)
        
    f1_score_df = compute_f1_score(mask_df, prediction_df,path, threshold=0.8)
    f1_score_df.to_csv(path_or_buf=path+'\\f1_score.csv')
    print_score(f1_score_df, path)
    F1_all = F1_all.append(f1_score_df)
    
    precision_df, recall_df = compute_precision_recal(mask_df, prediction_df, path)
    plot_curve_PRC(precision_df, recall_df, path)
    
    fpr_df, tpr_df = compute_ROC(mask_df, prediction_df, path)
    plot_curve_ROC(fpr_df, tpr_df, path)
    
    print(str(round(i*2.43))+' %', end = '\r')
    i=i+1

	unet10ki10 = Ria_all[0:10].values.tolist()
unet10ki15 = Ria_all[10:15].values.tolist()
unet10ki20 = Ria_all[15:20].values.tolist()
unet1k = Ria_all[21:31].values.tolist()
unet5k = Ria_all[31:41].values.tolist()


plt.figure(1, figsize=(16,8))


plt.ylabel('RI')
plt.title('Boxplots Rand Index')
labels = ['RDZ1K', 'RDZ5K', 'RDZ10KI10', 'RDZ10KI15', 'RDZ10KI20']
plt.rc('font', size=22)
plt.ylim((0, 1))
plt.boxplot([unet1k, unet5k, unet10ki10, unet10ki15, unet10ki20], labels=labels)
plt.savefig('C:\\Users\\smocko\\Desktop\\FinalResults\\RI_boxplots.png')
plt.close()

unet10ki10 = F1_all[0:10].values.tolist()
unet10ki15 = F1_all[10:15].values.tolist()
unet10ki20 = F1_all[15:20].values.tolist()
unet1k = F1_all[21:31].values.tolist()
unet5k = F1_all[31:41].values.tolist()


plt.figure(1, figsize=(16,8))


plt.ylabel('Dice')
plt.title('Boxplots Dice')
labels = ['RDZ1K', 'RDZ5K', 'RDZ10KI10', 'RDZ10KI15', 'RDZ10KI20']
plt.rc('font', size=22)
plt.ylim((0, 1))
plt.boxplot([unet1k, unet5k, unet10ki10, unet10ki15, unet10ki20], labels=labels)
plt.savefig('C:\\Users\\smocko\\Desktop\\FinalResults\\Dice_boxplots.png')
plt.close()