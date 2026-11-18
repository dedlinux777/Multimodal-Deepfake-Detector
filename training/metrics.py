"""Branch-wise classification metrics and collapse diagnostics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,confusion_matrix,
                             f1_score,precision_score,recall_score,roc_auc_score)

def classification_metrics(labels, probabilities) -> dict:
    y=np.asarray(labels,dtype=int); p=np.asarray(probabilities,dtype=float)
    pred=p.argmax(1); cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    try: auc=float(roc_auc_score(y,p[:,1])) if len(np.unique(y))==2 else float("nan")
    except ValueError: auc=float("nan")
    return {"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),"specificity":float(tn/(tn+fp)) if tn+fp else 0.,
        "f1":float(f1_score(y,pred,zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "roc_auc":auc,"real_recall":float(tn/(tn+fp)) if tn+fp else 0.,
        "fake_recall":float(tp/(tp+fn)) if tp+fn else 0.,"confusion_matrix":cm.tolist(),"predicted_real":int((pred==0).sum()),
        "predicted_fake":int((pred==1).sum()),"mean_fake_probability":float(p[:,1].mean()),
        "fake_probability_min":float(p[:,1].min()),"fake_probability_max":float(p[:,1].max()),
        "fake_probability_std":float(p[:,1].std())}

def log_diagnostics(metrics:dict, branch:str, logger=print)->None:
    logger(f"{branch}: REAL={metrics['predicted_real']} FAKE={metrics['predicted_fake']} "
           f"mean_fake_probability={metrics['mean_fake_probability']:.4f}")
    if metrics['predicted_real']==0 or metrics['predicted_fake']==0: logger(f"WARNING: {branch} predicts only one class")
    if metrics['fake_probability_max']-metrics['fake_probability_min']<1e-3 or metrics['fake_probability_std']<1e-4:
        logger(f"WARNING: {branch} fake probabilities collapsed into a narrow range")
