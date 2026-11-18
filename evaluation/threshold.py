"""Validation-only operating threshold calibration."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

def calibrate_threshold(labels, fake_probability, method="max_balanced_accuracy", operating_point=None):
    """Select threshold using validation labels only; ties prefer threshold 0.5.

    Methods: max_balanced_accuracy, max_f1, or recall_at_least with target
    recall supplied as operating_point (e.g. 0.90). Predictions use p_fake >= t.
    """
    y=np.asarray(labels,dtype=int); p=np.asarray(fake_probability,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not len(y): raise ValueError("labels/probabilities must be nonempty equal-length vectors")
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid labels or probabilities")
    candidates=np.unique(np.r_[0.,p,1.]); rows=[]
    for t in candidates:
        pred=(p>=t).astype(int)
        b=float(balanced_accuracy_score(y,pred)); f=float(f1_score(y,pred,zero_division=0)); r=float(recall_score(y,pred,zero_division=0))
        rows.append((float(t),b,f,r))
    if method=="max_balanced_accuracy": key=lambda r:(r[1],-abs(r[0]-.5))
    elif method=="max_f1": key=lambda r:(r[2],-abs(r[0]-.5))
    elif method=="recall_at_least":
        if operating_point is None or not 0<operating_point<=1: raise ValueError("recall_at_least requires operating_point in (0,1]")
        eligible=[r for r in rows if r[3]>=operating_point]
        if not eligible: raise ValueError("No validation threshold satisfies requested recall")
        chosen=max(eligible,key=lambda r:(r[1],-abs(r[0]-.5)))
        return _result(chosen,y,p,method,operating_point)
    else: raise ValueError(f"Unsupported threshold method {method}")
    return _result(max(rows,key=key),y,p,method,operating_point)

def _result(row,y,p,method,operating_point):
    t,b,f,r=row; pred=(p>=t).astype(int)
    return {"threshold":t,"method":method,"operating_point":operating_point,"sample_count":int(len(y)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),"f1":float(f1_score(y,pred,zero_division=0)),
            "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0))}
