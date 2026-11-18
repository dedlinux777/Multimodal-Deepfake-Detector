"""Calibrate on validation data, then evaluate the selected best checkpoint on test."""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from dataset.video_dataset import FakeAVCelebDataset
from models.etmc import ETMC
from training.checkpoint import load_checkpoint
from .threshold import calibrate_threshold

BRANCH_TARGET={"audio":"audio_label","visual":"visual_label","pseudo":"overall_label","fused":"overall_label"}
BRANCH_OUTPUT={"audio":"audio_probabilities","visual":"visual_probabilities","pseudo":"pseudo_probabilities","fused":"fused_probabilities"}

def _safe(v):
    if isinstance(v,dict):return {str(k):_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [_safe(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,float)) and not np.isfinite(v):return None
    if isinstance(v,np.floating):return float(v)
    return v

def collect_predictions(model, manifest:pd.DataFrame, config:dict, device, batch_size:int, num_workers:int=0):
    """Run sequentially so per-item decode failures are captured, never hidden."""
    d=config["data"]; ds=FakeAVCelebDataset(manifest,d.get("video_num_frames",8),d.get("image_size",256),
         d.get("audio_sample_rate",16000),d.get("audio_num_samples",64000)/d.get("audio_sample_rate",16000))
    rows=[]; failures=[]; model.eval()
    with torch.inference_mode():
        for i in range(len(ds)):
            source=manifest.iloc[i]
            try:
                item=ds[i]; audio=item["audio"].unsqueeze(0).to(device); frames=item["frames"].unsqueeze(0).to(device)
                out=model(audio,frames)
            except Exception as exc:
                failures.append({"video_path":str(source.get("video_path","")),"error":f"{type(exc).__name__}: {exc}"}); continue
            row={"video_path":str(source.video_path),"audio_label":int(source.audio_label),
                 "visual_label":int(source.visual_label),"overall_label":int(source.overall_label)}
            for key in ("category","group_id","generator","dataset","domain"):
                if key in source.index: row[key]=source[key]
            for branch,key in BRANCH_OUTPUT.items():
                prob=getattr(out,key)
                if prob is not None and torch.isfinite(prob[0]).all():
                    row[f"{branch}_p_real"]=float(prob[0,0]); row[f"{branch}_p_fake"]=float(prob[0,1])
                else: row[f"{branch}_p_real"]=np.nan; row[f"{branch}_p_fake"]=np.nan
            rows.append(row)
    return pd.DataFrame(rows),failures

def probability_diagnostics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    return {"sample_count":len(y),"predicted_real":int((p<.5).sum()),"predicted_fake":int((p>=.5).sum()),
        "fake_probability_min":float(p.min()),"fake_probability_max":float(p.max()),"fake_probability_mean":float(p.mean()),
        "fake_probability_std":float(p.std()),"percentiles":{"1":float(np.percentile(p,1)),"5":float(np.percentile(p,5)),
        "25":float(np.percentile(p,25)),"50":float(np.percentile(p,50)),"75":float(np.percentile(p,75)),"95":float(np.percentile(p,95)),"99":float(np.percentile(p,99))},
        "by_ground_truth":{"REAL":{"count":int((y==0).sum()),"mean_fake_probability":float(p[y==0].mean()) if (y==0).any() else None},
        "FAKE":{"count":int((y==1).sum()),"mean_fake_probability":float(p[y==1].mean()) if (y==1).any() else None}}}

def _branch_metrics(frame,branch,threshold):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score
    y=frame[BRANCH_TARGET[branch]].to_numpy(int); p=frame[f"{branch}_p_fake"].to_numpy(float)
    pred=(p>=threshold).astype(int); cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    return {"sample_count":len(y),"threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
      "specificity":float(tn/(tn+fp)) if tn+fp else None,"f1":float(f1_score(y,pred,zero_division=0)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),"roc_auc":auc,"confusion_matrix":cm.tolist(),
      "classification_report":classification_report(y,pred,labels=[0,1],target_names=["REAL","FAKE"],output_dict=True,zero_division=0),
      "probability_diagnostics":probability_diagnostics(y,p)}

def _plots(frame,metrics,thresholds,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(5,4)); cm=np.array(metrics["fused"]["confusion_matrix"])
    ConfusionMatrixDisplay(cm,display_labels=["REAL","FAKE"]).plot(ax=ax,colorbar=False); fig.tight_layout();fig.savefig(out/"confusion_matrix.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for b in BRANCH_TARGET:
        y=frame[BRANCH_TARGET[b]].to_numpy(int);p=frame[f"{b}_p_fake"].to_numpy(float)
        if len(np.unique(y))==2:
            fpr,tpr,_=roc_curve(y,p);ax.plot(fpr,tpr,label=f"{b} (AUC={roc_auc_score(y,p):.3f})")
    ax.plot([0,1],[0,1],"k--");ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Branch ROC curves");ax.legend();fig.tight_layout();fig.savefig(out/"roc_curves.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,b in zip(axes.flat,BRANCH_TARGET):
        for label,name in ((0,"REAL"),(1,"FAKE")):
            vals=frame.loc[frame[BRANCH_TARGET[b]]==label,f"{b}_p_fake"]
            if len(vals):ax.hist(vals,bins=20,alpha=.5,label=name)
        ax.axvline(thresholds[b]["threshold"],color="black",linestyle="--");ax.set_title(b);ax.set_xlabel("P(FAKE)");ax.legend()
    fig.tight_layout();fig.savefig(out/"probability_distributions.png",dpi=150);plt.close(fig)

def evaluate(checkpoint,validation_manifest,test_manifest,config_path,output_dir,threshold_method="max_balanced_accuracy",operating_point=None):
    """Calibrate branch thresholds on validation only; apply unchanged to test."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); dev=cfg["project"].get("device","auto")
    device=torch.device("cuda" if dev=="auto" and torch.cuda.is_available() else ("cpu" if dev=="auto" else dev))
    m=ETMC(pretrained_visual=bool(cfg["model"]["visual"].get("pretrained",True)),freeze_visual=True,
      rawnet_checkpoint=cfg["model"]["audio"].get("checkpoint"),visual_checkpoint=cfg["model"]["visual"].get("checkpoint"),
      prior=float(cfg["model"]["heads"].get("dirichlet_prior",1.))).to(device)
    state=torch.load(checkpoint,map_location=device,weights_only=False); m.load_state_dict(state.get("model",state),strict=True)
    d=cfg["data"]
    val=pd.read_parquet(validation_manifest) if str(validation_manifest).endswith(".parquet") else pd.read_csv(validation_manifest)
    test=pd.read_parquet(test_manifest) if str(test_manifest).endswith(".parquet") else pd.read_csv(test_manifest)
    if "group_id" in val and "group_id" in test and set(val.group_id.astype(str))&set(test.group_id.astype(str)):
        raise ValueError("Validation/test group leakage detected")
    batch=int(cfg.get("evaluation",{}).get("batch_size",cfg["training"].get("batch_size",8)))
    vpred,vfail=collect_predictions(m,val,cfg,device,batch)
    if vpred.empty: raise RuntimeError(f"No validation predictions succeeded; {len(vfail)} files failed. See validation manifest/media.")
    thresholds={}
    for branch,target in BRANCH_TARGET.items():
        ok=vpred[f"{branch}_p_fake"].notna()
        thresholds[branch]=calibrate_threshold(vpred.loc[ok,target],vpred.loc[ok,f"{branch}_p_fake"],threshold_method,operating_point)
    tpred,tfail=collect_predictions(m,test,cfg,device,batch)
    if tpred.empty: raise RuntimeError(f"No test predictions succeeded; {len(tfail)} files failed. See excluded_files.csv.")
    if tfail:
        # Inference failures are explicit in artifact, reported below, never silently excluded.
        pd.DataFrame(tfail).to_csv(Path(output_dir)/"excluded_files.csv",index=False)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    metrics={b:_branch_metrics(tpred.dropna(subset=[f"{b}_p_fake"]),b,thresholds[b]["threshold"]) for b in BRANCH_TARGET}
    predictions=tpred.copy()
    for b in BRANCH_TARGET: predictions[f"{b}_predicted_label"]=(predictions[f"{b}_p_fake"]>=thresholds[b]["threshold"]).astype("Int64")
    predictions.to_csv(out/"predictions.csv",index=False)
    with (out/"classification_report.txt").open("w",encoding="utf-8") as f:
        for b in BRANCH_TARGET:
            subset=tpred.dropna(subset=[f"{b}_p_fake"])
            f.write(f"[{b}] threshold={thresholds[b]['threshold']:.6f}\n"+classification_report(subset[BRANCH_TARGET[b]],(subset[f"{b}_p_fake"]>=thresholds[b]["threshold"]).astype(int),labels=[0,1],target_names=["REAL","FAKE"],zero_division=0)+"\n")
    comparison=pd.DataFrame([{ "branch":b,**{k:v for k,v in m.items() if k!="classification_report" and k!="probability_diagnostics" and k!="confusion_matrix"}} for b,m in metrics.items()]);comparison.to_csv(out/"branch_comparison.csv",index=False)
    report={"checkpoint":str(checkpoint),"validation_threshold_calibration":thresholds,"test":{"sample_count":len(test),"evaluated_count":len(tpred),"excluded_count":len(tfail),"metrics":metrics}}
    if "generator" in tpred or "dataset" in tpred or "domain" in tpred:
        domains=tpred.get("domain",tpred.get("dataset",tpred.get("generator"))).astype(str)
        known=set(val.get("domain",val.get("dataset",val.get("generator",pd.Series(dtype=str)))).astype(str))
        report["robustness"]={"in_distribution":{},"out_of_distribution":{},"domain_column":"domain/dataset/generator"}
        for domain in domains.unique():
            sub=tpred[domains==domain]; dest="in_distribution" if domain in known else "out_of_distribution"
            report["robustness"][dest][domain]={b:_branch_metrics(sub.dropna(subset=[f"{b}_p_fake"]),b,thresholds[b]["threshold"]) for b in BRANCH_TARGET}
    (out/"test_metrics.json").write_text(json.dumps(_safe(report),indent=2,allow_nan=False),encoding="utf-8")
    _plots(tpred,metrics,thresholds,out)
    if vfail:pd.DataFrame(vfail).to_csv(out/"validation_excluded_files.csv",index=False)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--validation",required=True);p.add_argument("--test",required=True);p.add_argument("--config",default="config.yaml");p.add_argument("--output",default="artifacts/evaluation/test");p.add_argument("--threshold",choices=["max_balanced_accuracy","max_f1","recall_at_least"],default="max_balanced_accuracy");p.add_argument("--operating-point",type=float);a=p.parse_args()
    report=evaluate(a.checkpoint,a.validation,a.test,a.config,a.output,a.threshold,a.operating_point)
    print(f"Evaluated {report['test']['evaluated_count']}/{report['test']['sample_count']} test rows; excluded {report['test']['excluded_count']}.")
if __name__=="__main__":main()
