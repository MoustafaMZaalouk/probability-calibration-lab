"""Small, reproducible educational experiment. See README for limitations."""
import json
from pathlib import Path
import numpy as np

def save(result, output):
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

def ece(y,p,bins=10):
    bucket=np.minimum((p*bins).astype(int),bins-1); total=0.
    for i in range(bins):
        mask=bucket==i
        if mask.any(): total+=mask.mean()*abs(float(y[mask].mean()-p[mask].mean()))
    return float(total)
def run(output="results.json"):
    X,y=make_classification(n_samples=3000,n_features=16,n_informative=8,n_redundant=3,weights=[.7,.3],flip_y=.08,class_sep=.8,random_state=42)
    Xdev,Xte,ydev,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    Xtr,Xcal,ytr,ycal=train_test_split(Xdev,ydev,test_size=1/3,stratify=ydev,random_state=43)
    base=RandomForestClassifier(n_estimators=160,max_depth=5,min_samples_leaf=8,random_state=42,n_jobs=1).fit(Xtr,ytr)
    calibrated=CalibratedClassifierCV(FrozenEstimator(base),method="sigmoid").fit(Xcal,ycal)
    rows={}
    for name,m in [("forest",base),("sigmoid",calibrated)]:
        p=m.predict_proba(Xte)[:,1]
        rows[name]={"brier":float(brier_score_loss(yte,p)),"log_loss":float(log_loss(yte,p)),"roc_auc":float(roc_auc_score(yte,p)),"ece_10_bins":ece(yte,p)}
    return save({"seed":42,"train_rows":len(ytr),"calibration_rows":len(ycal),"test_rows":len(yte),"models":rows},output)

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
