import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from scipy.cluster.hierarchy import fcluster, linkage
F=pd.read_pickle('Ftr.pkl'); F=F[F.src=='guided']; T=pd.read_pickle('Fte.pkl')
F=F[((F.label!='rest')&(F.off>0.35)&(F.off<1.45))|((F.label=='rest')&(F.dur>1.4))].reset_index(drop=True)
meta=['file','label','src','off','run','dur','action_id','n']
cols=[c for c in F.columns if c not in meta]
ev=(((F.label!='rest')&F.off.round(1).isin([0.8,1.0]))|((F.label=='rest')&(F.off==0))).values
def z(D,ref):
    X=D[cols]; R=X[ref]; g=D.file
    return ((X-R.groupby(g[ref]).median().reindex(g).values)/(R.groupby(g[ref]).std().reindex(g).values+1e-6)).clip(-6,6).values
X=np.hstack([z(F,ev),F[cols].values]); Xt=np.hstack([z(T,np.ones(len(T),bool)),T[cols].values])
y=F.label.values; cls=np.unique(y); w=pd.Series(y).map(1/pd.Series(y).value_counts()).values
# subject-like groups: merge recordings with near-identical channel noise profile
tr=pd.read_pickle('train.pkl'); ch=[f'ch{i}' for i in range(1,9)]
d=np.log(tr.groupby('file')[ch].apply(lambda x:x.diff().abs().median()))
cl=pd.Series(fcluster(linkage(d.values,'single'),0.85,'distance'),d.index); print('groups',cl.nunique(),'from',len(cl))
grp=F.file.map(cl).values
et=lambda s:ET(800,min_samples_leaf=2,n_jobs=2,max_features=0.2,random_state=s)
P=np.zeros((len(y),6))
for a,b in GroupKFold(min(7,cl.nunique())).split(X,y,grp):
    P[b]=et(0).fit(X[a],y[a],sample_weight=w[a]).predict_proba(X[b])
p=cls[P[ev].argmax(1)]
print('subject-grouped CV macro-F1 %.3f'%f1_score(y[ev],p,average='macro'))
print(pd.Series(f1_score(y[ev],p,average=None,labels=cls),cls).round(3).to_dict())
pf=[f1_score(y[ev][F.file[ev]==f],p[F.file[ev]==f],average='macro') for f in F.file.unique()]
print('per-recording F1: min %.2f median %.2f max %.2f'%(min(pf),np.median(pf),max(pf)))
Pt=sum(et(s).fit(X,y,sample_weight=w).predict_proba(Xt) for s in range(3))/3
T['gesture']=cls[Pt.argmax(1)]
ss=pd.read_csv('data/sample_submission.csv')
sub=ss[['action_id']].merge(T[['action_id','gesture']],on='action_id',how='left')
assert sub.gesture.notna().all() and len(sub)==len(ss) and sub.action_id.is_unique
sub.to_csv('submission.csv',index=False)
print(sub.gesture.value_counts().to_dict()); print(pd.crosstab(T.file,T.gesture)); print('mean max prob %.2f'%Pt.max(1).mean())
