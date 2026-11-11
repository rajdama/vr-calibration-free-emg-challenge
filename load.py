import pandas as pd, glob, os, warnings, sys
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
def rd(f):
    d=pd.read_excel(f); d['file']=os.path.basename(f).split('.')[0]; return d
for s in ['train','test']:
    fs=sorted(glob.glob(f'data/{s}/*.xlsx'))
    ds=Parallel(n_jobs=2)(delayed(rd)(f) for f in fs)
    d=pd.concat(ds,ignore_index=True); d.to_pickle(f'{s}.pkl'); print(s,d.shape)
