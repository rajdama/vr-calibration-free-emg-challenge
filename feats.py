import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.signal import welch
from joblib import Parallel, delayed
ch=[f'ch{i}' for i in range(1,9)]; im=['ax','ay','az','gx','gy','gz']
W=1.4
def feat(t,E,I,imu_new):
    t=t-t[0]; f={}
    n=len(E); f['n']=n
    E=E-E.mean(0)
    k=np.ones(9)/9
    ma=np.column_stack([np.convolve(np.pad(E[:,c],4,mode='edge'),k,'valid') for c in range(8)])
    hp=E-ma; d=np.diff(E,axis=0)
    rms=np.sqrt((hp**2).mean(0))+1e-6; drms=np.sqrt((d**2).mean(0))+1e-6
    # robust: median abs
    mad=np.median(np.abs(hp),0)+1e-6
    zc=((hp[1:]*hp[:-1])<0).mean(0)
    lf=ma.std(0)+1e-6
    q=np.minimum((t/W*4).astype(int),3)
    fr,P=welch(hp,fs=250,nperseg=min(64,n),axis=0)
    bands=[(5,20),(20,45),(45,55),(55,80),(80,125)]
    for c in range(8):
        f[f'rms{c}']=np.log(rms[c]); f[f'drms{c}']=np.log(drms[c]); f[f'mad{c}']=np.log(mad[c])
        f[f'zc{c}']=zc[c]; f[f'lf{c}']=np.log(lf[c]); f[f'kurt{c}']=np.log(rms[c]/mad[c])
        f[f'slope{c}']=(ma[-n//4:,c].mean()-ma[:n//4,c].mean())/rms[c]
        for b,(lo,hi) in enumerate(bands):
            f[f'bp{c}_{b}']=np.log(P[(fr>=lo)&(fr<hi),c].mean()+1e-6)
        for j in range(4):
            m=q==j
            f[f'q{c}_{j}']=np.log(np.sqrt((hp[m,c]**2).mean())+1e-6)-np.log(rms[c]) if m.sum()>3 else 0
    lr=np.log(rms); 
    for c in range(8): f[f'rel{c}']=lr[c]-lr.mean()
    f['rms_mean']=lr.mean(); f['rms_sd']=lr.std()
    C=np.corrcoef(hp.T); iu=np.triu_indices(8,1)
    for a,b,v in zip(*iu,C[iu]): f[f'cor{a}_{b}']=v
    env=np.abs(hp); env=np.column_stack([np.convolve(env[:,c],np.ones(25)/25,'same') for c in range(8)])
    Ce=np.nan_to_num(np.corrcoef(env.T))
    for a,b,v in zip(*iu,Ce[iu]): f[f'ecor{a}_{b}']=v
    # IMU
    m=imu_new>0; I=I[m]; ti=t[m]; I=I-I.mean(0); qi=np.minimum((ti/W*4).astype(int),3)
    for a in range(6):
        x=I[:,a]; f[f'i_sd{a}']=np.log(x.std()+1e-4); f[f'i_rg{a}']=x.max()-x.min()
        f[f'i_ad{a}']=np.log(np.abs(np.diff(x)).mean()+1e-5) if len(x)>2 else 0
        for j in range(4):
            mm=qi==j; f[f'i_q{a}_{j}']=x[mm].mean() if mm.sum() else 0
            f[f'i_qs{a}_{j}']=np.log(x[mm].std()+1e-4) if mm.sum()>2 else 0
        f[f'i_am{a}']=np.argmax(np.abs(x))/max(len(x),1)
    f['g_tot']=np.log(np.sqrt((I[:,3:]**2).sum(1).mean())+1e-4); f['a_tot']=np.log(np.sqrt((I[:,:3]**2).sum(1).mean())+1e-5)
    return f
def train_file(g):
    out=[]; t=g.t.values; E=g[ch].values; I=g[im].values; nw=g.imu_new.values
    lab=g.lab.values; src=g.label_source.values
    b=np.flatnonzero(np.r_[True,lab[1:]!=lab[:-1]]); e=np.r_[b[1:],len(g)]
    for ri,(s,en) in enumerate(zip(b,e)):
        if lab[s]=='none': continue
        dur=t[en-1]-t[s]
        if dur<1.2: continue
        offs=np.arange(0,max(dur-W,0)+0.01,0.2) if dur>W else [0]
        for o in offs:
            i0=np.searchsorted(t,t[s]+o); i1=min(np.searchsorted(t,t[s]+o+W),en)
            if i1-i0<60: continue
            f=feat(t[i0:i1],E[i0:i1],I[i0:i1],nw[i0:i1])
            f.update(file=g.file.iloc[0],label=lab[s],src=src[s],off=o,run=ri,dur=dur); out.append(f)
    return out
def test_seg(g):
    f=feat(g.t.values,g[ch].values,g[im].values,g.imu_new.values); f.update(file=g.file.iloc[0],action_id=g.action_id.iloc[0]); return f
if __name__=='__main__':
    tr=pd.read_pickle('train.pkl'); te=pd.read_pickle('test.pkl'); tr['lab']=tr.label.fillna('none')
    r=Parallel(n_jobs=2)(delayed(train_file)(g) for _,g in tr.groupby('file'))
    F=pd.DataFrame([x for y in r for x in y]); F.to_pickle('Ftr.pkl'); print(F.shape); print(F.groupby(['label','src']).size())
    T=pd.DataFrame([test_seg(g) for _,g in te.groupby('action_id')]); T.to_pickle('Fte.pkl'); print(T.shape)
