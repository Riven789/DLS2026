"""Commands: generate, train, evaluate, predict. See README for array contracts."""
import argparse,csv,json,platform
from pathlib import Path
import numpy as np
import data_api as api
from model import Model
from common import load_npz,save_json,digest,split,take,fit,metrics,softmax

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate'); g.add_argument('--out',default='data'); g.add_argument('--n',type=int,default=None); g.add_argument('--seed',type=int,default=42)
    t=sub.add_parser('train'); t.add_argument('--data',required=True); t.add_argument('--out',required=True); t.add_argument('--task',choices=['classification','regression'],default='classification')
    t.add_argument('--epochs',type=int,default=80); t.add_argument('--batch',type=int,default=32); t.add_argument('--lr',type=float,default=.003); t.add_argument('--patience',type=int,default=15); t.add_argument('--seed',type=int,default=7); t.add_argument('--weight-decay',type=float,default=1e-4); t.add_argument('--clip',type=float,default=5.)
    for name in ['evaluate','predict']:
        q=sub.add_parser(name); q.add_argument('--data',required=True); q.add_argument('--run',required=True)
        if name=='predict': q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='generate':
        out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
        for task in ['classification','regression']:
            kwargs={'task':task,'seed':a.seed}
            if a.n is not None: kwargs['n']=a.n
            d=api.generate(**kwargs); api.validate(d); np.savez_compressed(out/f'{task}.npz',**d)
        d=api.generate('classification',n=6,seed=a.seed+100)
        np.savez_compressed(out/'unlabeled.npz',**{k:v for k,v in d.items() if k!='y'})
        print(f'{api.NAME}: generated data in {out}'); return
    d=load_npz(a.data); api.validate(d,labeled=a.command!='predict')
    if a.command=='train':
        out=Path(a.out)
        if out.exists() and any(out.iterdir()): p.error('Choose an empty output directory')
        y=d['y']; task=a.task
        if y.shape!=(len(d['X']),) or not np.isfinite(y).all(): p.error('Need one finite target per example')
        if task=='classification' and (not np.array_equal(y,y.astype(int)) or not np.array_equal(np.unique(y),np.arange(len(np.unique(y)))) or len(np.unique(y))<2): p.error('Labels must be contiguous integer IDs 0..K-1, K>=2')
        if 'split' in d:
            sp=d['split']
            if sp.shape!=y.shape or not np.isin(sp,[0,1,2]).all(): p.error('split must have one 0/1/2 per example')
            parts={name:np.flatnonzero(sp==i).tolist() for i,name in enumerate(['train','val','test'])}
            if any(not v for v in parts.values()): p.error('Every split must be nonempty')
        else: parts=split(y,task,a.seed)
        if task=='classification' and set(y[parts['train']])!=set(y): p.error('Training must contain every class')
        state=api.fit_pre(d,parts['train']); inputs=api.transform(d,state)
        target={'mean':0.,'scale':1.}
        if task=='regression': target={'mean':float(y[parts['train']].mean()),'scale':max(float(y[parts['train']].std()),1e-8)}
        yt=(y-target['mean'])/target['scale'] if task=='regression' else y.astype(int)
        cfg=api.config(d,1 if task=='regression' else len(np.unique(y))); net=Model(cfg,a.seed)
        hist,best=fit(net,inputs,yt,parts,task,a.epochs,a.batch,a.lr,a.patience,a.seed,a.weight_decay,a.clip)
        out.mkdir(parents=True,exist_ok=True); net.save(out/'weights.npz')
        baseline=float(y[parts['train']].mean()) if task=='regression' else int(np.bincount(yt[parts['train']]).argmax())
        names=api.class_names() if task=='classification' and cfg['outputs']==len(api.class_names()) else [str(i) for i in range(cfg['outputs'])]
        meta=dict(model=api.NAME,config=cfg,preprocessing=state,target=target,task=task,split=parts,data_sha256=digest(a.data),best_epoch=best,baseline=baseline,class_names=names,arguments=vars(a),versions={'python':platform.python_version(),'numpy':np.__version__})
        save_json(out/'metadata.json',meta); save_json(out/'history.json',hist)
        print(json.dumps({'model':api.NAME,'task':task,'best_epoch':best,'val_loss':hist[best-1]['val_loss'],'parameters':sum(v.size for v in net.params.values())})); return
    out=Path(a.run); meta=json.loads((out/'metadata.json').read_text())
    if meta['model']!=api.NAME: p.error('Wrong model family for this checkpoint')
    model=Model(meta['config']).load(out/'weights.npz'); task=meta['task']
    if a.command=='evaluate':
        if digest(a.data)!=meta['data_sha256']: p.error('Changed dataset: held-out indices require original file')
        ids=meta['split']['test']; y=d['y'][ids]; inputs=api.transform(take(d,ids),meta['preprocessing'])
    else: inputs=api.transform(d,meta['preprocessing'])
    # Bound evaluation memory and preserve incomplete final batches.
    z=np.concatenate([model.forward(take(inputs,np.arange(i,min(i+64,len(inputs['X']))))) for i in range(0,len(inputs['X']),64)])
    if task=='regression': z=z*meta['target']['scale']+meta['target']['mean']
    if a.command=='predict':
        with open(a.out,'w',newline='') as f:
            writer=csv.writer(f)
            if task=='regression': writer.writerow(['prediction']); writer.writerows(z)
            else:
                prob=softmax(z); writer.writerow(['class_id','class_name']+['p_'+s for s in meta['class_names']]); writer.writerows([[int(v.argmax()),meta['class_names'][int(v.argmax())],*v.tolist()] for v in prob])
        print(f'Wrote {len(z)} predictions'); return
    if task=='regression': base=np.full_like(z,meta['baseline'])
    else: base=np.zeros_like(z); base[:,meta['baseline']]=1
    result={'n_test':len(y),'best_epoch':meta['best_epoch'],'model':metrics(y,z,task),'baseline':metrics(y,base,task)}
    save_json(out/'test_metrics.json',result)
    np.savez_compressed(out/'test_predictions.npz',y=y,prediction=z)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
