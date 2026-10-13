"""One-command offline lab. Existing run directories are intentionally protected."""
import os,subprocess,sys
from pathlib import Path

def run(*args): subprocess.run([sys.executable,*args],check=True)
if __name__=='__main__':
    os.chdir(Path(__file__).resolve().parent)
    run('-m','unittest','discover','-s','tests','-v')
    run('lab.py','generate')
    for task in ['classification','regression']:
        run('lab.py','train','--task',task,'--data',f'data/{task}.npz','--out',f'runs/{task}')
        run('lab.py','evaluate','--data',f'data/{task}.npz','--run',f'runs/{task}')
        run('lab.py','predict','--data','data/unlabeled.npz','--run',f'runs/{task}','--out',f'runs/{task}/new_predictions.csv')
