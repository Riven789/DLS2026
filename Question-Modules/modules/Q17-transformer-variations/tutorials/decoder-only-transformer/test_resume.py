"""CPU integration check: interrupted training matches uninterrupted training."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from corpus import write_corpus

class ResumeTest(unittest.TestCase):
    def test_resume_preserves_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data.txt'; write_corpus(data)
            base=[sys.executable,str(Path(__file__).with_name('train.py')),'--data',str(data),
                '--steps','8','--context','16','--width','16','--layers','1','--heads','2',
                '--batch-size','2','--eval-batches','1','--eval-every','4','--warmup','2',
                '--device','cpu','--threads','1','--dropout','0.1']
            subprocess.run(base+['--out',str(root/'full')],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(base+['--out',str(root/'split'),'--stop-after','4'],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(base+['--out',str(root/'split'),'--resume',str(root/'split/last.pt')],check=True,stdout=subprocess.DEVNULL)
            full=torch.load(root/'full/last.pt',weights_only=True)
            split=torch.load(root/'split/last.pt',weights_only=True)
            for name,weight in full['model'].items():
                torch.testing.assert_close(weight,split['model'][name],rtol=0,atol=0)
            self.assertEqual(full['step'],split['step'])

if __name__=='__main__': unittest.main(verbosity=2)
