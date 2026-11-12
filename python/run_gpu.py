"""Optional real Triton validation: requires PyTorch, Triton and a CUDA GPU."""
import argparse, importlib.util, subprocess, tempfile
from pathlib import Path
import torch
import triton
p=argparse.ArgumentParser();p.add_argument('--binary',default='build/kforge');a=p.parse_args()
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'generated.py';path.write_text(subprocess.check_output([a.binary],text=True))
 spec=importlib.util.spec_from_file_location('generated',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for n in (1,127,1024,1025):
  x=torch.randn(n,device='cuda');y=torch.randn(n,device='cuda');out=torch.empty_like(x)
  module.kernel[(triton.cdiv(n,256),)](x,y,out,n,256)
  torch.testing.assert_close(out,torch.relu((x+y)*y))
 print('Triton numerical checks passed')
