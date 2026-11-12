"""Check syntax and execute emitted code with a tiny vectorized Triton test double.
This validates generated arithmetic and tail masking, not GPU compilation.
"""
import ast, subprocess, sys, types
source=subprocess.check_output([sys.argv[1]],text=True);ast.parse(source)
class V(list):
 def __add__(self,o):return V(a+b for a,b in zip(self,o)) if isinstance(o,list) else V(a+o for a in self)
 __radd__=__add__
 def __mul__(self,o):return V(a*b for a,b in zip(self,o)) if isinstance(o,list) else V(a*o for a in self)
 def __lt__(self,o):return V(a<o for a in self)
class Ptr:
 def __init__(self,data,offset=None):self.data,self.offset=data,offset
 def __add__(self,offset):return Ptr(self.data,offset)
def load(p,mask,other):return V(p.data[i] if m else other for i,m in zip(p.offset,mask))
def store(p,v,mask):
 for i,x,m in zip(p.offset,v,mask):
  if m:p.data[i]=x
triton=types.ModuleType('triton');triton.jit=lambda f:f
tl=types.ModuleType('triton.language');tl.constexpr=int;tl.program_id=lambda axis:0;tl.arange=lambda a,b:V(range(a,b));tl.load=load;tl.store=store;tl.maximum=lambda a,b:V(max(x,b) for x in a)
sys.modules['triton']=triton;sys.modules['triton.language']=tl
ns={};exec(source,ns)
a=[-3,2,1,-2,5];b=[1,3,-4,-1,2];out=[None]*5
ns['kernel'](Ptr(a),Ptr(b),Ptr(out),5,8)
assert out==[max((x+y)*y,0) for x,y in zip(a,b)],out
print('Generated source syntax, arithmetic and masked tail passed')
