"""Native-resolution retinal vessel U-Net; no external data or pretrained weights."""
from pathlib import Path
import argparse,csv,json,random,time
import numpy as np
from PIL import Image,ImageOps,ImageEnhance,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/segmentation'
OUT=ROOT/'vessel_unet_results'

def seed(n=42):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def load(split,i,label=False):
 p=DATA/split/('label' if label else 'image')/f'{i}.jpg'
 a=Image.open(p)
 if label:return np.asarray(a.convert('L'))>127
 a=np.asarray(a.convert('RGB'),dtype=np.float32)/255
 # Local green-channel contrast provides a fourth input channel.
 from scipy.ndimage import gaussian_filter
 g=a[:,:,1];local=np.clip((g-gaussian_filter(g,10))*.8+.5,0,1)
 return np.concatenate([a,local[:,:,None]],axis=2)

class Block(nn.Module):
 def __init__(self,a,b):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU(inplace=True))
 def forward(self,x):return self.net(x)
class UNet(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.ModuleList([Block(4,16),Block(16,32),Block(32,64),Block(64,128)]);self.dec=nn.ModuleList([Block(192,64),Block(96,32),Block(48,16)]);self.head=nn.Conv2d(16,1,1)
 def forward(self,x):
  skips=[]
  for k,b in enumerate(self.enc):
   x=b(x);skips.append(x)
   if k<3:x=F.max_pool2d(x,2)
  for b,s in zip(self.dec,reversed(skips[:-1])):x=b(torch.cat([F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1))
  return self.head(x)

def batch(images,labels,batch_size=12,size=128):
 xs=[];ys=[]
 for _ in range(batch_size):
  i=random.randrange(len(images));a=images[i];b=labels[i];h,w=b.shape
  y=random.randrange(h-size+1);x=random.randrange(w-size+1)
  a=a[y:y+size,x:x+size].copy();b=b[y:y+size,x:x+size].copy()
  k=random.randrange(4);a=np.rot90(a,k);b=np.rot90(b,k)
  if random.random()<.5:a=a[:,::-1];b=b[:,::-1]
  a=a.copy();a[:,:,:3]=np.clip(a[:,:,:3]*random.uniform(.8,1.2)**1+random.uniform(-.035,.035),0,1)
  xs.append(a.transpose(2,0,1));ys.append(b[None])
 return torch.from_numpy(np.stack(xs)).float().cuda(),torch.from_numpy(np.stack(ys)).float().cuda()

@torch.inference_mode()
def predict(m,a,tta=False):
 m.eval();h,w=a.shape[:2];x=torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).cuda();x=F.pad(x,(0,(-w)%8,0,(-h)%8),mode='reflect')
 preds=[]
 for dims in ([(),(-1,),(-2,),(-2,-1)] if tta else [()]):
  z=torch.flip(x,dims) if dims else x
  with torch.autocast('cuda'):z=m(z).sigmoid()
  if dims:z=torch.flip(z,dims)
  preds.append(z[0,0,:h,:w].float().cpu().numpy())
 return np.mean(preds,axis=0)

def scores(ps,ys):
 result={}
 for t in np.arange(.2,.71,.05):
  ds=[]
  for p,y in zip(ps,ys):
   z=p>t;ds.append(float(2*np.sum(z&y)/(z.sum()+y.sum()+1e-8)))
  result[f'{t:.2f}']=ds
 key=max(result,key=lambda k:np.mean(result[k]));return float(np.mean(result[key])),float(key),result

def train(ids,epochs,name,validate=True):
 images=[load('train',i) for i in ids];labels=[load('train',i,True) for i in ids]
 vi=[load('train',i) for i in range(17,21)] if validate else []
 vy=[load('train',i,True) for i in range(17,21)] if validate else []
 m=UNet().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');best=-1;history=[]
 for e in range(1,epochs+1):
  m.train();losses=[]
  for group in opt.param_groups:group['lr']=.001*(.1+.9*(1-(e-1)/epochs))
  for step in range(60):
   x,y=batch(images,labels);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda'):
    logits=m(x);p=logits.sigmoid();dice=(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)
    loss=F.binary_cross_entropy_with_logits(logits,y)+.5*(1-dice.mean())
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(loss.item())
  row={'epoch':e,'loss':float(np.mean(losses))}
  if validate:
   ds,t,detail=scores([predict(m,a) for a in vi],vy);row.update(dice=ds,threshold=t)
   if ds>best:
    best=ds;torch.save(m.state_dict(),OUT/f'{name}.pt');(OUT/'validation_selection.json').write_text(json.dumps({'epoch':e,'dice':ds,'threshold':t,'per_threshold':detail},indent=2))
  else:torch.save(m.state_dict(),OUT/f'{name}.pt')
  history.append(row);(OUT/f'{name}_history.json').write_text(json.dumps(history,indent=2));print(name,json.dumps(row),flush=True)
 return m

def rle(mask):
 a=mask.reshape(-1).astype(np.int8);q=np.flatnonzero(np.diff(np.r_[0,a,0]))+1;q[1::2]-=q[::2];return ' '.join(map(str,q))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--stage',choices=['all','validate','final'],default='all');args=ap.parse_args();OUT.mkdir(exist_ok=True);seed();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
 if args.stage in ['all','validate']:train(list(range(1,17)),args.epochs,'holdout',True)
 if args.stage=='validate':return
 selection=json.loads((OUT/'validation_selection.json').read_text());m=UNet().cuda();m.load_state_dict(torch.load(OUT/'holdout.pt',weights_only=True));vi=[load('train',i) for i in range(17,21)];vy=[load('train',i,True) for i in range(17,21)]
 # Match the previous baseline's 512x512 evaluation grid.
 ps=[predict(m,a,True) for a in vi];native=scores(ps,vy)
 ps512=[np.asarray(Image.fromarray(p).resize((512,512),Image.Resampling.BILINEAR)) for p in ps];ys512=[np.asarray(Image.fromarray(y).resize((512,512),Image.Resampling.NEAREST)) for y in vy];ds,t,detail=scores(ps512,ys512)
 report={'split':'train 1-16, validation 17-20 (same as baseline)','baseline_dice':.7507341018908654,'unet_512_dice':ds,'unet_native_dice':native[0],'threshold':t,'per_threshold_512':detail,'selected_epoch':selection['epoch'],'tta':'four flips','note':'Validation is used for epoch and threshold selection; not an unbiased test score.'};(OUT/'validation.json').write_text(json.dumps(report,indent=2));print('VALIDATION',json.dumps(report),flush=True)
 seed(43);m=train(list(range(1,21)),selection['epoch'],'full',False)
 (OUT/'image').mkdir(exist_ok=True);(OUT/'native_masks').mkdir(exist_ok=True);rows=[];vrows=[]
 for i in range(1,21):
  a=load('test',i);p=predict(m,a,True);Image.fromarray(np.where(p>t,0,255).astype('uint8')).save(OUT/f'native_masks/{i}.png');p=np.asarray(Image.fromarray(p).resize((512,512),Image.Resampling.BILINEAR));z=p>t
  Image.fromarray(np.where(z,0,255).astype('uint8')).save(OUT/f'image/{i}.png');rows.append([str(i),rle(~z)]);vrows.append([str(i),rle(z)]);print('saved',i,flush=True)
 for name,rs in [('submission.csv',rows),('submission_vessels.csv',vrows)]:
  with open(OUT/name,'w',newline='') as f:w=csv.writer(f);w.writerow(['Id','Predicted']);w.writerows(rs)
 # Decode every row and verify it exactly matches the delivered binary mask.
 for name,vessel in [('submission.csv',False),('submission_vessels.csv',True)]:
  rs=list(csv.DictReader(open(OUT/name)));assert len(rs)==20
  for row in rs:
   q=np.fromstring(row['Predicted'],sep=' ',dtype=int).reshape(-1,2);assert np.all(q>0);assert np.all(q[1:,0]>=q[:-1,0]+q[:-1,1]);decoded=np.zeros(512*512,bool)
   for start,length in q:assert start+length-1<=len(decoded);decoded[start-1:start-1+length]=True
   mask=np.asarray(Image.open(OUT/f"image/{row['Id']}.png"));assert set(np.unique(mask))<={0,255};assert np.array_equal(decoded.reshape(512,512),mask== (0 if vessel else 255))
 print('All 40 RLE rows round-trip verified.',flush=True)
if __name__=='__main__':main()
