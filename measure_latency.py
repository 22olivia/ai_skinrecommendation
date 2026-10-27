# -*- coding: utf-8 -*-
import os, sys, time, statistics, numpy as np
import torch, cv2
from PIL import Image
BASE = r'C:\Users\olivi\Downloads\miniproject\dermaai'
TEST = r'C:\Users\olivi\.cache\kagglehub\datasets\shubhamgoel27\dermnet\versions\1\test'
sys.path.insert(0, BASE)
import importlib.util
spec = importlib.util.spec_from_file_location('derma', os.path.join(BASE, 'derma.py'))
derma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(derma)
model, transform = derma.model, derma.transform
model.eval()

folders = sorted(os.listdir(TEST))
# deterministic spread across classes
imgs = []
for c in range(23):
    fl = sorted(os.listdir(os.path.join(TEST, folders[c])))
    imgs.append(os.path.join(TEST, folders[c], fl[len(fl)//2]))

def one(p):
    pil_img = Image.open(p).convert('RGB')
    inp = transform(pil_img).unsqueeze(0); inp.requires_grad = True
    t0 = time.perf_counter()
    model.zero_grad(); derma.gradients = None; derma.activations = None
    outs = model(inp)
    t1 = time.perf_counter()
    ti = int(outs.argmax(1).item())
    outs[0, ti].backward(retain_graph=True)
    t2 = time.perf_counter()
    pooled = torch.mean(derma.gradients, dim=[0,2,3])
    act = derma.activations.clone()
    for i in range(act.shape[1]):
        act[:,i,:,:] *= pooled[i]
    heat = torch.mean(act, dim=1).squeeze().detach().numpy()
    heat = np.maximum(heat,0); heat /= heat.max()+1e-8
    img_np = cv2.imdecode(np.frombuffer(open(p,'rb').read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img_np.shape[:2]
    heat_r = cv2.resize(heat,(w,h))
    heat_c = cv2.applyColorMap(np.uint8(255*heat_r), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(img_np, 0.6, heat_c, 0.4, 0)
    t3 = time.perf_counter()
    return (t1-t0)*1e3, (t2-t1)*1e3, (t3-t2)*1e3

warm = one(imgs[0])
fwd, gc, bl = [], [], []
for p in imgs[:12]:
    a,b,c = one(p); fwd.append(a); gc.append(b); bl.append(c)
print('n=12  forward mean %.0f ms (sd %.0f) | gradcam+blend mean %.0f ms | total mean %.0f ms'
      % (statistics.mean(fwd), statistics.stdev(fwd), statistics.mean(gc)+statistics.mean(bl),
         statistics.mean(fwd)+statistics.mean(gc)+statistics.mean(bl)))
print('forward p50 %.0f total p50 %.0f' % (statistics.median(fwd), statistics.median([f+g+b for f,g,b in zip(fwd,gc,bl)])))
np.save(os.path.join(BASE, 'eval_output', 'latency_measured.npy'),
        np.array([statistics.mean(fwd), statistics.mean(gc), statistics.mean(bl)]))