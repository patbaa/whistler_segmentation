import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from skimage import measure
from mmdet.apis import inference_detector, init_detector

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--meta_csv', default='', type=Path)
parser.add_argument('--model_config', default='config.py', type=Path)
parser.add_argument('--model_weights', default='published_weights-70db75db.pth', type=Path)
parser.add_argument('--save_folder', required=True, type=Path)
parser.add_argument('--GPU_ID', default=0, type=int)

args = parser.parse_args()

################################################################################

config = str(args.model_config) 
checkpoint = str(args.model_weights) 
model = init_detector(config, checkpoint, device=f'cuda:{args.GPU_ID}')

model.test_cfg.rcnn.mask_thr_binary = 0.4
model.test_cfg.rcnn.score_thr = 0.5
model.test_cfg.rcnn.nms.iou_threshold = 0.99

model.test_cfg.rcnn.max_per_img = 250
model.test_cfg.rpn.nms.iou_threshold = 0.99

################################################################################

# make prediction for a vr2
def get_ID_predmask(files, model=model):
    Ns = np.array([int(i.split('_')[-1].replace('.png', '')) for i in files])
    files = files[np.argsort(Ns)]
    
    results = {}
    for idx, f in enumerate(files):
        tmpres = inference_detector(model, f)
        results[idx+1] = np.array(tmpres[1][0])
        
    all_res = np.zeros((1024, 1024 + 500*(len(files)-1)))
       
    for i in range(1, len(files) + 1):                   
        if len(results[i]) > 0:
            all_res[:,(i-1)*500:((i-1)*500 + results[i].sum(0).shape[1])] += results[i].sum(0)
    
    return (all_res != 0)[::-1, :] #flip frequency
                       
                       
def make_blobs_from_preds(predictions):
    blobs = measure.label(predictions)
    for i in pd.unique(blobs.ravel()):
        # if a blob has less than 200 pixel, throw it
        if (blobs == i).sum() < 200:
            blobs[blobs == i] = 0
        else:
            freqs = (blobs == i).sum(1)
            minfreq = np.array(range(1024))[freqs > 0].min()
            maxfreq = np.array(range(1024))[freqs > 0].max()
            # if a blob covers less than 100 frequency bins, throw it
            if (maxfreq-minfreq) < 100:
                blobs[blobs == i] = 0
                continue

            times = (blobs == i).sum(0)
            mintime = np.array(range(len(times)))[times > 0].min()
            maxtime = np.array(range(len(times)))[times > 0].max()
            # if a blob is shorter than 100ms, throw it
            if (maxtime-mintime) < 100:
                blobs[blobs == i] = 0
    return blobs

def narrow_trace(preds):
    times = []
    freqs = np.arange(1024)
    
    for f in freqs:
        if preds[int(f), :].sum() == 0:
            t = 0
        else:
            # narrowed trace = middle of the predicted, wider trace
            t = np.arange(preds.shape[1])[preds[f, :]].mean()
        times.append(t)
    times = np.array(times)

        
    freqs = freqs[times != 0]
    times = times[times != 0]
    
    return {'f':freqs, 't':times}  

def save_trace(trace, fname):
    traceDF = pd.DataFrame(trace)
    traceDF['f'] = (traceDF['f']+1)*19.5312
    traceDF['t'] = traceDF['t']/1000
    traceDF[['f', 't']].sort_values('f').to_csv(fname, sep=' ', header=None, index=None)
    
################################################################################                       
                               
df = pd.read_csv(args.meta_csv)
df['ID'] = ['_'.join(i.split('/')[-1].split('_')[:-1]) for i in df.sqrtis_png.values]
df['N'] = [int(i.split('_')[-1].split('.')[0]) for i in df.sqrtis_png.values]
df = df.sort_values('ID')                               
IDs = pd.unique(df.ID.values)
                               
for ID in IDs:
    tmp = df[df.ID == ID].sort_values('N')                                    
    predmask = get_ID_predmask(tmp.sqrtis_png.values)
    blob_predmask = make_blobs_from_preds(predmask)
                               
    trace_cnt = 1                           
    for i in pd.unique(blob_predmask.ravel()):
        if i == 0:
            continue
        trace = narrow_trace(blob_predmask == i)
        save_trace(trace, f'{args.save_folder.as_posix()}/{ID}.ft{trace_cnt}')                       
        trace_cnt += 1                       
                               