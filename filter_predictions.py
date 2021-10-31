import shutil
import argparse
import numpy as np
import pandas as pd
import matlab.engine
from pathlib import Path
from datetime import datetime

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--unfiltered_folder', required=True, type=Path)
parser.add_argument('--unfiltered_folder_reg', required=True, type=str)
parser.add_argument('--inversion_file', required=True, type=Path)
parser.add_argument('--matlab_scripts', 
                    default='/home/pataki/wh_paper/matlab_src/', type=Path)

args = parser.parse_args()

################################################################################

eng = matlab.engine.start_matlab()
eng.addpath(args.matlab_scripts.as_posix(), nargout=0)

def invert_trace(fname):
    res = eng.sfred_ft_Lsorfun_halo_wh_nn_ozsogin_fof2_lepcso_lut_v1(fname)    
    df = pd.DataFrame(np.array(res)[0]).T
    df.columns = ['L', 'neq', 'sig_L', 'sig_neq', 'resn', 'max(abs(res))', 
                  'foF2', 'fheq', 'fn', 'tn', 'tsf_ftinv']
    df['file'] = fname
    df['ID'] = fname.split('/')[-1].split('.ft')[0]
    return df

def trace_max_freq(fname):
    df = pd.read_csv(fname, names = ['f', 't'], delim_whitespace=True)
    return df['f'].max()

################################################################################

filtered_inversion = pd.DataFrame()

unfiltered_traces = list(args.unfiltered_folder.glob(args.unfiltered_folder_reg))
print(len(unfiltered_traces), ' traces are gonna be processed.')
for idx, trace_fn in enumerate(unfiltered_traces):
    if (idx%1000) == 0:
        now = datetime.now()
        print(idx, now.strftime("%d/%m/%Y %H:%M:%S"))
    try:
        inversion = invert_trace(str(trace_fn.as_posix()))
    except matlab.engine.MatlabExecutionError:
        continue
    max_freq = trace_max_freq(str(trace_fn.as_posix()))
    
    if inversion.resn.values[0] >= 0.005:
        inversion['high_quality'] = False
        filtered_inversion = filtered_inversion.append(inversion)
        continue
    
    if inversion.fn.values[0]*0.6 > max_freq:
        inversion['high_quality'] = False
        filtered_inversion = filtered_inversion.append(inversion)
        continue
        
    inversion['high_quality'] = True
    filtered_inversion = filtered_inversion.append(inversion)
    
    
filtered_inversion.to_csv(args.inversion_file, index=False)
