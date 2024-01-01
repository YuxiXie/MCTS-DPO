import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor as Pool


def main(argv):
    gpu_id, fname = argv
    completed_process = subprocess.run(["bash", fname], 
                                       capture_output=True, 
                                       text=True)
    fname = fname.split('/')[-1].split('.')[0]
    # with open(f'{LOGDIR}/{fname}_stdout.log', 'w', encoding='utf-8') as f:
    #     f.write(completed_process.stdout)
    # with open(f'{LOGDIR}/{fname}_stderr.log', 'w', encoding='utf-8') as f:
    #     f.write(completed_process.stderr)
    
    return 'done.'

if __name__ == '__main__':
    fnames = sys.argv[1:]
    n_jobs = len(fnames)
    
    with Pool(n_jobs) as pool:
        rst = pool.map(main, zip(range(n_jobs), fnames))
    