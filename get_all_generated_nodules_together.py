import os
import numpy as np
from shutil import move
from tqdm import tqdm

path_data = '/data/OMM/project_results/Jun_17_2020_cell_aut_grown_nodules/batch_center/'
path_dest = '/data/OMM/project_results/Jun_17_2020_cell_aut_grown_nodules/batches_together/'

folders = os.listdir(path_data)

print(folders)

for i in folders:
    if i != 'ca_nodules_generated_center_G':
        print(f'{i}')
        for sub in range(10):
            subset = f'subset{sub}'
            ff = os.listdir(f'{path_data}{i}/patched/{subset}/')
            for f in tqdm(ff, total=len(ff)):
                move(f'{path_data}{i}/patched/{subset}/{f}',f'{path_dest}{subset}/{f}')
