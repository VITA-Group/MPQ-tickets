import os
from collections import OrderedDict
import re

result_files={}
for x in os.listdir('.'):
    if x.startswith('iter') and x.endswith('out'):
        idx=int(x.split('_')[-1].split('.')[0])
        result_files[idx]=x
result_files=OrderedDict(sorted(result_files.items()))

best_precision=[]
avg_bits=[]

for i in result_files:
    with open(result_files[i],'r') as f:
        lis=f.readlines()
        for line in lis[::-1]:
            if '(best_acc' in line:
                best_precision.append(float(re.split(',| ', line)[4]))
                break
        for line in lis:
            if 'avg_bit' in line:
                avg_bits.append(float(re.split(',| ',line)[4]))
                break

print('best_precision:')
for x in best_precision:
    print(x)

print('avg_bits:')
for x in avg_bits:
    print(x)