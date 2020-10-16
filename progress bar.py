# from tqdm import tqdm
# import time
# for i in tqdm(range(10000)):
#     time.sleep(0.01)


# -*- coding: utf-8 -*-
from tqdm import tqdm
import time
import sys

# for i in tqdm(range(100)):
#     time.sleep(.1)
for i in range(10):
    a = i
    b = 9-i
    # sys.stdout.write('\r>>convert image %d/%d'%(i,b))

    sys.stdout.write('\r|%s%s|%d%%' % (a * '▇', b * ' ', (i + 1)*10))
    sys.stdout.flush()
    time.sleep(.1)