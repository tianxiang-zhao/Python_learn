import numpy as np; import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.DataFrame(np.random.randint(0, 2, (2, 5)))
df1 = pd.DataFrame(np.random.randint(0, 2, (2, 5)))
print(df,"\n",df1)

result=cosine_similarity(df,df1)
print(result)

result=sorted(result[0],reverse=True)
print(result)
