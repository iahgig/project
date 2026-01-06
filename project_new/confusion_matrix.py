import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize

sensor_emb=pd.read_excel('sensor_emb.xlsx')
'''
t=torch.tensor([[1,2,3],[4,5,6]],dtype=float)
t=t.cuda()
a=t.detach().cpu().numpy()
a=pd.DataFrame(a)
a.to_excel('a.xlsx')
'''

sensor_emb=np.array(sensor_emb)
sensor_emb_norm=normalize(sensor_emb,norm='l2',axis=1)

similarity_score_matrix=np.matmul(sensor_emb_norm,sensor_emb_norm.transpose(1,0))
data=pd.DataFrame(similarity_score_matrix)
data.to_excel('matrix.xlsx')
