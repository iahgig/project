import matplotlib.pyplot as plt
import numpy as np
iter_num=list(range(1,7))

prec=[0.4,0.9958,0.9958,0.9958,0.9958,0.9958]
rec=[0.0176,0.6779,0.7326,0.7482,0.7542,0.7542]
prec=np.array(prec)
rec=np.array(rec)
f1=2*prec*rec/(prec+rec)
print(f1)
plt.plot(iter_num,prec,label='Prec')
plt.plot(iter_num,rec,label='Rec')
plt.plot(iter_num,f1,label='F1')
plt.xlabel('Number of iterations')
plt.xticks(iter_num)
plt.title("Swat")
plt.legend()
#plt.show()
print(2*0.9953*0.9802/(0.9953+0.9802))