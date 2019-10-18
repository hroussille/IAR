import numpy as np

Q = np.load('log/1571350514.9283292_Qlearning_values.npy', allow_pickle=True)
Q = Q.item()

print('00002' , Q['00002'])
print('00072' , Q['00072'])

print('00000' , Q['00000'])
print('00070' , Q['00070'])

print('11101' , Q['11101'])
print('11171' , Q['11171'])



