# grbdp



this is training without AT.

run training.py

you can choose model with/without DP here:

#from grb.model.dgl import GAT

from gatdp import GAT

then run injection_leaderboard.py


this is training with AT.

run training.py

note change the model with raw GCN, first train GCN to make surrogate model for injection.

run adv_training.py

you can choose model with/without DP here:

#from grb.model.dgl import GAT

from gatdp import GAT

then run injection_leaderboard.py, note change line53 to "final_model.pt"
