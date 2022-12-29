import os
import torch
import grb.utils as utils

from grb.dataset import Dataset

dataset_name = 'grb-cora'
dataset = Dataset(name=dataset_name,
                  data_dir="../../data/",
                  mode='full',
                  feat_norm='arctan')

adj = dataset.adj
features = dataset.features
labels = dataset.labels
num_features = dataset.num_features
num_classes = dataset.num_classes
test_mask = dataset.test_mask

from grb.model.torch import GCN
from grb.utils.normalize import GCNAdjNorm

model_name = "gcn"
model_sur = GCN(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=64,
                n_layers=3,
                adj_norm_func=GCNAdjNorm,
                layer_norm=False,
                residual=False,
                dropout=0.6)
print(model_sur)

save_dir = "./saved_models/{}/{}".format(dataset_name, model_name)
save_name = "model_sur.pt"
device = "cuda:0"
feat_norm = None
train_mode = "inductive"  # "transductive"

from grb.trainer.trainer import Trainer

trainer = Trainer(dataset=dataset,
                  optimizer=torch.optim.Adam(model_sur.parameters(), lr=0.01),
                  loss=torch.nn.functional.cross_entropy,
                  lr_scheduler=False,
                  early_stop=True,
                  early_stop_patience=500,
                  feat_norm=feat_norm,
                  device=device)

trainer.train(model=model_sur,
              n_epoch=5000,
              eval_every=1,
              save_after=0,
              save_dir=save_dir,
              save_name=save_name,
              train_mode=train_mode,
              verbose=False)

# by trainer
test_score = trainer.evaluate(model_sur, dataset.test_mask)
print("Test score of surrogate model: {:.4f}".format(test_score))


# GATLN77 GAT65 GCNLN78 GCN42
# from grb.attack.injection import TDGIA
# attack = TDGIA(lr=0.01,
#                n_epoch=1000,
#                n_inject_max=60,
#                n_edge_max=20,
#                feat_lim_min=-1,
#                feat_lim_max=1,
#                device=device)

# 59 34 GCNLN63.8 GCN35.2
# from grb.attack.injection import SPEIT
# attack = SPEIT(lr=0.01,
#                n_epoch=1000,
#                n_inject_max=60,
#                n_edge_max=20,
#                feat_lim_min=-1,
#                feat_lim_max=1,
#                device=device)

# 81 GCNLN81.6 80.9 GCN81.59
# from grb.attack.injection import RAND
# attack = RAND(n_inject_max=60,
#               n_edge_max=20,
#               feat_lim_min=-1,
#               feat_lim_max=1,
#               device=device)

# 68 34 GCNLN64 GCN35.6
from grb.attack.injection import FGSM
attack = FGSM(epsilon=0.01,
              n_epoch=1000,
              n_inject_max=60,
              n_edge_max=20,
              feat_lim_min=-1,
              feat_lim_max=1,
              device=device)

# 60 27 GCNLN67 GCN34.8
# from grb.attack.injection import PGD
# attack = PGD(epsilon=0.01,
#              n_epoch=1000,
#              n_inject_max=60,
#              n_edge_max=20,
#              feat_lim_min=-1,
#              feat_lim_max=1,
#              device=device)

adj_attack, features_attack = attack.attack(model=model_sur,
                                            adj=adj,
                                            features=features,
                                            target_mask=test_mask,
                                            adj_norm_func=model_sur.adj_norm_func)

features_attacked = torch.cat([features.to(device), features_attack])
test_score = utils.evaluate(model_sur,
                            features=features_attacked,
                            adj=adj_attack,
                            labels=dataset.labels,
                            adj_norm_func=model_sur.adj_norm_func,
                            mask=dataset.test_mask,
                            device=device)
print("Test score after attack for surrogate model: {:.4f}.".format(test_score))
#
# model_name = "gcn"
# model = GCN(in_features=dataset.num_features,
#                 out_features=dataset.num_classes,
#                 hidden_features=64,
#                 n_layers=3,
#                 adj_norm_func=GCNAdjNorm,
#                 layer_norm=False,
#                 residual=False,
#                 dropout=0.5)

from grb.model.dgl import GAT
#from gatdp import GAT
model_name = "gat"
model = GAT(in_features=dataset.num_features,
            out_features=dataset.num_classes,
            hidden_features=64,
            n_layers=3,
            n_heads=6,
            adj_norm_func=None,
            layer_norm=False,
            residual=False,
            feat_dropout=0.6,
            attn_dropout=0.6,
            dropout=0.5)
print(model)

save_dir = "./saved_models/{}/{}".format(dataset_name, model_name)
save_name = "model.pt"
device = "cuda:0"
feat_norm = None
train_mode = "inductive"  # "transductive"


trainer = Trainer(dataset=dataset,
                  optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                  loss=torch.nn.functional.cross_entropy,
                  lr_scheduler=False,
                  early_stop=True,
                  early_stop_patience=500,
                  feat_norm=feat_norm,
                  device=device)

trainer.train(model=model,
              n_epoch=5000,
              eval_every=1,
              save_after=0,
              save_dir=save_dir,
              save_name=save_name,
              train_mode=train_mode,
              verbose=False)

test_score = utils.evaluate(model,
                            features=features_attacked,
                            adj=adj_attack,
                            labels=dataset.labels,
                            adj_norm_func=model.adj_norm_func,
                            mask=dataset.test_mask,
                            device=device)
print("Test score after attack for target model: {:.4f}.".format(test_score))