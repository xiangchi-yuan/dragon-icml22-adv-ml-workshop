import os
import torch
import grb.utils as utils
from grb.dataset import Dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from numba import cuda
cuda.select_device(0)
cuda.close()
cuda.select_device(0)

dataset_name = "grb-cora"
dataset = Dataset(name=dataset_name,
                  data_dir="../data/",
                  mode="full",
                  feat_norm="arctan")

# from gcndp import GCN
# model_name = "gcndp"
# model = GCN(in_features=dataset.num_features,
#             out_features=dataset.num_classes,
#             hidden_features=256,
#             n_layers=4,
#             layer_norm=False,
#             dropout=0.5)

from grb.model.dgl import GAT
from gatdp import GAT
model_name = "gatdp1"
model = GAT(in_features=dataset.num_features,
            out_features=dataset.num_classes,
            hidden_features=64,
            n_layers=3,
            n_heads=6,
            layer_norm=False,
            dropout=0.5)

# # from gatdp import GAT
# from gatdp_moe import GAT_moe
# model_name = "gatdp_moe"
# model = GAT_moe(in_features=dataset.num_features,  # GAT(in_features=dataset.num_features,
#                 out_features=dataset.num_classes,
#                 hidden_features=64,
#                 n_layers=3,
#                 n_heads=4,
#                 adj_norm_func=None,
#                 layer_norm=False,
#                 residual=False,
#                 feat_dropout=0.75,
#                 attn_dropout=0.75,
#                 dropout=0.6)


# from grb.defense import RobustGCN
# #from rgcndp import RobustGCN
# model_name = "rgcn"
# model = RobustGCN(in_features=dataset.num_features,
#                   out_features=dataset.num_classes,
#                   hidden_features=128,
#                   n_layers=4,
#                   dropout=0.5)

# from grb.defense import GATGuard
# model_name = "gatguard"
# model = GATGuard(in_features=dataset.num_features,
#                  out_features=dataset.num_features,
#                  hidden_features=128,
#                  n_heads=4,
#                  n_layers=3,
#                 dropout=0.5)

# from grb.defense import GCNGuard
# model_name = "gcnguard"
# model = GCNGuard(in_features=dataset.num_features,
#                  out_features=dataset.num_classes,
#                  hidden_features=128,
#                  n_layers=3,
#                  dropout=0.5)

# from grb.defense.gcnsvd import GCNSVD
# model_name = "gcnsvd"
# model = GCNSVD(in_features=dataset.num_features,
#                out_features=dataset.num_features,
#                hidden_features=128,
#                n_layers=3,
#                dropout=0.5)

# from grb.model.torch import SGCN
# model_name = "sgcn_ln"
# model = SGCN(in_features=dataset.num_features,
#              out_features=dataset.num_classes,
#              hidden_features=256,
#              n_layers=4,
#              k=4,
#              layer_norm=True if "ln" in model_name else False,
#              dropout=0.5)

# from grb.model.torch import GIN
# model_name = "gin_ln"
# model = GIN(in_features=dataset.num_features,
#             out_features=dataset.num_features,
#             hidden_features=256,
#             n_layers=2,
#             layer_norm=True if "ln" in model_name else False,
#             dropout=0.6)

print("Number of parameters: {}.".format(utils.get_num_params(model)))
print(model)

save_dir = "./saved_models/{}/{}".format(dataset_name, model_name)
save_name = "model.pt"
device = "cuda:0"
feat_norm = None
train_mode = "inductive"  # "transductive"
#from grb.trainer.trainer import Trainer
from trainer import Trainer
trainer = Trainer(dataset=dataset,
                  optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                  loss=torch.nn.functional.cross_entropy,
                  lr_scheduler=False,
                  early_stop=True,
                  early_stop_patience=500,
                  feat_norm=feat_norm,
                  device=device)
trainer.train(model=model,
              n_epoch=1000,
              eval_every=1,
              save_after=0,
              save_dir=save_dir,
              save_name=save_name,
              train_mode=train_mode,
              verbose=False,
              )

model = torch.load(os.path.join(save_dir, save_name))
model = model.to(device)
model.eval()

# by trainer
pred = trainer.inference(model)
# by utils
pred = utils.inference(model,
                       features=dataset.features,
                       feat_norm=feat_norm,
                       adj=dataset.adj,
                       adj_norm_func=model.adj_norm_func,
                       device=device)
# by trainer
test_score = trainer.evaluate(model, dataset.test_mask)
print("Test score: {:.4f}".format(test_score))

# by utils
test_score = utils.evaluate(model,
                            features=dataset.features,
                            adj=dataset.adj,
                            labels=dataset.labels,
                            feat_norm=feat_norm,
                            adj_norm_func=model.adj_norm_func,
                            mask=dataset.test_mask,
                            device=device)
print("Test score: {:.4f}".format(test_score))


