import argparse
import os
import torch
import grb.utils as utils

from grb.dataset import Dataset
import gc

gc.collect()

torch.cuda.empty_cache()

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset_name', type=str, default='grb-reddit')
argparser.add_argument('--train_mode', type=str, default='inductive')
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--n_inject_max', type=int, default=500)
argparser.add_argument('--n_edge_max', type=int, default=200)
argparser.add_argument('--early_stop_patience', type=int, default=500)
argparser.add_argument('--hidden_features', type=int, default=32)
argparser.add_argument('--n_layers', type=int, default=3)
argparser.add_argument('--layer_norm', type=bool, default=False)
args = argparser.parse_args()

dataset_name = 'grb-flickr'
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
                hidden_features=256,
                n_layers=3,
                adj_norm_func=GCNAdjNorm,
                layer_norm=True,
                residual=False,
                dropout=0.5)
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
              n_epoch=1000,
              eval_every=1,
              save_after=0,
              save_dir=save_dir,
              save_name=save_name,
              train_mode=train_mode,
              verbose=False)
# by trainer
test_score = trainer.evaluate(model_sur, dataset.test_mask)
print("Test score of surrogate model: {:.4f}".format(test_score))


from grb.attack.injection import FGSM

attack = FGSM(epsilon=0.01,
              n_epoch=1000,
              n_inject_max=1000,
              n_edge_max=100,
              feat_lim_min=-1,
              feat_lim_max=1,
              device=device)
#
# from grb.attack.injection import TDGIA
# attack = TDGIA(lr=0.01,
#                n_epoch=1000,
#                n_inject_max=300,
#                n_edge_max=20,
#                feat_lim_min=-1,
#                feat_lim_max=1,
#                device=device)

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


model_at_name_list = ["gatdp0.1","gatdp","gatdp1","gat", "rgcn"]
model_name_list = ["gat","gatdp0.1", "gatdp","gatdp1", "rgcn", "sgcn_ln", "gin_ln"]

model = torch.load("./saved_models/{}/GAT_at/final_model.pt".format(dataset_name))
model = model.to(device)
model.eval()
test_score = utils.evaluate(model,
                            features=features_attacked,
                            adj=adj_attack,
                            labels=dataset.labels,
                            adj_norm_func=model.adj_norm_func,
                            mask=dataset.test_mask,
                            device=device)
print("Test score after attack for target model: {:.4f}.".format(test_score) + "GAT_dp_moe_at")

for name in model_at_name_list:
    save_dir = "./saved_models/{}/{}_at".format(dataset_name, name)
    save_name = "final_model.pt"
    device = "cuda:0"
    model = torch.load(os.path.join(save_dir, save_name))
    model = model.to(device)
    model.eval()
    test_score = utils.evaluate(model,
                                features=features_attacked,
                                adj=adj_attack,
                                labels=dataset.labels,
                                adj_norm_func=model.adj_norm_func,
                                mask=dataset.test_mask,
                                device=device)
    print("Test score after attack for target model: {:.4f}.".format(test_score) + name)

# for name in model_name_list:
#     save_dir = "../saved_models/{}/{}".format(dataset_name, name)
#     save_name = "model.pt"
#     device = "cuda:0"
#     model = torch.load(os.path.join(save_dir, save_name))
#     model = model.to(device)
#     model.eval()
#     test_score = utils.evaluate(model,
#                                 features=features_attacked,
#                                 adj=adj_attack,
#                                 labels=dataset.labels,
#                                 adj_norm_func=model.adj_norm_func,
#                                 mask=dataset.test_mask,
#                                 device=device)
#     print("Test score after attack for target model: {:.4f}.".format(test_score) + name)
