import os

import torch
import grb.utils as utils
from grb.dataset import Dataset
from gcndp import GCN
#from grb.model.torch.gcn import GCN
from grb.utils.normalize import GCNAdjNorm
from grb.attack.injection import FGSM
from adv_trainer import AdvTrainer
from grb.attack.injection.tdgia import TDGIA

def main():


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



    model_name = "gcn"
    model = GCN(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=64,
                n_layers=3,
                adj_norm_func=GCNAdjNorm,
                layer_norm=True,
                residual=False,
                dropout=0.5)
    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)



    #device = 'cuda:0'
    device = "cpu"
    attack = FGSM(epsilon=0.01,
                  n_epoch=10,
                  n_inject_max=10,
                  n_edge_max=20,
                  feat_lim_min=features.min(),
                  feat_lim_max=features.max(),
                  early_stop=False,
                  device=device,
                  verbose=False)



    save_dir = "./saved_models/{}/{}_at".format(dataset_name, model_name)
    save_name = "model.pt"
    device = "cpu"
    feat_norm = None
    train_mode = "inductive"  # "transductive"


    trainer = AdvTrainer(dataset=dataset,
                         attack=attack,
                         optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                         loss=torch.nn.functional.cross_entropy,
                         lr_scheduler=False,
                         early_stop=True,
                         early_stop_patience=500,
                         device=device)

    trainer.train(model=model,
                  n_epoch=2000,
                  eval_every=1,
                  save_after=0,
                  save_dir=save_dir,
                  save_name=save_name,
                  train_mode=train_mode,
                  verbose=False)


    # by trainer
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test score: {:.4f}".format(test_score))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
