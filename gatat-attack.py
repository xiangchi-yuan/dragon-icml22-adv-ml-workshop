import os

import torch
import grb.utils as utils
from grb.dataset import Dataset
from gatdp import GAT
#from grb.model.dgl import GAT
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



    model_name = "gcn_at_attack"
    model_name = "gat"
    model_sur = GAT(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=64,
                n_layers=3,
                n_heads=4,
                adj_norm_func=None,
                layer_norm=False,
                residual=False,
                feat_dropout=0.6,
                attn_dropout=0.6,
                dropout=0.5)
    print("Number of parameters: {}.".format(utils.get_num_params(model_sur)))
    print(model_sur)



    #device = 'cuda:0'
    device = "cpu"
    attack1 = FGSM(epsilon=0.01,
                  n_epoch=10,
                  n_inject_max=10,
                  n_edge_max=20,
                  feat_lim_min=features.min(),
                  feat_lim_max=features.max(),
                  early_stop=False,
                  device=device,
                  verbose=False)



    save_dir = "./saved_models/{}/{}".format(dataset_name, model_name)
    save_name = "model_sur.pt"
    device = "cpu"
    feat_norm = None
    train_mode = "inductive"  # "transductive"


    trainer = AdvTrainer(dataset=dataset,
                         attack=attack1,
                         optimizer=torch.optim.Adam(model_sur.parameters(), lr=0.01),
                         loss=torch.nn.functional.cross_entropy,
                         lr_scheduler=False,
                         early_stop=True,
                         early_stop_patience=500,
                         device=device)

    trainer.train(model=model_sur,
                  n_epoch=2000,
                  eval_every=1,
                  save_after=0,
                  save_dir=save_dir,
                  save_name=save_name,
                  train_mode=train_mode,
                  verbose=False)


    # by trainer
    test_score = trainer.evaluate(model_sur, dataset.test_mask)
    print("Test score: {:.4f}".format(test_score))

    # Attack configuration
    attack2 = TDGIA(lr=0.01,
                   n_epoch=1000,
                   n_inject_max=100,
                   n_edge_max=200,
                   feat_lim_min=-1,
                   feat_lim_max=1,
                   )
    # Apply attack
    adj_attack, features_attack = attack2.attack(model=model_sur,
                                                adj=adj,
                                                features=features,
                                                target_mask=test_mask
                                                )

    features_attacked = torch.cat([features.to(device), features_attack])
    test_score = utils.evaluate(model_sur,
                                features=features_attacked,
                                adj=adj_attack,
                                labels=dataset.labels,
                                adj_norm_func=model_sur.adj_norm_func,
                                mask=dataset.test_mask
                                )
    print("Test score after attack for surrogate model: {:.4f}.".format(test_score))

    # Transfer to target model
    model_name = "gcn"

    save_name = "model.pt"
    device = "cpu"

    model = torch.load('D:\ALab\grb\saved_models\grb-cora\gcn_at_attack\model_sur.pt')
    model = model.to(device)
    model.eval()

    test_score = utils.evaluate(model,
                                features=features_attacked,
                                adj=adj_attack,
                                labels=dataset.labels,
                                adj_norm_func=model.adj_norm_func,
                                mask=dataset.test_mask)
    print("Test score after attack for target model: {:.4f}.".format(test_score))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()