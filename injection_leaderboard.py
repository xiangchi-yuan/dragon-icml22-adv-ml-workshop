import os
import torch
import grb.utils as utils
from grb.dataset import Dataset
from grb.utils.normalize import GCNAdjNorm
from grb.attack.injection import FGSM
from grb.attack.injection import TDGIA

def main():
    dataset_name = 'grb-cora'
    dataset = Dataset(name=dataset_name,
                      data_dir="../data/",
                      mode='full',
                      feat_norm='arctan')
    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    test_mask = dataset.test_mask
    device = "cuda:0"

    attack = FGSM(epsilon=0.01,
                  n_epoch=1000,
                  n_inject_max=60,
                  n_edge_max=20,
                  feat_lim_min=-1,
                  feat_lim_max=1,
                  device=device)

    # model sur ,always gcn
    save_dir = "./saved_models/{}/gcn".format(dataset_name)
    save_name = "model.pt"
    model_sur = torch.load(os.path.join(save_dir, save_name))
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

    model_name = "gatdp"
    target_dir = "./saved_models/{}/{}".format(dataset_name, model_name)
    target_name = "model.pt"
    device = "cuda:0"
    model = torch.load(os.path.join(target_dir, target_name))
    model = model.to(device)
    model.eval()
    test_score = utils.evaluate(model,
                                features=features_attacked,
                                adj=adj_attack,
                                labels=dataset.labels,
                                adj_norm_func=model.adj_norm_func,
                                mask=dataset.test_mask,
                                device=device)
    print("Test score after attack for target model: {:.4f}.".format(test_score))



if __name__ == '__main__':
    main()
