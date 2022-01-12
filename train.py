import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from localization import export_gradient_maps
from model import FastFlow, save_model, save_weights
from utils import *


import neptune.new as neptune # comment this statement if you don't use neptune
import config as c
import neptuneparams as nep_params # comment this statement if you don't use neptune

# Neptune.ai set up, in order to keep track of your experiments
if c.neptune_activate:
    run = neptune.init(
        project = nep_params.project,
        api_token = nep_params.api_token,
    )  # your credentials

    run["name_dataset"] = [c.dataset_path]
    run["img_dims"] = [c.img_dims]
    run["device"] = c.device
    run["n_scales"] = c.n_scales
    run["class_name"] = [c.class_name]
    run["meta_epochs"] = c.meta_epochs
    run["sub_epochs"] = c.sub_epochs
    run["batch_size"]= c.batch_size
    run["n_coupling_blocks"] = c.n_coupling_blocks
    run["n_transforms"] = c.n_transforms
    run["n_transforms_test"] = c.n_transforms_test
    run["dropout"] =c.dropout
    run["learning_rate"] = c.lr_init
    run["subnet_conv_dim"]= c.subnet_conv_dim





class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))


def train(train_loader, test_loader):
    model = FastFlow()
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)

    score_obs_auroc = Score_Observer('AUROC')
    score_obs_aucpr = Score_Observer('AUCPR')

    for epoch in range(c.meta_epochs):

        # train some epochs
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                optimizer.zero_grad()
                inputs, labels = preprocess_batch(data)  # move to device and reshape
                # TODO inspect
                # inputs += torch.randn(*inputs.shape).cuda() * c.add_img_noise

                z, log_jac_det = model(inputs)
                loss = get_loss(z, log_jac_det)
                train_loss.append(t2np(loss))
                loss.backward()
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if c.verbose:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
            if c.neptune_activate:
                run["train/train_loss"].log(mean_train_loss)

        # evaluate
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_z = list()
        test_labels = list()
        anomaly_score = list()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                z, log_jac_det = model(inputs)
                # Why do I compute the loss also for defective images, which will have great loss values?
                loss = get_loss(z, log_jac_det)
                test_z.append(z)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))

                #I compute the values of anomaly score here in order to use less GPU memory
                z_grouped_temp = z.view(-1, c.n_transforms_test, c.n_feat)
                anomaly_score.append(t2np(torch.mean(z_grouped_temp ** 2, dim=(-2, -1))))



        test_loss_good = list()
        test_loss_defective = list()
        for i in range(len(test_labels)):
            if test_labels[i] == 0: # label value of good TODO eliminate magic numbers
                test_loss_good.append(test_loss[i])
            else:
                test_loss_defective.append(-test_loss[i])
        test_loss_good = np.mean(np.array(test_loss_good))
        test_loss_defective = np.mean(np.array(test_loss_defective))

        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f} \t test_loss_good: {:.4f} \t test_loss_defective: {:.4f}'.format(epoch, test_loss, test_loss_good, test_loss_defective))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        #anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        score_obs_auroc.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                         print_score=c.verbose or epoch == c.meta_epochs - 1)
        score_obs_aucpr.update(average_precision_score(is_anomaly, anomaly_score), epoch,
                         print_score=c.verbose or epoch == c.meta_epochs - 1)

        if c.neptune_activate:
            run["train/auroc"].log(score_obs_auroc.last)
            run["train/aucpr"].log(score_obs_aucpr.last)
            run["train/test_loss"].log(test_loss)
            run["train/test_loss_good"].log(test_loss_good)
            run["train/test_loss_defective"].log(test_loss_defective)

        



    if c.grad_map_viz:
        export_gradient_maps(model, test_loader, optimizer, -1)

    if c.save_model:
        model.to('cpu')
        save_model(model.state_dict(), c.modelname)
        save_weights(model, c.modelname)
    return model
