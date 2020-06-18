#!/usr/bin/env python
# coding: utf-8
# Report bug to tiwarilokender@gmail.com

import torch
import torch.nn.functional as F
import scipy.io as sio
import torchvision.transforms as transforms
import vgg
import glob
import numpy as np
from foolbox.models import PyTorchModel
from foolbox.v1 import Adversarial
from foolbox.v1.attacks import *
from foolbox.criteria import *
from foolbox.distances import *
import torchvision.models as models
import foolbox
import copy



class REGroup(torch.nn.Module):
    def __init__(self, eval_model, gc_path, device='cpu'):
        super(REGroup, self).__init__()
        self.model = eval_model
        self.device = device
        self.gc_path = gc_path
        self.gen_classifiers_pos={} 
        self.gen_classifiers_neg={}
        self.prob_scores_pos = np.zeros((19,10),dtype=np.float)
        self.prob_scores_neg = np.zeros((19,10),dtype=np.float)
        self.get_generative_classifiers()

    def get_generative_classifiers(self):
        layer_names = sio.loadmat('layers_name_vgg19.mat')['layers_name_vgg19']
        for lid in range(19):
            layer_id      = layer_names[lid].split(' ')[0]
            gen_classifiers=sio.loadmat(self.gc_path+'vgg19_'+layer_id+'_summarize.mat')
            self.gen_classifiers_pos[str(lid+1)]=gen_classifiers['class_1000_pos']
            self.gen_classifiers_neg[str(lid+1)]=gen_classifiers['class_1000_neg']
    
    def get_robust_predictions(self,img):
        self.VGG_Layer_hook_and_KL_score(img)
        rank_pref_pos=np.argsort(-self.prob_scores_pos,axis=1)
        orig_idxs_pos = 10-np.argsort(rank_pref_pos,axis=1)

        rank_pref_neg=np.argsort(-self.prob_scores_neg,axis=1)
        orig_idxs_neg = 10-np.argsort(rank_pref_neg,axis=1)

        borda_count_pos = np.sum(orig_idxs_pos,axis=0)  # borda count All layers pos
        borda_count_neg = np.sum(orig_idxs_neg,axis=0)  # borda count All layers neg
        pos_neg_borda_count = borda_count_pos+borda_count_neg  # borda count All layers (pos+neg)
        borda_count_ranked_pred = np.argsort(-pos_neg_borda_count)  # Ranking

        return borda_count_ranked_pred

    def get_foolbox_pgd_linf_adversary(self,image,gt_label,epsilon=2.0):
        print(image.shape)
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        fmodel = foolbox.models.PyTorchModel(
            self.model, bounds=(0, 1), num_classes=10, preprocessing=(mean, std))

        attack = RandomStartProjectedGradientDescentAttack()
        criterion = Misclassification()
        pert=epsilon
        attack_thresh = float(pert)/256.0         # attack threshold or epsilon of adv perturbation in normalized units
        adv_instance = Adversarial(fmodel,criterion,image,gt_label, distance=Linfinity, threshold=None, verbose=False)


        attack(adv_instance,binary_search=True,epsilon=attack_thresh)
        dist = adv_instance.distance
        dist = float(str(dist).split('=')[1])
        eff_adv_dist = dist*256
        # print('Adv Dist',dist*256)

        adv_tsor = torch.from_numpy(adv_instance.perturbed).to(self.device)
        adv_tsor = normalize(adv_tsor)

        return adv_tsor, eff_adv_dist



    def VGG_Layer_hook_and_KL_score(self,img):
        lay_out = {}
        feature_layers    = sio.loadmat('vgg_modules_cifar10.mat')['vgg19_feature_modules'][0]
        classifier_layers = sio.loadmat('vgg_modules_cifar10.mat')['vgg19_classifier_modules'][0]
        def get_layer_out_conv(m, i, o):
            lay_out['layer_'+str(fid+1)+'_conv'] = o.data[0,:,:,:].detach().clone()#.cpu()


        for fid in range(len(feature_layers)):
            h =  self.model._modules['features']._modules['module']._modules[str(feature_layers[fid])].register_forward_hook(get_layer_out_conv)
            with torch.no_grad():
                m_out = self.model(img)
            h.remove()


        def get_layer_out_fc(m, i, o):
            lay_out['layer_'+str(len(feature_layers)+cid+1)+'_fc'] = o.data[0,:].detach().clone()#.cpu()
        
        for cid in range(len(classifier_layers)):
            h =  self.model._modules['classifier']._modules[str(classifier_layers[cid])].register_forward_hook(get_layer_out_fc)
            with torch.no_grad():
                m_out = self.model(img)
            h.remove()


        for l_key  in lay_out.keys():
            if l_key.split('_')[-1] == 'conv':
                lid_num = l_key.split('_')[1]
                tmp_layer = lay_out[l_key].view(lay_out[l_key].shape[0],-1).to(self.device)
                pos_ind = tmp_layer>0
                neg_ind = tmp_layer<0

                pos_accum_conv = torch.sum(torch.mul(pos_ind.to(self.device).float(),tmp_layer),dim=1).cpu().detach().numpy()
                neg_accum_conv = torch.sum(torch.mul(neg_ind.to(self.device).float(),tmp_layer),dim=1).cpu().detach().numpy()
                pos_accum_conv   = pos_accum_conv     + 1e-20
                neg_accum_conv   = neg_accum_conv     + 1e-20

                
                pos_accum_conv   =  np.abs(pos_accum_conv)/np.sum(np.abs(pos_accum_conv))
                neg_accum_conv   =  np.abs(neg_accum_conv)/np.sum(np.abs(neg_accum_conv))
                
                kl_corr_pos = np.sum(np.log(self.gen_classifiers_pos[lid_num]/pos_accum_conv)*self.gen_classifiers_pos[lid_num],axis=1)
                kl_corr_neg = np.sum(np.log(self.gen_classifiers_neg[lid_num]/neg_accum_conv)*self.gen_classifiers_neg[lid_num],axis=1)

                kl_corr_pos = np.exp(-kl_corr_pos)
                kl_corr_neg = np.exp(-kl_corr_neg)

                self.prob_scores_pos[int(lid_num)-1] = kl_corr_pos/np.sum(kl_corr_pos)
                self.prob_scores_neg[int(lid_num)-1] = kl_corr_neg/np.sum(kl_corr_neg)



            else:
                lid_num = l_key.split('_')[1]
                fc_output = lay_out[l_key].cpu().detach().numpy()
                fc_two_np = fc_output.astype('float64')

                sam_pos_norm_np = fc_two_np.copy()
                sam_pos_idxs_np = fc_two_np>0
                sam_pos_norm_np[sam_pos_idxs_np==0] = 0
                sam_pos_norm_np = sam_pos_norm_np + 1e-20
                sam_pos_norm_np = sam_pos_norm_np / np.sum(sam_pos_norm_np)


                sam_neg_norm_np = fc_two_np.copy()
                sam_neg_idxs_np = fc_two_np<0
                sam_neg_norm_np[sam_neg_idxs_np==0] = 0
                sam_neg_norm_np = np.abs(sam_neg_norm_np)
                sam_neg_norm_np = sam_neg_norm_np + 1e-20
                sam_neg_norm_np = sam_neg_norm_np / np.sum(sam_neg_norm_np)
                

                kl_corr_pos = np.sum(np.log(self.gen_classifiers_pos[lid_num]/sam_pos_norm_np)*self.gen_classifiers_pos[lid_num],axis=1)
                kl_corr_neg = np.sum(np.log(self.gen_classifiers_neg[lid_num]/sam_neg_norm_np)*self.gen_classifiers_neg[lid_num],axis=1)

                kl_corr_pos = np.exp(-kl_corr_pos)
                kl_corr_neg = np.exp(-kl_corr_neg)

                self.prob_scores_pos[int(lid_num)-1] = kl_corr_pos/np.sum(kl_corr_pos)
                self.prob_scores_neg[int(lid_num)-1] = kl_corr_neg/np.sum(kl_corr_neg)

