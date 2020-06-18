#!/usr/bin/env python
# coding: utf-8

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
  #      print(lay_out.keys())


        for l_key  in lay_out.keys():
  #          print(l_key.split('_'))
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

                # KLpos[lid_num] = np.exp(-kl_corr_pos)
                # KLneg[lid_num] = np.exp(-kl_corr_neg)

                kl_corr_pos = np.exp(-kl_corr_pos)
                kl_corr_neg = np.exp(-kl_corr_neg)

                self.prob_scores_pos[int(lid_num)-1] = kl_corr_pos/np.sum(kl_corr_pos)
                self.prob_scores_neg[int(lid_num)-1] = kl_corr_neg/np.sum(kl_corr_neg)

        # return KLpos, KLneg




# split = 'test'
# save_path='/media/lokender/MyPassport/rebuttal_cifar10/intermediate_outputs/layer_outputs/'


# device = 'cuda'
# PATH = './model_best.pth.tar'
# model = vgg.__dict__['vgg19']()
# model.features = torch.nn.DataParallel(model.features)

# checkpoint = torch.load(PATH)
# best_prec1 = checkpoint['best_prec1']
# model.load_state_dict(checkpoint['state_dict'])
# model.to(device)



# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# transform=transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#         ])


# # switch to evaluate mode
# model.eval()
# vgg_version='vgg19'
# for cid in range(10):
#     samples = glob.glob('/media/lokender/MyPassport/rebuttal_cifar10/split_cifar10_data/'+split+'/'+str(cid)+'/*.mat')
#     acc =0
#     for i in range(len(samples)):
#     #for i in range(1):
# #        print(i)
#     #    print(samples[i])
#         print('cid:{},sid:{}'.format(cid, i))
#         data = sio.loadmat(samples[i])
#         input = Image.fromarray(data['img'])
#         input = transform(input).unsqueeze(0).to(device)

#         label = torch.tensor(data['label'])[0][0].to(device)

#         # compute output
#         with torch.no_grad():
#             output = model(input)
#             out = torch.softmax(output,dim=1)
#             soft_ordered,soft_ordered_idx = torch.sort(out.data,  dim=1,descending=True)


#         top_soft =soft_ordered[0,:]         # TOP-20 CLASSES
#         top_class=soft_ordered_idx[0,:]

#         pred_class_id = top_class[0]

#         if pred_class_id == label:
#             acc = acc+1  

#           #  print('success')
#             rdl = robdl(eval_model=model,device=device)
#             hook_outputs=rdl.hook_Layer_VGG(vgg_ver=vgg_version,img=input)
#             save_name = save_path+split+'/'+str(label.cpu().numpy())+'/'+samples[i].split('/')[-1].split('.')[0].split('/')[-1]+'_prefs'
#           #  print(save_name)
#             prefs = {}
#             lnames =[]
#             for l_key  in hook_outputs.keys():
#                 lnames.append(l_key)
#                 if l_key.split('_')[-1] == 'conv':
#                     tmp_layer = hook_outputs[l_key].view(hook_outputs[l_key].shape[0],-1).to(device)
#                     pos_ind = tmp_layer>0
#                     neg_ind = tmp_layer<0

#                     prefs[l_key+'_p_count'] = torch.sum(pos_ind,dim=1).cpu().detach().numpy()
#                     prefs[l_key+'_n_count'] = torch.sum(neg_ind,dim=1).cpu().detach().numpy()
#                     prefs[l_key+'_p_sum'] = torch.sum(torch.mul(pos_ind.to(device).float(),tmp_layer),dim=1).cpu().detach().numpy()
#                     prefs[l_key+'_n_sum'] = torch.sum(torch.mul(neg_ind.to(device).float(),tmp_layer),dim=1).cpu().detach().numpy()
#                     prefs[l_key+'_p_count_minK'] = torch.sum(torch.sum(pos_ind,dim=1)>0).cpu().detach().numpy()
#                     prefs[l_key+'_n_count_minK'] = torch.sum(torch.sum(neg_ind,dim=1)>0).cpu().detach().numpy()
#                 else:
#                     prefs[l_key] = hook_outputs[l_key].cpu().detach().numpy()

#                 prefs['top_soft']  = top_soft.cpu().detach().numpy()
#                 prefs['top_class'] = top_class.cpu().detach().numpy()
#                 prefs['wnid'] = label
#                 prefs['layers'] = lnames

#             sio.savemat(save_name+'.mat',prefs)

#     # print('acc:{},cid:{},label:{}'.format(acc,cid, label))
