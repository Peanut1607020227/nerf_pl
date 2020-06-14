import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *



import torchvision.transforms as T

from torch.utils import data
import torch

from tqdm import tqdm

import numpy as np

import random
import PIL
from PIL import Image
import collections
import math
import copy

from .ibr_dynamic import IBRDynamicDataset


class Cam_Transforms(object):
    def __init__(self, size, focal ,interpolation=Image.BICUBIC):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.focal = focal
        
    def __call__(self, img, Ks = None, Ts = None,  mask = None):

        K = Ks.clone()
        Tc = Ts.clone()
        img_np = np.asarray(img)

 
        width, height = img.size



        
        m_scale = height/self.size[0]

        cx, cy = 0, 0


        ration_w = self.focal / K[0,0] * m_scale
        ration_h = self.focal / K[1,1] * m_scale

        translation = [0,0]
        translation[1] = (self.size[0]/2)/(self.size[0]*ration_h  / height) - K[1,2]
        translation[0] = (self.size[1]/2)/(self.size[0]*ration_w  / height) - K[0,2]
        translation = tuple(translation)

        #translation = (width /2-K[0,2],height/2-K[1,2])

        
        #img = T.functional.rotate(img, angle = np.rad2deg(rotation), resample = Image.BICUBIC, center =(K[0,2],K[1,2]))
        img = T.functional.affine(img, angle = 0, translate = translation, scale= 1,shear=0)
        img = T.functional.crop(img, 0, 0,  int(height/ration_h),int(height*self.size[1]/ration_w/self.size[0]) )
        img = T.functional.resize(img, self.size, self.interpolation)
        img = T.functional.to_tensor(img)

        
        ROI = np.ones_like(img_np)*255.0

        ROI = Image.fromarray(np.uint8(ROI))
        #ROI = T.functional.rotate(ROI, angle = np.rad2deg(rotation), resample = Image.BICUBIC, center =(K[0,2],K[1,2]))
        ROI = T.functional.affine(ROI, angle = 0, translate = translation, scale= 1,shear=0)
        ROI = T.functional.crop(ROI, 0,0, int(height/ration_h),int(height*self.size[1]/ration_w/self.size[0]) )
        
        ROI = T.functional.resize(ROI, self.size, self.interpolation)
        ROI = T.functional.to_tensor(ROI)
        ROI = ROI[0:1,:,:]
        

        
        if mask is not None:
            #mask = T.functional.rotate(mask, angle = np.rad2deg(rotation), resample = Image.BICUBIC, center =(K[0,2],K[1,2]))
            mask = T.functional.affine(mask, angle = 0, translate = translation, scale= 1,shear=0)
            mask = T.functional.crop(mask, 0, 0,  int(height/ration_h),int(height*self.size[1]/ration_w/self.size[0]) )
            mask = T.functional.resize(mask, self.size, self.interpolation)
            mask = T.functional.to_tensor(mask)
        else:
            mask = torch.zeros((1,)+self.size,device = img.device)
        
               
        #K = K / m_scale
        #K[2,2] = 1


        K[0,2] = K[0,2] + translation[0]
        K[1,2] = K[1,2] + translation[1]

        K[0,1] = 0  

        s = self.size[0] * ration_w / height

        K[0,:] = K[0,:]*s


        s = self.size[0] * ration_h / height
        K[1,:] = K[1,:]*s

              
        return img, K, Tc, mask, ROI
    
    def __repr__(self):
        return self.__class__.__name__ + '()'



def gen_pose(poses, num =8):
    res = copy.deepcopy(poses)
    pose = poses[17]

    z = -pose[0:3,2]


    for i in range(-num,0):
        t = pose.copy()
        t[0:3,3] += (i+ num//2)*z*0.3 
        res[i] = t

    return res









def IBRay_NHR(data_folder_path, h,w,focal,skip_step ):




    transforms = Cam_Transforms((h,w), focal)
    transforms = transforms


    NHR_dataset = IBRDynamicDataset(data_folder_path, 1, False, transforms, [1.0, 6.5, 0.8], skip_step = skip_step, random_noisy = 0, holes='None')


    if not os.path.exists(os.path.join(data_folder_path,'nerf_pl.pth')):


        all_imgs = []
        all_poses = []

        all_mask = []
        counts = [0,len(NHR_dataset),len(NHR_dataset)*2-20,len(NHR_dataset)*2]
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        for i in tqdm(range(len(NHR_dataset))):
            img, vs, _, T, K, _,_ = NHR_dataset.__getitem__(i)
            img_rgb = img[0:3,:,:].permute(1,2,0).cpu().numpy()

            mask = img[4,:,:]
            

            T[0:3,1] = -T[0:3,1]
            T[0:3,2] = -T[0:3,2]


            all_imgs.append(img_rgb)
            all_poses.append(T.cpu().numpy())
            all_mask.append(mask.cpu().numpy())



        imgs = np.stack(all_imgs, 0)
        poses = np.stack(all_poses, 0)
        masks = np.stack(all_mask, 0)

        H, W = imgs[0].shape[:2]
            
        data = {'imgs':imgs, 'poses':poses, 'hwf': [H, W, focal],'masks':masks,'near_far':(NHR_dataset.near,NHR_dataset.far)}

        torch.save(data,os.path.join(data_folder_path,'nerf_pl.pth'))
    else:
        data = torch.load(os.path.join(data_folder_path,'nerf_pl.pth'))
        imgs = data['imgs']
        poses = data['poses']
        H, W, focal = data['hwf']
        masks = data['masks']
        NHR_dataset.near,NHR_dataset.far = data['near_far']




    print('near far',NHR_dataset.near,NHR_dataset.far)

    return imgs, poses, [H, W, focal], masks,NHR_dataset.near,NHR_dataset.far










class NHRDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800),focal = 2200):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh

        self.focal = focal


        self.read_meta()
        self.white_back = False

    def read_meta(self):

        h = self.img_wh[1]
        w = self.img_wh[0]


        images, poses, hwf, mask_nhr,near,far = IBRay_NHR(self.root_dir,w,h,self.focal,1)

        self.data = {'image':images,'pose' :poses, 'mask':mask_nhr}
        


        # bounds, common for all scenes
        self.near = near
        self.far = far
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)


        print(self.directions.size(),self.focal)
            
        if self.split == 'train': # create buffer of all rays and rgb data
  
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for i in range(images.shape[0]):
                pose = poses[i,:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)


                img = torch.tensor(images[i])
                img = img.view(-1,3)     # (h*w, 3) RGB
                mask = torch.tensor(mask_nhr[i]).reshape(-1)




                img = img[mask>0.8,:]

                self.all_rgbs += [img]

                

            


                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)



                rays_o = rays_o[mask>0.8,:]
                rays_d = rays_d[mask>0.8,:]

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)



    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        
        return 8 # only validate 8 images (to support <=8 gpus)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately

            c2w = torch.FloatTensor(self.data['pose'][idx][:3, :4])

            img = self.data['image'][idx]
            img = torch.tensor(img).view(-1,3)     # (h*w, 3) RGB


            rays_o, rays_d = get_rays(self.directions, c2w)

            valid_mask = (torch.tensor(self.data['mask'])[idx]>0.8).flatten()

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample