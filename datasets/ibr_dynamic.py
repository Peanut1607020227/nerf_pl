import torch
import cv2
import numpy as np
import os
from PIL import Image
import torchvision
import torch.distributions as tdist



def campose_to_extrinsic(camposes):
    if camposes.shape[1]!=12:
        raise Exception(" wrong campose data structure!")
        return
    
    res = np.zeros((camposes.shape[0],4,4))
    
    res[:,0:3,2] = camposes[:,0:3]
    res[:,0:3,0] = camposes[:,3:6]
    res[:,0:3,1] = camposes[:,6:9]
    res[:,0:3,3] = camposes[:,9:12]
    res[:,3,3] = 1.0
    
    return res


def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data= fo.readlines()
    i = 0
    Ks = []
    while i<len(data):
        if len(data[i])>5:
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            a = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            b = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            c = np.array(tmp)
            res = np.vstack([a,b,c])
            Ks.append(res)

        i = i+1
    Ks = np.stack(Ks)
    fo.close()

    return Ks




def merge_holes(pc1,pc2):

    # change point color here

    return np.concatenate([pc1, pc2], axis=0)


class IBRDynamicDataset(torch.utils.data.Dataset):

    def __init__(self,data_folder_path, frame_num, use_mask, transforms, near_far_size, skip_step, random_noisy,holes,NOH = False):
        super(IBRDynamicDataset, self).__init__()

        self.frame_num = frame_num
        self.data_folder_path = data_folder_path
        self.use_mask = use_mask
        self.skip_step = skip_step
        self.random_noisy  =random_noisy
        self.holes = holes

        self.NOH = NOH



        self.file_path = os.path.join(data_folder_path,'img')

        self.vs = []
        self.vs_rgb = []
        self.vs_num = []
        self.vs_index =[]

        sum_tmp = 0
        for i in range(frame_num):
            #tmp = np.loadtxt(os.path.join(data_folder_path,'pointclouds/frame%d.obj' % (i+1)), usecols = (1,2,3,4,5,6))
            tmp = np.load(os.path.join(data_folder_path,'pointclouds/frame%d.npy' % (i+1)))

            if os.path.exists(os.path.join(self.holes,'holes/frame%d.npy' % (i+1))):
                tmp2 = np.load(os.path.join(self.holes,'holes/frame%d.npy' % (i+1)))
                tmp = merge_holes(tmp, tmp2)
                if i%50 == 0:
                    print('merge holes', tmp2.shape[0])


            vs_tmp = tmp[:,0:3] 
            vs_rgb_tmp = tmp[:,3:6]
            self.vs_index.append(sum_tmp)
            self.vs.append(torch.Tensor(vs_tmp))
            self.vs_rgb.append(torch.Tensor(vs_rgb_tmp))
            self.vs_num.append(vs_tmp.shape[0])
            sum_tmp = sum_tmp + vs_tmp.shape[0]
            
            if i%50 == 0:
                print(i,'/',frame_num)


        self.vs = torch.cat( self.vs, dim=0 )
        self.vs_rgb = torch.cat( self.vs_rgb, dim=0 )

        if random_noisy>0:
            n = tdist.Normal(torch.tensor([0.0, 0.0,0.0]), torch.tensor([random_noisy,random_noisy,random_noisy]))
            kk = torch.min((torch.max(self.vs,dim = 1)[0] - torch.min(self.vs,dim = 1)[0])/500)
            self.vs = self.vs + kk*n.sample((self.vs.size(0),))
        
        

        camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        self.cam_num = self.Ts.size(0)
        
        self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))
        '''
        for i in range(self.Ks.size(0)):
            if self.Ks[i,0,2] > 1100:
                self.Ks[i] = self.Ks[i] * 2048.0/2448.0
                self.Ks[i] = self.Ks[i] / (2048.0/800)
            else:
                self.Ks[i] = self.Ks[i] / (2048.0/800)

        self.Ks[:,2,2] = 1
        '''

        self.transforms = transforms
        self.near_far_size = torch.Tensor(near_far_size)

        #self.black_list = [625,747,745,738,62,750,746,737,739,762]

        print('load %d Ts, %d Ks, %d frame, %d vertices' % (self.Ts.size(0),self.Ks.size(0),self.frame_num,self.vs.size(0)))


        self._all_imgs = None
        self._all_Ts = None
        self._all_Ks = None
        self._all_width_height = None


        inv_Ts = torch.inverse(self.Ts).unsqueeze(1)  #(M,1,4,4)
        vs = self.vs.clone().unsqueeze(-1)   #(N,3,1)
        vs = torch.cat([vs,torch.ones(vs.size(0),1,vs.size(2)) ],dim=1) #(N,4,1)

        vs = vs[0:-1:5,:,:]

        pts = torch.matmul(inv_Ts,vs) #(M,N,4,1)

        pts = torch.max(pts, dim=1)[0].squeeze() #(M,4)

        pts_max = torch.max(pts, dim=0)[0]   #(4)
        pts_min = torch.min(pts, dim=0)[0]   #(4)

        self.near = pts_min[2].item() *0.5
        
        self.far = pts_max[2].item() *2

        self.near = max(self.near,self.far*0.1)

        print('dataset initialed.')




    def __len__(self):
        return (self.cam_num//self.skip_step) *  (self.frame_num) 

    def __getitem__(self, index, need_transform = True):

        frame_id = (index // (self.cam_num//self.skip_step))  %self.frame_num
        cam_id = (index*self.skip_step) % (self.cam_num)
        
        if not self.NOH:
            img = Image.open(os.path.join(self.file_path,'%d/img_%04d.jpg' % ( frame_id, cam_id)))
        else:
            img = Image.open(os.path.join(self.file_path,'%d/img_%04d.jpg' % ( frame_id, cam_id+1)))

        K = self.Ks[cam_id]

        if self.use_mask:
            if not self.NOH:
                img_mask = Image.open(os.path.join(self.file_path,'%d/mask/img_%04d.jpg' % ( frame_id, cam_id)))
            else:
                img_mask = Image.open(os.path.join(self.file_path,'%d/img_%04d_alpha.png' % ( frame_id, cam_id+1)))

            img,K,T,img_mask, ROI = self.transforms(img,self.Ks[cam_id],self.Ts[cam_id],img_mask)
            img = torch.cat([img,img_mask[0:1,:,:]], dim=0)
        else:
            img,K,T,img_mask, ROI = self.transforms(img,self.Ks[cam_id],self.Ts[cam_id])
            img = torch.cat([img,img_mask[0:1,:,:]], dim=0)
        
        img = torch.cat([img,ROI], dim=0)


        return img, self.vs[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:], index, T, K, self.near_far_size, self.vs_rgb[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:]

    def get_vertex_num(self):
        return torch.Tensor(self.vs_num)





