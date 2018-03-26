import caffe

import numpy as np
from PIL import Image
import scipy.io
from os.path import exists
import random
class NYUDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from NYUDv2
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 40 class task defined by

        S. Gupta, R. Girshick, p. Arbelaez, and J. Malik. Learning rich features
        from RGB-D images for object detection and segmentation. ECCV 2014.

    with 0 as the void label and 1-40 the classes.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - nyud_dir: path to NYUDv2 dir
        - split: train / val / test
        - tops: list of tops to output from {color, depth, hha, label}
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for NYUDv2 semantic segmentation.

        example: params = dict(nyud_dir="/path/to/NYUDVOC2011", split="val",
                               tops=['color', 'hha', 'label'])
        """
        # config
        params = eval(self.param_str)
        self.nyud_dir = params['nyud_dir']
        self.split = params['split']
        self.tops = params['tops']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        
        # aug params
        self.input_scale = 1.0
        self.do_aug = params.get('do_aug',False)
        self.do_test_scale = params.get('do_test_scale',1.0)
        self.do_scale_aug = False
        self.do_flip_aug = False
        self.do_crop_aug = False
        self.aug_scale = []
        self.aug_flip = []
        # store top data for reshape + forward
        self.data = {}

        # means
#        self.mean_bgr = np.array((116.190, 97.203, 92.318), dtype=np.float32)
        self.mean_bgr = np.array((128.0, 128.0, 128.0), dtype=np.float32)
        self.mean_hhc = np.array((132.431, 94.076, 111.8078), dtype=np.float32)

        self.mean_hha = np.array((132.431, 94.076, 118.477), dtype=np.float32)
        self.mean_xyz = np.array((112.203, 68.398, 97.728), dtype=np.float32)
#        self.mean_gha = np.array((102.325, 68.398, 97.728), dtype=np.float32)
       
#        self.mean_logd = np.array((7.844,), dtype=np.float32)
        self.mean_d = np.array((66.3793,), dtype=np.float32)

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.nyud_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)        
        
        if 'train' not in self.split:
            self.aug_scales = [self.do_test_scale]
        else:
            self.aug_scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]            
        self.aug_flips = [False , True]
        if self.do_aug:
            self.do_scale_aug = True # move to 
            if 'train' not in self.split:
                self.do_flip_aug = False
                self.do_crop_aug = False                
            else:
                self.do_flip_aug = True
                self.do_crop_aug = True
            # do augmentation (change an image for each iteration by now , batch size=1 )
            if self.do_scale_aug:
                self.aug_scale = self.aug_scales[random.randint(0,len(self.aug_scales)-1)]
#                print self.aug_scale
            if self.do_flip_aug:
                self.aug_flip = self.aug_flips[random.randint(0,len(self.aug_flips)-1)]

                                          
        self.crop_box_size= 400
        self.crop_box_ratio = 0.2
        self.crop_x = 0
        self.crop_y = 0
        self.crop_x_end = 300        
        self.crop_y_end = 300

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.indices[self.idx])
            top[i].reshape(1, *self.data[t].shape)

    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]
        
        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
                
        if self.do_scale_aug:
            self.aug_scale = self.aug_scales[random.randint(0,len(self.aug_scales)-1)]
        if self.do_flip_aug:
            self.aug_flip = self.aug_flips[random.randint(0,len(self.aug_flips)-1)]

            
                
    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, idx):
        if top == 'color' or top =='data':
            return self.load_image(idx)
        elif top == 'label4':
            return self.load_label4(idx)
        elif top == 'label':
            return self.load_label(idx)            
        elif top == 'label8':
            return self.load_label8(idx)                   
        elif top == 'depth':
            return self.load_depth(idx)
        elif top == 'hha':
            return self.load_hha(idx)   
        elif top == 'xyz':
            return self.load_xyz(idx)              
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/images_org/img_{}.png'.format(self.nyud_dir, idx))
        im = im.crop((7,7,im.size[0]-7,im.size[1]-7)) # shave white boundary   
       
        if(self.do_aug):
            if(self.do_flip_aug and self.aug_flip):
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if(self.do_scale_aug and (self.aug_scale>0)):
                d1 = int(round(im.size[0] * self.aug_scale))
                d2 = int(round(im.size[1] * self.aug_scale))
#                if(d1%2): d1 = d1+1
#                if(not d2%2): d2 = d2+1
                im = im.resize((d1,d2), Image.BICUBIC)
#                print im.size

                
            if(self.do_crop_aug):
#                if(self.crop_box_ratio>0):
#                    step_size = int(round(self.crop_box_size * self.crop_box_ratio))
#                    step_size = max(step_size,1)
#                else:
#                    step_size = 1
#                
#                max_range = [max(d1 - self.crop_box_size,0), max(d2 - self.crop_box_size,0)]
#                crop_d1s = range(0,max_range[0],step_size)
#                crop_d1s.append(max_range[0])
#                crop_d2s = range(0,max_range[1],step_size)
#                crop_d2s.append(max_range[1])
#                
#                self.crop_x = crop_d1s[random.randint(0, len(crop_d1s)-1)]
#                self.crop_y = crop_d2s[random.randint(0, len(crop_d2s)-1)]
#
#                self.crop_x_end =min(self.crop_x+self.crop_box_size-1,d1-1);
#                self.crop_y_end =min(self.crop_y+self.crop_box_size-1,d2-1);
#
#                if(not ((self.crop_y_end-self.crop_y-1)%2)): self.crop_y_end = self.crop_y_end-1
                im = im.crop((self.crop_x,self.crop_y,self.crop_x_end+1,self.crop_y_end+1))   
                im = im.resize( (int(round(im.size[0]*self.input_scale)), int(round(im.size[1]*self.input_scale))), Image.BICUBIC)
           
                         
#        im = Image.open('{}/images/{}.png'.format(self.nyud_dir, idx))
#        if 'trainval' not in self.split:
#            print(idx)
#        if( (im.size[0]%32 !=0) or (im.size[1] %32 !=0) ):
#            pad_im = Image.new("RGB",(int(32*np.ceil(im.size[0]/32.0)),int(32*np.ceil(im.size[1]/32.0))),(128,128,128))
#            pad_im.paste(im,(0,0))
#            im = pad_im
##            im = im.resize( (int(32*np.ceil(im.size[0]/32.0)),int(32*np.ceil(im.size[1]/32.0))), Image.BILINEAR)
#            del pad_im

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean_bgr
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label4(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-39 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
#        random.seed()
        if(exists('{}/segmentation_org/img_{}.mat'.format(self.nyud_dir, idx))):
            label = scipy.io.loadmat('{}/segmentation_org/img_{}.mat'.format(self.nyud_dir, idx))['segmentation'].astype(np.uint8)
            
    #        lb = Image.open('{}/label/{}.png'.format(self.nyud_dir,idx))
    #        lb = Image.open('{}/labels/img_{}.png'.format(self.nyud_dir,idx))
    #        label = np.array(lb,dtype=np.uint8)
            
            label -= 1  # rotate labels

            if(self.do_aug):
                label = Image.fromarray(label) #numpy.asarray
                label = label.crop((7,7,label.size[0]-7,label.size[1]-7)) # white boundary
                
                if(self.do_flip_aug and self.aug_flip):
                    label = label.transpose(Image.FLIP_LEFT_RIGHT)
                if(self.do_scale_aug and self.aug_scale):
                    d1 = int(round(label.size[0] * self.aug_scale))
                    d2 = int(round(label.size[1] * self.aug_scale))
                    label = label.resize((d1,d2), Image.NEAREST)

                if(self.do_crop_aug):
                    if(0): #old
                        if(self.crop_box_ratio>0):
                            step_size = int(round(self.crop_box_size * self.crop_box_ratio))
                            step_size = max(step_size,1)
                        else:
                            step_size = 1
                        
                        max_range = [max(d1 - self.crop_box_size,0), max(d2 - self.crop_box_size,0)]
                        crop_d1s = range(0,max_range[0],step_size)
                        crop_d1s.append(max_range[0])
                        crop_d2s = range(0,max_range[1],step_size)
                        crop_d2s.append(max_range[1])
                        
                        self.crop_x = crop_d1s[random.randint(0, len(crop_d1s)-1)]
                        self.crop_y = crop_d2s[random.randint(0, len(crop_d2s)-1)]
                    else:
                        label = np.asarray(label,dtype=np.uint8)
                        weak_label = np.unique(label)
                        if(weak_label[len(weak_label)-1]==255):
                            weak_label = weak_label[0:-1]
                        sampled_cls = weak_label[random.randint(0,len(weak_label)-1)]
                        sampled_cls_pixels = np.argwhere(label==sampled_cls)
                        sampled_loc = sampled_cls_pixels[random.randint(0,len(sampled_cls_pixels)-1)]
                        if( (sampled_loc[0]+round(self.crop_box_size/2)-1) > d2-1 ):
                            sampled_loc[0] = d2 - round(self.crop_box_size/2)
                        if( (sampled_loc[1]+round(self.crop_box_size/2)-1) > d1-1 ):
                            sampled_loc[1] = d1 - round(self.crop_box_size/2)

                        self.crop_x = max(0,sampled_loc[1]-round(self.crop_box_size/2))
                        self.crop_y = max(0,sampled_loc[0]-round(self.crop_box_size/2))
                        label = Image.fromarray(label) #numpy.asarray
    
                                           
                    self.crop_x_end =min(self.crop_x+self.crop_box_size-1,d1-1);
                    self.crop_y_end =min(self.crop_y+self.crop_box_size-1,d2-1);
    
#                    if(not ((self.crop_y_end-self.crop_y-1)%2)): self.crop_y_end = self.crop_y_end-1                    
                    label = label.crop((self.crop_x,self.crop_y,self.crop_x_end+1,self.crop_y_end+1))              

#                print label.shape
        else:
            im = Image.open('{}/images/img_{}.png'.format(self.nyud_dir, idx))
            w,h = im.size
            label = np.zeros([h, w],dtype=np.uint8)
            label = label + 255

        label = label.resize((int(round(label.size[0]*self.input_scale)),int(round(label.size[1]*self.input_scale))), Image.NEAREST) # scale input  
#        if( (label.size[0]%32 !=0) or (label.size[1] %32 !=0) ):
#            pad_lab = Image.new("L",(int(32*np.ceil(label.size[0]/32.0)),int(32*np.ceil(label.size[1]/32.0))),255)
#            pad_lab.paste(label,(0,0))
#            label = pad_lab
#            del pad_lab
##            label = label.resize( (int(32*round(label.size[0]/32.0)),int(32*round(label.size[1]/32.0))), Image.NEAREST)               
#            
#        #1/4downsampled label for training            
        label = label.resize( (int(round((round(label.size[0]/2.0)-1)/2)),int(round((round(label.size[1]/2.0)-1)/2))),Image.NEAREST )
##        if( (label.size[0]%32 !=0) or (label.size[1] %32 !=0) ):
##            label.resize( (int(round(label.size[0]/32.0)),int(round(label.size[1]/32.0))), Image.NEAREST)
#            
        label = np.asarray(label,dtype=np.uint8)
            
        label = label[np.newaxis, ...]
        return label

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-39 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        if(exists('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))):
            label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))['segmentation'].astype(np.uint8)
    #        lb = Image.open('{}/label/{}.png'.format(self.nyud_dir,idx))
    #        lb = Image.open('{}/labels/img_{}.png'.format(self.nyud_dir,idx))
    #        label = np.array(lb,dtype=np.uint8)
            
            label -= 1  # rotate labels
        else:
            im = Image.open('{}/images/img_{}.png'.format(self.nyud_dir, idx))
            w,h = im.size
            label = np.zeros([h, w],dtype=np.uint8)
            label= label + 255
        label = label[np.newaxis, ...]
        return label       
    def load_label8(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-39 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        if(exists('{}/segmentation8/img_{}.mat'.format(self.nyud_dir, idx))):
            label8 = scipy.io.loadmat('{}/segmentation8/img_{}.mat'.format(self.nyud_dir, idx))['label_8'].astype(np.uint8)
    #        lb = Image.open('{}/label/{}.png'.format(self.nyud_dir,idx))
    #        lb = Image.open('{}/labels/img_{}.png'.format(self.nyud_dir,idx))
    #        label = np.array(lb,dtype=np.uint8)
            
            label8 -= 1  # rotate labels
        else:
            im = Image.open('{}/images/img_{}.png'.format(self.nyud_dir, idx))
            w,h = im.size
            label8 = np.zeros([h, w],dtype=np.uint8)
            label8 = label8 + 255
        label8 = label8[np.newaxis, ...]
        return label8               
    def load_depth(self, idx):
        """
        Load pre-processed depth for NYUDv2 segmentation set.
        """
        im = Image.open('{}/depth_org/img_{}.png'.format(self.nyud_dir, idx))
        im = im.crop((7,7,im.size[0]-7,im.size[1]-7)) # white boundary        
        if(self.do_aug):
            if(self.do_flip_aug and self.aug_flip):
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if(self.do_scale_aug and (self.aug_scale>0)):
                d1 = int(round(im.size[0] * self.aug_scale))
                d2 = int(round(im.size[1] * self.aug_scale))
#                if(d1%2): d1 = d1+1
#                if(not d2%2): d2 = d2+1
                im = im.resize((d1,d2), Image.BICUBIC)
#                print im.size

                
            if(self.do_crop_aug):
#                if(self.crop_box_ratio>0):
#                    step_size = int(round(self.crop_box_size * self.crop_box_ratio))
#                    step_size = max(step_size,1)
#                else:
#                    step_size = 1
#                
#                max_range = [max(d1 - self.crop_box_size,0), max(d2 - self.crop_box_size,0)]
#                crop_d1s = range(0,max_range[0],step_size)
#                crop_d1s.append(max_range[0])
#                crop_d2s = range(0,max_range[1],step_size)
#                crop_d2s.append(max_range[1])
#                
#                self.crop_x = crop_d1s[random.randint(0, len(crop_d1s)-1)]
#                self.crop_y = crop_d2s[random.randint(0, len(crop_d2s)-1)]
#
#                self.crop_x_end =min(self.crop_x+self.crop_box_size-1,d1-1);
#                self.crop_y_end =min(self.crop_y+self.crop_box_size-1,d2-1);
#
#                if(not ((self.crop_y_end-self.crop_y-1)%2)): self.crop_y_end = self.crop_y_end-1
                im = im.crop((self.crop_x,self.crop_y,self.crop_x_end+1,self.crop_y_end+1))   
                im = im.resize( (int(round(im.size[0]*self.input_scale)), int(round(im.size[1]*self.input_scale))), Image.BICUBIC)
           
                         
#        im = Image.open('{}/images/{}.png'.format(self.nyud_dir, idx))
#        if 'trainval' not in self.split:
#            print(idx)
#        if( (im.size[0]%32 !=0) or (im.size[1] %32 !=0) ):
#            pad_im = Image.new("RGB",(int(32*np.ceil(im.size[0]/32.0)),int(32*np.ceil(im.size[1]/32.0))),(128,128,128))
#            pad_im.paste(im,(0,0))
#            im = pad_im
##            im = im.resize( (int(32*np.ceil(im.size[0]/32.0)),int(32*np.ceil(im.size[1]/32.0))), Image.BILINEAR)
#            del pad_im
        
#        im = Image.open('{}/hha/{}.png'.format(self.nyud_dir, idx))        
        
        d = np.array(im, dtype=np.float32)
#        d = np.log(d)
#        d -= self.mean_logd
        d = 150000 / d
        d -= self.mean_d
        d = np.repeat(d[:,:,np.newaxis],3,axis=2) #duplicate to 3d 
        d = d.transpose((2,0,1))

#        d = d[np.newaxis, ...]
        return d

    def load_hha(self, idx):
        """
        Load HHA features from Gupta et al. ECCV14.
        See https://github.com/s-gupta/rcnn-depth/blob/master/rcnn/saveHHA.m
        """
        im = Image.open('{}/hha_org/img_{}.png'.format(self.nyud_dir, idx))
        im = im.crop((7,7,im.size[0]-7,im.size[1]-7)) # white boundary        
        if(self.do_aug):
            if(self.do_flip_aug and self.aug_flip):
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if(self.do_scale_aug and (self.aug_scale>0)):
                d1 = int(round(im.size[0] * self.aug_scale))
                d2 = int(round(im.size[1] * self.aug_scale))
#                if(d1%2): d1 = d1+1
#                if(not d2%2): d2 = d2+1
                im = im.resize((d1,d2), Image.BICUBIC)
#                print im.size

                
            if(self.do_crop_aug):
#                if(self.crop_box_ratio>0):
#                    step_size = int(round(self.crop_box_size * self.crop_box_ratio))
#                    step_size = max(step_size,1)
#                else:
#                    step_size = 1
#                
#                max_range = [max(d1 - self.crop_box_size,0), max(d2 - self.crop_box_size,0)]
#                crop_d1s = range(0,max_range[0],step_size)
#                crop_d1s.append(max_range[0])
#                crop_d2s = range(0,max_range[1],step_size)
#                crop_d2s.append(max_range[1])
#                
#                self.crop_x = crop_d1s[random.randint(0, len(crop_d1s)-1)]
#                self.crop_y = crop_d2s[random.randint(0, len(crop_d2s)-1)]
#
#                self.crop_x_end =min(self.crop_x+self.crop_box_size-1,d1-1);
#                self.crop_y_end =min(self.crop_y+self.crop_box_size-1,d2-1);
#
#                if(not ((self.crop_y_end-self.crop_y-1)%2)): self.crop_y_end = self.crop_y_end-1
                im = im.crop((self.crop_x,self.crop_y,self.crop_x_end+1,self.crop_y_end+1))   
                im = im.resize( (int(round(im.size[0]*self.input_scale)), int(round(im.size[1]*self.input_scale))), Image.BICUBIC)
           
                         
#        im = Image.open('{}/images/{}.png'.format(self.nyud_dir, idx))
#        if 'trainval' not in self.split:
#            print(idx)
#        if( (im.size[0]%32 !=0) or (im.size[1] %32 !=0) ):
#            pad_im = Image.new("RGB",(int(32*np.ceil(im.size[0]/32.0)),int(32*np.ceil(im.size[1]/32.0))),(128,128,128))
#            pad_im.paste(im,(0,0))
#            im = pad_im
##            im = im.resize( (int(32*np.ceil(im.size[0]/32.0)),int(32*np.ceil(im.size[1]/32.0))), Image.BILINEAR)
#            del pad_im
        
#        im = Image.open('{}/hha/{}.png'.format(self.nyud_dir, idx))
        
        hha = np.array(im, dtype=np.float32)
        hha -= self.mean_hha
        hha = hha.transpose((2,0,1))
        return hha
        
    def load_xyz(self, idx):
        """
        Load xyz features
        """
        im = Image.open('{}/xyz/img_{}.png'.format(self.nyud_dir, idx))
#        im = Image.open('{}/hha/{}.png'.format(self.nyud_dir, idx))

        xyz = np.array(im, dtype=np.float32)
        xyz -= self.mean_xyz
        xyz = xyz.transpose((2,0,1))
        return xyz
        

        
