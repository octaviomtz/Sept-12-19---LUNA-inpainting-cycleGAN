import torch
import torch.nn as nn
import torchvision
import sys
from itertools import groupby, count
import pdb

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1], spatial_size[2]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False
        
def optimize3(optimizer_type, parameters, closure, LR, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []

    if optimizer_type == 'adam':
        #print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        lr_finder = 1e-7
        lrs_finder = []
        for j in range(num_iter):
            
            # LR finder
            if lr_finder_flag:
                if j == 0: print('Finding LR')
                lr_finder = lr_finder * 1.1
                if lr_finder >= 1: #completed
                    return total_loss, images_generated, j_best, 
                lrs_finder.append(lr_finder)
                optimizer = torch.optim.Adam(parameters, lr=lr_finder)
            
            # Init values for best loss selection
            if j == 0:
                loss_best = 1000
                j_best = 0
                
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp)
            
            # Restart if bad initialization
            if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
                """if the network is not learning (bad initialization) Restart"""
                restart = True
                return total_loss, images_generated, j_best, restart
            else:
                restart = False
               
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 500: # the first 500 iterations are blurry
                    images_generated.append(image_generated_temp)
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            if annealing == True and iterations_without_improvement == 300:
                LR = adjust_learning_rate(optimizer, j, LR)
                print(f'LR reduced to: {LR:.5f}')
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 500:
                print(f'training stopped at it {j} after 500 iterations without improvement')
                break
    else:
        assert False
    return total_loss, images_generated, j_best, restart

def optimize4_no_tqdm(optimizer_type, parameters, closure, LR, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    iterations_without_improvement = 0
    save_even_images=0

    if optimizer_type == 'adam':
        #print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        lr_finder = 1e-7
        lrs_finder = []
        for _, j in enumerate(range(num_iter)):
            
            # LR finder
            if lr_finder_flag:
                if j == 0: print('Finding LR')
                lr_finder = lr_finder * 1.2
                if lr_finder >= 1e-2: #completed
                    return total_loss, images_generated, j_best, 
                lrs_finder.append(lr_finder)
                optimizer = torch.optim.Adam(parameters, lr=lr_finder)
            
            # Init values for best loss selection
            if j == 0:
                loss_best = 1000
                j_best = 0
                
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp.detach().cpu().numpy())
            
            # Restart if bad initialization
            if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
                """if the network is not learning (bad initialization) Restart"""
                restart = True
                return total_loss, images_generated, j_best, restart
            else:
                restart = False
               
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 40: #500 
                    #if 10000 % j == 0 or (j == num_iter -1): # added in inpainting v11 REMOVE: impeding last img to be saved
                    images_generated = image_generated_temp
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            if annealing == True and iterations_without_improvement == 10 or j % 400==0: #200
                LR = lr_finder_function(parameters, closure)
                optimizer = torch.optim.Adam(parameters, lr=lr_finder)
                #LR = adjust_learning_rate(optimizer, j, LR)
                print(f'LR reduced to: {LR:.5f}')
            #print(f'total_loss_temp:{total_loss_temp}')

            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 500:
                print(f'training stopped at it {j} after 500 iterations without improvement')
                break
    else:
        assert False
    return total_loss, images_generated, j_best, restart

def optimize4_loss_multiplied(optimizer_type, parameters, closure, LR, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    iterations_without_improvement = 0
    save_even_images=0

    if optimizer_type == 'adam':
        #print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        lr_finder = 1e-7
        lrs_finder = []
        for _, j in tqdm(enumerate(range(num_iter))):
            
            # LR finder
            if lr_finder_flag:
                if j == 0: print('Finding LR')
                lr_finder = lr_finder * 1.2
                if lr_finder >= 1e-2: #completed
                    return total_loss, images_generated, j_best, 
                lrs_finder.append(lr_finder)
                optimizer = torch.optim.Adam(parameters, lr=lr_finder)
            
            # Init values for best loss selection
            if j == 0:
                loss_best = 1000
                j_best = 0
                optimizer.zero_grad()
                total_loss_temp, image_generated_temp = closure()
                total_loss.append(total_loss_temp.detach().cpu().numpy())
            # Training 
            elif total_loss_temp > 2e-4:
                optimizer.zero_grad()
                total_loss_temp, image_generated_temp = closure()
                total_loss.append(total_loss_temp.detach().cpu().numpy())
            # Training when loss is too small
            else:
                optimizer.zero_grad()
                total_loss_temp, image_generated_temp = closure(100)
                total_loss.append(total_loss_temp.detach().cpu().numpy())
            
            # Restart if bad initialization
            if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
                """if the network is not learning (bad initialization) Restart"""
                restart = True
                return total_loss, images_generated, j_best, restart
            else:
                restart = False
               
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 500: #500 
                    #if 10000 % j == 0 or (j == num_iter -1): # added in inpainting v11 REMOVE: impeding last img to be saved
                    images_generated = image_generated_temp
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            if annealing == True and iterations_without_improvement == 200:
                LR = adjust_learning_rate(optimizer, j, LR)
                print(f'LR reduced to: {LR:.5f}')
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 2000: #500
                print(f'training stopped at it {j} after 2000 (v15_loss_multiplied version) iterations without improvement')
                break
    else:
        assert False
    return total_loss, images_generated, j_best, restart

def find_optimal_lr(mse_error_lr, lrs_finder):
    loss_going_down = np.where(np.diff(mse_error_lr) < -1e-7) # indices that go down (negative diff)
    loss_going_down = list(loss_going_down[0] + 1) # for each pair of indices with neg diff take the 2nd one and convert to list
    c = count()
    val = max((list(g) for _, g in groupby(loss_going_down, lambda x: x-next(c))), key=len) # longest sequence of negative diff
    val = list(val)
    #pdb.set_trace()
    slope_diff = np.diff(mse_error_lr[val])
    largest_diff = np.where(slope_diff == np.min(slope_diff))[0]
    LR = lrs_finder[val[largest_diff[0]]]
    return LR, largest_diff, val

def lr_finder_function(parameters, closure):
    lr_finder = 1e-7
    lrs_finder = []
    total_loss = []
    while lr_finder <= 1e-2:
        lr_finder = lr_finder * 1.2
        lrs_finder.append(lr_finder)
        optimizer = torch.optim.Adam(parameters, lr=lr_finder)
        optimizer.zero_grad()
        total_loss_temp, image_generated_temp = closure()
        total_loss.append(total_loss_temp.detach().cpu().numpy())
        optimizer.step()
    total_loss = np.squeeze(total_loss)
    LR, largest_diff, val = find_optimal_lr(total_loss, lrs_finder)
    print(f'new LR = {LR:08f}')
    return LR
    


def optimize6(optimizer_type, parameters, closure, LRs, momentums, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    iterations_without_improvement = 0
    save_even_images=0

    if optimizer_type == 'SGD':
        lr_finder = 1e-4
        lrs_finder = []

        for _, j in tqdm(enumerate(range(num_iter))):
            
            # LR finder
            if lr_finder_flag:
                if j == 0: print('Finding LR')
                lr_finder = lr_finder * 1.2
                if lr_finder >= 1: #completed
                    return total_loss, images_generated, j_best, 
                lrs_finder.append(lr_finder)
                optimizer = torch.optim.SGD(parameters, lr=lr_finder)
            
            optimizer = torch.optim.SGD(parameters, lr=LRs[j], momentum=momentums[j])
            # Init values for best loss selection
            if j == 0:
                loss_best = 1000
                j_best = 0
                
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp.detach().cpu().numpy())
            
            # Restart if bad initialization
            if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
                """if the network is not learning (bad initialization) Restart"""
                restart = True
                return total_loss, images_generated, j_best, restart
            else:
                restart = False
               
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 500: #500 
                    #if 10000 % j == 0 or (j == num_iter -1): # added in inpainting v11 REMOVE: impeding last img to be saved
                    images_generated = image_generated_temp
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            #if annealing == True and iterations_without_improvement == 200:
                #LR = adjust_learning_rate(optimizer, j, LR)
                #print(f'LR reduced to: {LR:.5f}')
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 500:
                print(f'training stopped at it {j} after 500 iterations without improvement')
                break
    else:
        assert False
    return total_loss, images_generated, j_best, restart

def optimize5(optimizer_type, parameters, closure, closure_no_backprop, LR, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    iterations_without_improvement = 0

    # LR finder
    if lr_finder_flag:
                    
        lr_finder = 1e-8
        lrs_finder = []
        for j in range(num_iter):
            # Init values for best loss selection
            if j == 0:
                loss_best = 1000
                j_best = 0
            if j == 0: print('Finding LR')
            lr_finder = lr_finder * 1.2 # MOD 10-04-19
            if lr_finder >= 1e-4: #completed
                return total_loss, images_generated, j_best, 
            lrs_finder.append(lr_finder)
            optimizer = torch.optim.Adam(parameters, lr=lr_finder)
            
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure_no_backprop()
            total_loss.append(total_loss_temp.detach().cpu().numpy())
            
            # Restart if bad initialization
            if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
                """if the network is not learning (bad initialization) Restart"""
                restart = True
                return total_loss, images_generated, j_best, restart
            else:
                restart = False
            optimizer.step()
        return total_loss, images_generated, j_best, restart
        
    
    
    if optimizer_type == 'adam':
        #print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        if lr_finder_flag==False:
            for j in range(num_iter):

                # Init values for best loss selection
                if j == 0:
                    loss_best = 1000
                    j_best = 0

                optimizer.zero_grad()
                total_loss_temp, image_generated_temp = closure()
                total_loss.append(total_loss_temp.detach().cpu().numpy())

                # Restart if bad initialization
                if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
                    """if the network is not learning (bad initialization) Restart"""
                    restart = True
                    return total_loss, images_generated, j_best, restart
                else:
                    restart = False


                if total_loss_temp < loss_best:
                    loss_best = total_loss_temp
                    iterations_without_improvement = 0
                    j_best = j
                    if j > 500: 
                        #if 10000 % j == 0 or (j == num_iter -1): # added in inpainting v11 REMOVE: impeding last img to be saved
                            images_generated = image_generated_temp
                            #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
                if annealing == True and iterations_without_improvement == 300:
                    LR = adjust_learning_rate(optimizer, j, LR)
                    print(f'LR reduced to: {LR:.5f}')
                #print(f'total_loss_temp:{total_loss_temp}')
                optimizer.step()
                iterations_without_improvement += 1
                if iterations_without_improvement == 500:
                    print(f'training stopped at it {j} after 500 iterations without improvement')
                    break
    else:
        assert False
    return total_loss, images_generated, j_best, restart

def adjust_learning_rate(optimizer, epoch, LR):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    #LR = LR * (0.1 ** (epoch // 1000))
    LR = LR * 0.1 
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
    return LR

def optimize4_no_improvements_2000(optimizer_type, parameters, closure, LR, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    iterations_without_improvement = 0
    save_even_images=0

    if optimizer_type == 'adam':
        #print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        lr_finder = 1e-7
        lrs_finder = []
        for _, j in tqdm(enumerate(range(num_iter))):
            
            # LR finder
            if lr_finder_flag:
                if j == 0: print('Finding LR')
                lr_finder = lr_finder * 1.2
                if lr_finder >= 1e-2: #completed
                    return total_loss, images_generated, j_best, 
                lrs_finder.append(lr_finder)
                optimizer = torch.optim.Adam(parameters, lr=lr_finder)
            
            # Init values for best loss selection
            if j == 0:
                loss_best = 1000
                j_best = 0
                
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp.detach().cpu().numpy())
            
            # Restart if bad initialization
            if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
                """if the network is not learning (bad initialization) Restart"""
                restart = True
                return total_loss, images_generated, j_best, restart
            else:
                restart = False
               
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 500: #500 
                    #if 10000 % j == 0 or (j == num_iter -1): # added in inpainting v11 REMOVE: impeding last img to be saved
                    images_generated = image_generated_temp
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            if annealing == True and iterations_without_improvement == 200:
                LR = adjust_learning_rate(optimizer, j, LR)
                print(f'LR reduced to: {LR:.5f}\n')
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 2000:
                print(f'training stopped at it {j} after 2000 iterations without improvement\n')
                break
    else:
        assert False
    return total_loss, images_generated, j_best, restart