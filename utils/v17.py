def optimize_v17(optimizer_type, parameters, closure, LR, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """
    # 1. we start saving after the 1500 iteration because
    #    the first 1500 iterations are blurry (and to avoid memory problems)

    Runs optimization loop.

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
                if j > 1500: # the first 1500 iterations are blurry
                    images_generated.append(image_generated_temp)
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            if annealing == True and iterations_without_improvement == 200:
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
