import numpy as np
from deepfool import deepfool
import torch
import torchvision.transforms as T

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def universal_perturbation(dataset, f, grads, delta=0.2, max_iter_uni = np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)

    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).

    :param grads: gradient functions with respect to input (as many gradients as classes).

    :param delta: controls the desired fooling rate (default = 80% fooling rate)

    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

    :param xi: controls the l_p magnitude of the perturbation (default = 10)

    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)

    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

    :param max_iter_df: maximum number of iterations for deepfool (default = 10)

    :return: the universal perturbation.
    """
# Compute the estimated labels in batches
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION
    v = 0
    fooling_rate = 0.0
    
    itr = 0
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)

        print ('Starting pass number ', itr)

        # Go through the data set and compute the perturbation increments sequentially
        for k in range(0, num_images):
            cur_img = dataset[k:(k+1), :, :, :]

            if int(np.argmax(np.array(f((T.ToTensor()(cur_img[0])).unsqueeze(dim=0).to(torch.float)).detach()).flatten())) == int(np.argmax(np.array(f((T.ToTensor()((cur_img+v)[0])).unsqueeze(dim=0).to(torch.float)).detach()).flatten())):
                print('>> k = ', k, ', pass #', itr)

                # Compute adversarial perturbation
                dr,iter,_,_ = deepfool(cur_img + v, f, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + v

        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))

        batch_size = 100
        num_batches = int(np.ceil(float(num_images) / float(batch_size)))

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            for cnt in range(m, M):
                est_labels_orig[cnt] = np.argmax(f((T.ToTensor()(dataset[cnt])).unsqueeze(dim=0).to(torch.float)).detach(), axis=1).flatten()
                est_labels_pert[cnt] = np.argmax(f((T.ToTensor()(dataset_perturbed[cnt])).unsqueeze(dim=0).to(torch.float)).detach(), axis=1).flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)

    return v
