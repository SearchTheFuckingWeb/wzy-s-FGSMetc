import numpy as np
import torch
import torchvision.transforms as T

def deepfool(image, f, grads, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = np.array(f((T.ToTensor()(image[0])).unsqueeze(dim=0).to(torch.float)).detach()).flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1].copy()

    I = I[0:num_classes]
    label = I[0]
    la = torch.tensor([label])

    input_shape = image.shape
    pert_image = image

    f_i = np.array(f((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float)).detach()).flatten()
    k_i = int(np.argmax(f_i))

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        gradients = np.empty([num_classes, 1, 1, 1, 1], dtype=float)
        gradients = np.asarray([grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),la), 
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[1]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[2]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[3]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[4]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[5]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[6]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[7]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[8]])),
                                grads((T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float),torch.tensor([I[9]])),])

        for k in range(1, num_classes):

            # set new w_k and new f_k
            w_k = gradients[k, :, :, :, :] - gradients[0, :, :, :, :]
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i.transpose(0, 2, 3, 1)

        # compute new perturbed image
        pert_image = image + (1+overshoot)*r_tot
        loop_i += 1

        # compute new label
        pim = (T.ToTensor()(pert_image[0])).unsqueeze(dim=0).to(torch.float)
        f_i = np.array(f(pim).detach()).flatten()
        k_i = int(np.argmax(f_i))

    r_tot = (1+overshoot)*r_tot
    return r_tot, loop_i, k_i, pert_image
