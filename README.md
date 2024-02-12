# Python

A small practice for FGSM and PGD attacks against simple pretrained resnet18 model
(models can be simply modified by changing the variable "model")

# Single picture

python run.py --i [index] 

to run the specific picture
the code will output the perbutation and the perbuted picture
with the L1, L2, Linf norm

# Imagenet1k dataset

python run.py

The code will output acc, racc, asr, psr, sasr

# Universal perbutation

python run.py -M ‘UAP’

The code will output acc, racc, asr, psr, sasr
with the universal perbutation for the whole dataset(imagenet1k)

