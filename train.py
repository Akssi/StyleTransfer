from __future__ import print_function
import argparse
import os
import os.path
import sys
import time
import re
import random
import csv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import webp
from tensorboardX import SummaryWriter
from network import *

class VideoFrameDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transforms, eval=False):
        """
        Args:
            root_dir (string): Directory with all the frames.
        """
        self.videoFramesPath = []
        self.transform = transforms
        if eval:
            root_dir = os.path.join(root_dir, "Test")
        else:
            root_dir = os.path.join(root_dir, "Train")

        dir = os.path.expanduser(root_dir)

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                self.videoFramesPath.append(os.path.join(root, fname))

        print("len(self.videoFramesPath) : ", len(self.videoFramesPath))

    def __len__(self):
        return len(self.videoFramesPath)

    def __getitem__(self, idx):
        if self.videoFramesPath[idx].endswith(".webp"):
            frame = webp.load_image(self.videoFramesPath[idx], "RGB")
        else:
            frame = Image.open(self.videoFramesPath[idx]).convert('RGB')
        frame = self.transform(frame)
        return frame

#region Utility 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# From https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def normalizeImageTensor(img):
    # normalize using imagenet mean and std
    mean = img.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = img.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    img = img.div_(255.0)
    return (img - mean) / std

#endregion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=360, help='the height / width of the input image to network')
    parser.add_argument('--alpha', type=int, default=1e5)
    parser.add_argument('--beta', type=int, default=1e10)
    parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate, default=0.001')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--init', default='', help="path to reconet weights (to continue training/test)")
    parser.add_argument('--initIter',type=int, default=-1, help="path to reconet weights (to continue training/test)")
    parser.add_argument('--evalOutput', default='evalOutput', help='folder to output images and model checkpoints')
    parser.add_argument('--eval', action='store_true', help='run on test data')
    parser.add_argument('--v', action='store_true', help='print to console')
    parser.add_argument('--style', type=int, default=0)

    opt = parser.parse_args()

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    onlineWriter = SummaryWriter()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.cuda:
        device = 'cuda'
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'

    seed = random.randint(1, 10000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(seed)

    # Setup image transforms
    transform = transforms.Compose([
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255)),
                ])
    style_transform = transforms.Compose([
                transforms.Resize((opt.imageSize,opt.imageSize)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (1-x).mul(255)),
                ])

    # Setup image & style dataset
    dataset = VideoFrameDataset(opt.dataroot, transform, opt.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize)

    # Loss factors
    alpha = opt.alpha
    beta = opt.beta
    gamma = 0

    # Setup networks, optimizer, losses
    reconet = ReCoNet().to(device)
    optimizer = optim.Adam(reconet.parameters(), lr=opt.lr)
    criterionL2 = nn.MSELoss().to(device)
    lossNetwork = Vgg16().to(device)

    for param in lossNetwork.parameters():
        param.requires_grad = False

    if opt.init != '':
        reconet.load_state_dict(torch.load(opt.init))
        initDone = False

    # -----------------------------------------------------------------------------------
    # Run training
    # -----------------------------------------------------------------------------------
    if not opt.eval:

        # Load style reference image
        style_names = ('autoportrait', 'candy', 'composition', 'mosaic', 'udnie', 'color')
        style_model_path = 'reconet/models/weights/'
        style_img_path = 'reconet/models/style/'+style_names[min(max(opt.style,0),5)]

        styleRef = transform(Image.open(style_img_path+'.jpg'))
        onlineWriter.add_image('Input/StyleRef', style_transform(Image.open(style_img_path+'.jpg')))
        styleRef = styleRef.unsqueeze(0).expand(1, 3, opt.imageSize, opt.imageSize).to(device)

        # Get style feature maps from VGG-16 layers relu1_2, relu2_2, relu3_3, relu4_3
        styleRef_features = lossNetwork(normalizeImageTensor(styleRef))
        styleRef_gram = [gram_matrix(feature) for feature in styleRef_features]

        # Log initial parameters value
        onlineWriter.add_scalar('Input/Alpha (Content loss)', alpha)
        onlineWriter.add_scalar('Input/Beta (Style loss)', beta)
        onlineWriter.add_scalar('Input/Learning Rate', opt.lr, 0)

    # -----------------------------------------------------------------------------------
    # Run training
    # -----------------------------------------------------------------------------------
    
        if opt.init != '' and opt.initIter == -1:
            raise ValueError("--initIter undefined can't load checkpoint")

        for epoch in range(opt.niter):
            for i, frame in enumerate(dataloader):

                # Skip until i == initIter if specified
                if i < opt.initIter and (not initDone):
                    i = opt.initIter
                    continue
                initDone = True

                # Learning rate annealing
                if i % 1000 == 0:
                    opt.lr = max(opt.lr/1.2, 1e-3)

                onlineWriter.add_scalar('Input/Learning Rate', opt.lr, i)

                # Reset optimizer
                optimizer.zero_grad()
                
                ############################
                # (1) Generate Stylized Frame & Calculate Losses
                ###########################
                frame = Variable(frame.data.to(device))

                # Generate stylizd frame using ReCoNet
                _, stylizedFrame = reconet(frame)
                
                stylizedFrame_norm = normalizeImageTensor(stylizedFrame)
                frame_norm = normalizeImageTensor(frame)

                # Get features maps from VGG-16 network for the stylized and actual frame
                stylizedFrame_features = lossNetwork(stylizedFrame_norm)
                frame_features = lossNetwork(frame_norm)
                
                # Calculate content loss using layer relu3_3 feature map from VGG-16
                contentLoss = criterionL2(stylizedFrame_features[1], frame_features[1].expand_as(stylizedFrame_features[1]))
                contentLoss *= alpha
                
                # Sum style loss on all VGG-16 feature maps (relu1_2, relu2_2, relu3_3, relu4_3)
                styleLoss = 0.
                for feature, refFeature in zip(stylizedFrame_features, styleRef_gram):
                    gramFeature = gram_matrix(feature)
                    styleLoss += criterionL2(gramFeature, refFeature.expand_as(gramFeature))
                styleLoss *= beta

                # Final loss
                loss = contentLoss + styleLoss 
                
                ############################
                # (2) Backpropagate and optimize network weights
                ###########################
                loss.backward()
                optimizer.step()
                
                ############################
                # (3) Log and do checkpointing
                ###########################
                onlineWriter.add_scalar('Loss/Current Iter/ContentLoss', contentLoss, i)
                onlineWriter.add_scalar('Loss/Current Iter/StyleLoss', styleLoss, i)
                onlineWriter.add_scalar('Loss/Current Iter/FinalLoss', loss, i)

                if opt.v:
                    # Write to console
                    print('[%d/%d][%d/%d] Style Loss: %.4f Content Loss: %.4f'
                        % (epoch, opt.niter, i, len(dataloader),
                            styleLoss, contentLoss))

                if (i+1) % 1500 == 0:
                    torch.save(reconet.state_dict(), '%s/reconet_epoch_%d.pth' % ("runs/output/batch", i))
                    vutils.save_image(stylizedFrame.data,
                            '%s/stylizedFrame_samples_batch_%03d.png' % ("runs/output/batch", i))
                    onlineWriter.add_image('Output/Current Iter/Frame', (frame), i)
                    onlineWriter.add_image('Output/Current Iter/StylizedFrame', (stylizedFrame), i)

            # Write to online logs
            onlineWriter.add_scalar('Loss/ContentLoss', contentLoss, epoch)
            onlineWriter.add_scalar('Loss/StyleLoss', styleLoss, epoch)
            onlineWriter.add_scalar('Loss/FinalLoss', loss, epoch)
            onlineWriter.add_image('Output/Frame', frame, epoch)
            onlineWriter.add_image('Output/StylizedFrame', stylizedFrame, epoch)
            # do checkpointing
            torch.save(reconet.state_dict(), '%s/reconet_epoch_%d.pth' % (style_model_path, epoch))

        onlineWriter.close()
    # -----------------------------------------------------------------------------------
    # Run test
    # -----------------------------------------------------------------------------------
    else:
        reconet.eval()


        for i, frame in enumerate(dataloader):
            
            ############################
            # (1) Generate Stylized Frame & Calculate Losses
            ###########################
            frame = frame.to(device)

            # Generate stylizd frame using ReCoNet
            _, stylizedFrame = reconet(frame)

            stylizedFrame_norm = normalizeImageTensor(stylizedFrame)
            frame_norm = normalizeImageTensor(frame)

            vutils.save_image(stylizedFrame.data,
                    '%s/stylizedFrame_%03d.png' % (opt.evalOutput, i))
            onlineWriter.add_image('Output/Current Iter/Frame', (frame), i)
            onlineWriter.add_image('Output/Current Iter/StylizedFrame', (stylizedFrame), i)

        onlineWriter.close()
        
if __name__ == "__main__":
    main()