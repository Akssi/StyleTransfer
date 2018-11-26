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
        # greyImage  = webp.open(self.greyImgsPath[idx])
        frame = self.transform(frame)
        # greyImage  = self.transform(greyImage)
        return frame


def normalizeImageTensor(img):
    # normalize using imagenet mean and std
    mean = img.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = img.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    img = img.div_(255.0)
    return (img - mean) / std

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | stylizedFrame')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--alpha', type=int, default=1e5)
    parser.add_argument('--beta', type=int, default=1e10)
    parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate, default=0.001')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--init', default='', help="path to reconet weights (to continue training/test)")
    parser.add_argument('--initIter',type=int, default=-1, help="path to reconet weights (to continue training/test)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--evalOutput', default='evalOutput', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--eval', action='store_true', help='run on test data')
    parser.add_argument('--v', action='store_true', help='print to console')
    parser.add_argument('--style', type=int, default=0)


    # Holds console output
    logs = []
    with open('log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow('New experiment')

    opt = parser.parse_args()
    # print(opt)

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

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    transform = transforms.Compose([
                transforms.Resize((360,360)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255)),
                ])
    style_transform = transforms.Compose([
                transforms.Resize((360,360)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (1-x).mul(255)),
                ])

    dataset = VideoFrameDataset(opt.dataroot, transform, opt.eval)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize)

    alpha = opt.alpha
    beta = opt.beta
    gamma = 0

    reconet = ReCoNet().to(device)

    # setup optimizer
    optimizer = optim.Adam(reconet.parameters(), lr=opt.lr)
    criterionL2 = nn.MSELoss().to(device)

    lossNetwork = Vgg16().to(device)

    for param in lossNetwork.parameters():
      param.requires_grad = False

    # reconet.apply(weights_init)
    if opt.init != '':
            reconet.load_state_dict(torch.load(opt.init))
            initDone = False

    style_names = ('autoportrait', 'candy', 'composition', 'edtaonisl', 'udnie')
    style_model_path = 'models/weights/'
    style_img_path = 'models/style/'+style_names[min(max(opt.style,0),4)]


    styleRef = transform(Image.open(style_img_path+'.jpg'))
    onlineWriter.add_image('Input/StyleRef', style_transform(Image.open(style_img_path+'.jpg')))
    styleRef = styleRef.unsqueeze(0).expand(opt.batchSize, 3, 360, 360).to(device)

    # Get style feature maps from VGG-16 layers relu1_2, relu2_2, relu3_3, relu4_3
    styleRef_features = lossNetwork(normalizeImageTensor(styleRef))
    styleRef_gram = [gram_matrix(feature) for feature in styleRef_features]

    onlineWriter.add_scalar('Input/Alpha (Content loss)', alpha)
    onlineWriter.add_scalar('Input/Beta (Style loss)', beta)
    onlineWriter.add_scalar('Input/Gamma (TV loss)', gamma)
    onlineWriter.add_scalar('Input/Learning Rate', opt.lr, 0)

    # -----------------------------------------------------------------------------------
    # Run training
    # -----------------------------------------------------------------------------------
    if not opt.eval:
    
        if opt.init != '' and opt.initIter == -1:
            raise ValueError("--initIter undefined can't load checkpoint")

        for epoch in range(opt.niter):
            for i, frame in enumerate(dataloader):
                if i < opt.initIter and (not initDone):
                    i = opt.initIter
                    continue
                if i % 1000 == 0:
                    opt.lr = max(opt.lr/1.2, 1e-3)
                initDone = True
                onlineWriter.add_scalar('Input/Learning Rate', opt.lr, i)

                optimizer.zero_grad()
                
                ############################
                # (1) Generate Stylized Frame & Calculate Losses
                ###########################
                # frame.copy_(img)
                # frame = Variable(frame)
                frame = Variable(frame.data.to(device))

                # Generate stylizd frame using ReCoNet
                stylizedFrame = reconet(frame)
                
                # totalDivergenceLoss = torch.sum(torch.abs(stylizedFrame[:,:,:,:-1] - stylizedFrame[:,:,:,1:])) \
                #     + torch.sum(torch.abs(stylizedFrame[:,:,:-1,:] - stylizedFrame[:,:,1:,:]))
                
                stylizedFrame_norm = normalizeImageTensor(stylizedFrame)
                frame_norm = normalizeImageTensor(frame)

                # Get features maps from VGG-16 network for the stylized and actual frame
                stylizedFrame_features = lossNetwork(stylizedFrame_norm)
                frame_features = lossNetwork(frame_norm)
                
                # Calculate content loss using layer relu3_3 feature map from VGG-16
                contentLoss = criterionL2(stylizedFrame_features[2], frame_features[2].expand_as(stylizedFrame_features[1]))
                contentLoss *= alpha
                # Sum style loss on all feature maps
                styleLoss = 0.
                for feature, refFeature in zip(stylizedFrame_features, styleRef_gram):
                    gramFeature = gram_matrix(feature)
                    styleLoss += criterionL2(gramFeature, refFeature.expand_as(gramFeature))
                styleLoss *= beta

                # Final loss
                loss = contentLoss + styleLoss 
                    # + gamma * totalDivergenceLoss
                
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
                # onlineWriter.add_scalar('Loss/Current Iter/TVLoss', totalDivergenceLoss, i)
                onlineWriter.add_scalar('Loss/Current Iter/FinalLoss', loss, i)

                if opt.v:
                    # Write to console
                    print('[%d/%d][%d/%d] Style Loss: %.4f Content Loss: %.4f'
                        % (epoch, opt.niter, i, len(dataloader),
                            styleLoss, contentLoss))

                # with open('log.csv', 'a') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([epoch, i, styleLoss, contentLoss])
                if (i+1) % 1500 == 0:
                    torch.save(reconet.state_dict(), '%s/reconet_epoch_%d.pth' % (opt.outf, i))
                    vutils.save_image(stylizedFrame.data,
                            '%s/stylizedFrame_samples_batch_%03d.png' % (style_model_path, i))
                    onlineWriter.add_image('Output/Current Iter/Frame', (frame), i)
                    onlineWriter.add_image('Output/Current Iter/StylizedFrame', (stylizedFrame), i)

            # Write to online logs
            onlineWriter.add_scalar('Loss/ContentLoss', contentLoss, epoch)
            onlineWriter.add_scalar('Loss/StyleLoss', styleLoss, epoch)
            # onlineWriter.add_scalar('Loss/TVLoss', totalDivergenceLoss, epoch)
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
            stylizedFrame = reconet(frame)
            
            # totalDivergenceLoss = torch.sum(torch.abs(stylizedFrame[:,:,:,:-1] - stylizedFrame[:,:,:,1:])) \
            #     + torch.sum(torch.abs(stylizedFrame[:,:,:-1,:] - stylizedFrame[:,:,1:,:]))
            
            stylizedFrame_norm = normalizeImageTensor(stylizedFrame)
            frame_norm = normalizeImageTensor(frame)

            # Get features maps from VGG-16 network for the stylized and actual frame
            stylizedFrame_features = lossNetwork(stylizedFrame_norm)
            frame_features = lossNetwork(frame_norm)
            
            # Calculate content loss using layer relu3_3 feature map from VGG-16
            contentLoss = criterionL2(stylizedFrame_features[1], frame_features[1])
            contentLoss *= alpha
            # Sum style loss on all feature maps
            styleLoss = 0.
            for feature, refFeature in zip(stylizedFrame_features, styleRef_gram):
                gramFeature = gram_matrix(feature)
                styleLoss += criterionL2(gramFeature, refFeature)
            styleLoss *= beta

            # Final loss
            loss = contentLoss + styleLoss 
                # + gamma * totalDivergenceLoss
                        
            ############################
            # (3) Log and do checkpointing
            ###########################
            # onlineWriter.add_scalar('Loss/Current Iter/ContentLoss', contentLoss, i)
            # onlineWriter.add_scalar('Loss/Current Iter/StyleLoss', styleLoss, i)
            # onlineWriter.add_scalar('Loss/Current Iter/TVLoss', totalDivergenceLoss, i)
            # onlineWriter.add_scalar('Loss/Current Iter/FinalLoss', loss, i)

            # # Write to console
            # print('[%d/%d][%d/%d] Style Loss: %.4f Content Loss: %.4f'
            #     % (1, 1, i, len(dataloader),
            #         styleLoss, contentLoss))

            vutils.save_image(stylizedFrame.data,
                    '%s/stylizedFrame_%03d.png' % (opt.evalOutput, i))
       
        
if __name__ == "__main__":
    main()