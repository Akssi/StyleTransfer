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
import torch.functional as F
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
import matplotlib.pyplot as plt
import flowlib

from network import *

class VideoFrameDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, flow_dir, occlusion_dir, frameTransforms, eval=False):
        """
        Args:
            root_dir (string): Directory with all the frames.
        """
        self.videoFramesPath = []
        self.transform = frameTransforms
        self.occTransform = transforms.Compose([
                            transforms.Resize((360,640)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: (1-x)),
                            ])
        self.flow_dir = flow_dir
        self.occlusion_dir = occlusion_dir

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
        return len(self.videoFramesPath)-2

    def __getitem__(self, idx):
        if self.videoFramesPath[idx].endswith(".webp"):
            frame = webp.load_image(self.videoFramesPath[idx], "RGB")
        else:
            frame = Image.open(self.videoFramesPath[idx]).convert('RGB')
        frame = self.transform(frame)
        flow = flowlib.readFlow(os.path.join(self.flow_dir, "{:06d}.flo".format(idx)))
        # flow = Image.open(os.path.join(self.flow_dir, "{:05d}.png".format(idx))).convert('RGB')
        # flow = self.transform(flow)
        flow = transforms.ToTensor()(flow)
        occ = self.occTransform(Image.open(os.path.join(self.occlusion_dir, "{:05d}.png".format(idx))).convert('RGB'))
        return (frame, flow, occ)

#region Utility 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Normalize using imagenet mean and std
def normalizeImageTensor(img):
    mean = img.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = img.new_tensor(IMAGENET_STD).view(-1, 1, 1)

    img = img.div_(255.0)
    return (img - mean) / std

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

# From https://github.com/sniklaus/pytorch-spynet/blob/master/run.py
def warpFrame(frame, flow, cuda=False):
    tensorHorizontal = torch.linspace(-1.0, 1.0, flow.size(3)) \
        .view(1, 1, 1, flow.size(3)) \
        .expand(flow.size(0), -1, flow.size(2), -1)
    tensorVertical = torch.linspace(-1.0, 1.0, flow.size(2)) \
        .view(1, 1, flow.size(2), 1) \
        .expand(flow.size(0), -1, -1, flow.size(3))

    tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1)
    if cuda:
        tensorGrid = tensorGrid.cuda()
    flow = torch.cat([
                    flow[:, 0:1, :, :] / ((frame.size(3) - 1.0) / 2.0),
                    flow[:, 1:2, :, :] / ((frame.size(2) - 1.0) / 2.0)
                    ], 1)

    return torch.nn.functional.grid_sample(input=frame, grid=(tensorGrid + flow).permute(0, 2, 3, 1), \
        mode='bilinear', padding_mode='border')
# From https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py#L139
def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask
#endregion

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | stylizedFrame')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--occlusionDir', required=True, help='path to occlusion directory')
    parser.add_argument('--flowDir', required=True, help='path to flow directory')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--alpha', type=int, default=1e5)
    parser.add_argument('--beta', type=int, default=1e10)
    parser.add_argument('--gamma', type=int, default=0)
    parser.add_argument('--niter', type=int, default=5, help='number of epochs to train for')
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

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    colorTransform = transforms.Lambda(lambda x: x.mul(255))
    transform = transforms.Compose([
                transforms.Resize((360,640)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255)),
                ])
    style_transform = transforms.Compose([
                transforms.Resize((360,640)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (1-x).mul(255)),
                ])

    dataset = VideoFrameDataset(opt.dataroot, opt.flowDir, opt.occlusionDir, transform, opt.eval)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize)

    alpha = opt.alpha
    beta = opt.beta
    gamma = opt.gamma
    lambdaOutput = 2e3
    lambdaFeatureMap = 1e7

    reconet = ReCoNet().to(device)

    # setup optimizer
    optimizer = optim.Adam(reconet.parameters(), lr=opt.lr)
    criterionL2 = nn.MSELoss().to(device)
    noReduceCriterionL2 = nn.MSELoss(reduction= "none").to(device)
    lossNetwork = Vgg16().to(device)

    for param in lossNetwork.parameters():
      param.requires_grad = False

    # reconet.apply(weights_init)
    if opt.init != '':
            reconet.load_state_dict(torch.load(opt.init))
            initDone = False

    if not opt.eval:
        style_names = ('autoportrait', 'candy', 'composition', 'mosaic', 'udnie', 'color')
        style_model_path = 'reconet/models/weights/'
        style_img_path = 'reconet/models/style/'+style_names[min(max(opt.style,0),5)]


        styleRef = transform(Image.open(style_img_path+'.jpg'))
        onlineWriter.add_image('Input/StyleRef', style_transform(Image.open(style_img_path+'.jpg')))
        styleRef = styleRef.unsqueeze(0).expand(opt.batchSize, 3, 360, 640).to(device)

        # Get style feature maps from VGG-16 layers relu1_2, relu2_2, relu3_3, relu4_3
        styleRef_features = lossNetwork(normalizeImageTensor(styleRef))
        styleRef_gram = [gram_matrix(feature) for feature in styleRef_features]

        onlineWriter.add_scalar('Input/Alpha (Content loss)', alpha)
        onlineWriter.add_scalar('Input/Beta (Style loss)', beta)
        onlineWriter.add_scalar('Input/Gamma (TV loss)', gamma)
        onlineWriter.add_scalar('Input/Lambda Output (Temporal loss)', lambdaOutput)
        onlineWriter.add_scalar('Input/Lambda Feature (Temporal loss)', lambdaFeatureMap)
        onlineWriter.add_scalar('Input/Learning Rate', opt.lr, 0)

        # -----------------------------------------------------------------------------------
        # Run training
        # -----------------------------------------------------------------------------------

        if opt.init != '' and opt.initIter == -1:
            raise ValueError("--initIter undefined can't load checkpoint")

        for epoch in range(opt.niter):
            for i, (frame, flow, occ) in enumerate(dataloader):
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
                frame = Variable(frame.data.to(device))
                occ = Variable(occ.data.to(device))
                flow = torch.nn.functional.interpolate(flow, size=(frame.shape[2], frame.shape[3]), mode='bilinear')
                flow = Variable(flow.data.to(device))

                # Generate stylizd frame using ReCoNet
                featureMap, stylizedFrame = reconet(frame)

                stylizedFrame_norm = normalizeImageTensor(stylizedFrame)
                frame_norm = normalizeImageTensor(frame)

                # Get features maps from VGG-16 network for the stylized and actual frame
                stylizedFrame_features = lossNetwork(stylizedFrame_norm)
                frame_features = lossNetwork(frame_norm)

                featureMapFlow = torch.nn.functional.interpolate(flow, size=(featureMap.shape[2],featureMap.shape[3]), mode='bilinear', align_corners = False)
                featureMapOcc = torch.nn.functional.interpolate(occ, size=(featureMap.shape[2],featureMap.shape[3]), mode='bilinear', align_corners = False)
                warpedFeatureMap = warp(featureMap, featureMapFlow)

                featureTemporalLoss = noReduceCriterionL2(featureMap[1], warpedFeatureMap[0])
                featureTemporalLoss = (1/(featureMap.shape[1]*featureMap.shape[2]*featureMap.shape[3])) * (featureMapOcc * featureTemporalLoss)
                featureTemporalLoss = torch.mean(featureTemporalLoss)
                featureTemporalLoss *= lambdaFeatureMap

                warpedStylizedFrame = warp(stylizedFrame[0].unsqueeze(0), flow[0].unsqueeze(0))
                warpedFrame = warpFrame(frame, flow)

                inputTerm = (frame[1] - warpedFrame[0])
                inputTerm = 0.2126 * inputTerm[0,:,:] + 0.7152 * inputTerm[1,:,:] + 0.0722 * inputTerm[2,:,:]
                outputTerm = (stylizedFrame[1] - warpedStylizedFrame[0])
                inputTerm = inputTerm.expand_as(outputTerm)

                outputTemporalLoss = (1/(frame.shape[1]*frame.shape[2]*frame.shape[3])) * (occ * noReduceCriterionL2(outputTerm, inputTerm))
                outputTemporalLoss = torch.mean(outputTemporalLoss)
                outputTemporalLoss *= lambdaOutput

                # totalDivergenceLoss = gamma * (torch.sum(torch.abs(stylizedFrame_norm[:, :, :, :-1] - stylizedFrame_norm[:, :, :, 1:])) \
                #     + torch.sum(torch.abs(stylizedFrame_norm[:, :, :-1, :] - stylizedFrame_norm[:, :, 1:, :])))
                # Calculate content loss using layer relu3_3 feature map from VGG-16
                contentLoss = criterionL2(stylizedFrame_features[2], frame_features[2].expand_as(stylizedFrame_features[2]))

                contentLoss *= alpha
                # Sum style loss on all feature maps
                styleLoss = 0.
                for feature, refFeature in zip(stylizedFrame_features, styleRef_features):
                    # gramFeature = gram_matrix(feature)
                    styleLoss += criterionL2(feature, refFeature.expand_as(feature))
                styleLoss *= beta

                # Final loss
                loss = outputTemporalLoss + featureTemporalLoss + contentLoss + styleLoss

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
                onlineWriter.add_scalar('Loss/Current Iter/OutputTemporalLoss', outputTemporalLoss, i)
                onlineWriter.add_scalar('Loss/Current Iter/FeatureTemporalLoss', featureTemporalLoss, i)
                onlineWriter.add_scalar('Loss/Current Iter/FinalLoss', loss, i)

                if opt.v:
                    # Write to console
                    print('[%d/%d][%d/%d] Style Loss: %.4f Content Loss: %.4f'
                        % (epoch, opt.niter, i, len(dataloader),
                            styleLoss, contentLoss))

                if (i+1) % 100 == 0:
                    torch.save(reconet.state_dict(), '%s/reconet_epoch_%d.pth' % ("runs/output/batch", i))
                    vutils.save_image(stylizedFrame.data,
                            '%s/stylizedFrame_samples_batch_%03d.png' % ("runs/output/batch", i))
                    # onlineWriter.add_image('Output/Current Iter/Frame', colorTransform(frame), i)
                    # onlineWriter.add_image('Output/Current Iter/WarpedFrame', colorTransform(warpedFrame), i)
                    # onlineWriter.add_image('Output/Current Iter/StylizedFrame', colorTransform(stylizedFrame), i)
                    onlineWriter.add_image('Output/Current Iter/StylizedFrame', (stylizedFrame), i)
                    onlineWriter.add_image('Output/Current Iter/Frame', (occ*frame), i)
                    onlineWriter.add_image('Output/Current Iter/WarpedFrame', (occ*warpedFrame), i)
                    onlineWriter.add_image('Output/Current Iter/OcclusionMask', (occ), i)

            # Write to online logs
            onlineWriter.add_scalar('Loss/ContentLoss', contentLoss, epoch)
            onlineWriter.add_scalar('Loss/StyleLoss', styleLoss, epoch)
            onlineWriter.add_scalar('Loss/Current Iter/OutputTemporalLoss', outputTemporalLoss, epoch)
            onlineWriter.add_scalar('Loss/Current Iter/FeatureTemporalLoss', featureTemporalLoss, epoch)
            onlineWriter.add_scalar('Loss/FinalLoss', loss, epoch)
            # onlineWriter.add_image('Output/Frame', colorTransform(frame), epoch)
            # onlineWriter.add_image('Output/StylizedFrame', colorTransform(stylizedFrame), epoch)
            onlineWriter.add_image('Output/Frame', (frame), epoch)
            onlineWriter.add_image('Output/StylizedFrame', (stylizedFrame), epoch)
            # do checkpointing
            torch.save(reconet.state_dict(), '%s/reconet_epoch_%d.pth' % ("runs/output", epoch))

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

            # totalDivergenceLoss = torch.sum(torch.abs(stylizedFrame[:,:,:,:-1] - stylizedFrame[:,:,:,1:])) \
            #     + torch.sum(torch.abs(stylizedFrame[:,:,:-1,:] - stylizedFrame[:,:,1:,:]))

            stylizedFrame_norm = normalizeImageTensor(stylizedFrame)
            frame_norm = normalizeImageTensor(frame)

            # # Get features maps from VGG-16 network for the stylized and actual frame
            # stylizedFrame_features = lossNetwork(stylizedFrame_norm)
            # frame_features = lossNetwork(frame_norm)

            # # Calculate content loss using layer relu3_3 feature map from VGG-16
            # contentLoss = criterionL2(stylizedFrame_features[1], frame_features[1])
            # contentLoss *= alpha
            # # Sum style loss on all feature maps
            # styleLoss = 0.
            # for feature, refFeature in zip(stylizedFrame_features, styleRef_gram):
            #     gramFeature = gram_matrix(feature)
            #     styleLoss += criterionL2(gramFeature, refFeature)
            # styleLoss *= beta

            # Final loss
            # loss = contentLoss + styleLoss
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
            vutils.save_image(frame.data,
                    '%s/frame_%03d.png' % (opt.evalOutput, i))

    onlineWriter.close()


if __name__ == "__main__":
    main()