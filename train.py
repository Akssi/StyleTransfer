from __future__ import print_function
import argparse
import os
import os.path
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

class ColorBWDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, color_dir="color", grey_dir="grey"):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.greyImgsPath = []
        self.colorImgsPath = []
        #self.transform = transforms.ToTensor()
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        dir = os.path.expanduser(root_dir)
        colorDir = os.path.join(dir, color_dir)
        greyDir = os.path.join(dir, grey_dir)

        for root, _, fnames in sorted(os.walk(colorDir)):
            for fname in sorted(fnames):
                self.colorImgsPath.append(os.path.join(colorDir, fname))

        for root, _, fnames in sorted(os.walk(greyDir)):
            for fname in sorted(fnames):
                self.greyImgsPath.append(os.path.join(greyDir, fname))

        print("len(self.colorImgsPath) : ", len(self.colorImgsPath))
        print("len(self.greyImgsPath) : ", len(self.greyImgsPath))

    def __len__(self):
        return len(self.colorImgsPath)

    def __getitem__(self, idx):
        colorImage = Image.open(self.colorImgsPath[idx]).convert('RGB')
        greyImage  = Image.open(self.greyImgsPath[idx])
        colorImage = self.transform(colorImage)
        greyImage  = self.transform(greyImage)
        return (colorImage, greyImage)

class VideoFrameDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, eval=False, video_dir="A"):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.videoFramesPath = []
        #self.transform = transforms.ToTensor()
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        self.transform = transforms.Compose([
            transforms.Resize((360,640)),
            transforms.ToTensor()])

        dir = os.path.expanduser(root_dir)
        videoDir = os.path.join(
            os.path.join(dir, "TRAIN" if not eval else "TEST"),
            video_dir
        )

        for root, sub_folders, _ in sorted(os.walk(videoDir)):
            for folder in sub_folders:
                for sub_root, _,fnames  in sorted(os.walk(os.path.join(os.path.join(root, folder), "left"))):
                    if len(fnames)%2 == 0:
                        for fname in sorted(fnames):
                            self.videoFramesPath.append(os.path.join(sub_root, fname))
                for sub_root, _,fnames  in sorted(os.walk(os.path.join(os.path.join(root, folder), "right"))):
                    if len(fnames)%2 == 0:
                        for fname in sorted(fnames):
                            self.videoFramesPath.append(os.path.join(sub_root, fname))


        print("len(self.videoFramesPath) : ", len(self.videoFramesPath))
    def __len__(self):
        return len(self.videoFramesPath)

    def __getitem__(self, idx):
        frame = webp.load_image(self.videoFramesPath[idx], "RGB")
        # greyImage  = webp.open(self.greyImgsPath[idx])
        frame = self.transform(frame)
        # greyImage  = self.transform(greyImage)
        return frame

def main():
    onlineWriter = SummaryWriter('Runs')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | stylizedFrame')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--reconet', default='', help="path to reconet (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--eval', action='store_true', help='run on test data')

    # Holds console output
    logs = []
    with open('log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow('New experiment')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    dataset = VideoFrameDataset(opt.dataroot,opt.eval)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = 1
    ngf = int(opt.ngf)
    ndf = 64
    nc = 3
    alpha = 10
    beta = 1
    gamma = 0.001


    # custom weights initialization
    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         m.weight.data.normal_(0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         m.weight.data.normal_(1.0, 0.02)
    #         m.bias.data.fill_(0)

    reconet = ReCoNet()
    lossNetwork = Vgg16().eval()
    for param in lossNetwork.parameters():
      param.requires_grad = False
    # reconet.apply(weights_init)
    if opt.reconet != '':
        reconet.load_state_dict(torch.load(opt.reconet))
    # print(reconet)
    # print(lossNetwork)

    criterionL1 = nn.L1Loss()
    criterionL2 = nn.MSELoss()

    frame = torch.FloatTensor(opt.batchSize, 3, 360, 640)

    if opt.cuda:
        reconet.cuda()
        lossNetwork.cuda()
        criterionL1.cuda()
        criterionL2.cuda()
        frame = frame.cuda()

    style_names = ('autoportrait', 'candy', 'composition', 'edtaonisl', 'udnie')
    style_model_path = 'models/weights/'
    style_img_path = 'models/style/'
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    imageTransform = transforms.Compose([
                transforms.Resize((360,640)),
                transforms.ToTensor()])
                
    styleImg = imageTransform(Image.open(style_img_path+style_names[0]+'.jpg').convert('RGB')).unsqueeze(0)
    onlineWriter.add_image('Input/StyleRef', styleImg)
    onlineWriter.add_scalar('Input/Alpha (Content loss)', alpha)
    onlineWriter.add_scalar('Input/Beta (Style loss)', beta)
    onlineWriter.add_scalar('Input/Gamma (TV loss)', gamma)
    onlineWriter.add_scalar('Input/Learning Rate', opt.lr)

    # -----------------------------------------------------------------------------------
    # Run training
    # -----------------------------------------------------------------------------------
    if not opt.eval:
        # setup optimizer
        optimizer = optim.Adam(reconet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
        if opt.cuda:
            styleImg = styleImg.cuda()

        with torch.no_grad():
            # Get style feature maps from VGG-16 layers relu1_2, relu2_2, relu3_3, relu4_3
            styleRefFeatures = lossNetwork.forward(styleImg)

        for epoch in range(opt.niter):
            if epoch > 1:
                alpha = 1
                beta = 10
            for i, img in enumerate(dataloader, 0):

                
                ############################
                # (1) Generate Stylized Frame & Calculate Losses
                ###########################

                frame.copy_(img)

                # Generate stylizd frame using ReCoNet
                stylizedFrame = reconet.forward(Variable(frame))

                # Get features maps from VGG-16 network for the stylized and actual frame
                stylizedFrameFeatures = lossNetwork.forward(stylizedFrame)
                contentRefFeature = lossNetwork.forward(frame.detach(), layer=2)

                # Sum style loss on all feature maps
                styleLoss = 0
                for refFeature, feature in zip(styleRefFeatures, stylizedFrameFeatures):
                    refFeature = gram_matrix(refFeature)
                    feature = gram_matrix(feature)
                    styleLoss += criterionL2(refFeature, feature)

                # Calculate content loss using layer relu3_3 feature map from VGG-16
                contentLoss = criterionL2(contentRefFeature, stylizedFrameFeatures[2])
                
                totalDivergenceLoss = torch.sum(torch.abs(stylizedFrame[:,:,:,:-1] - stylizedFrame[:,:,:,1:])) \
                    + torch.sum(torch.abs(stylizedFrame[:,:,:-1,:] - stylizedFrame[:,:,1:,:]))

                # Final loss
                loss = alpha * contentLoss \
                    + beta * styleLoss \
                    + gamma * totalDivergenceLoss

                
                ############################
                # (2) Backpropagate and optimize network weights
                ###########################

                loss.backward()
                optimizer.step()

                
                ############################
                # (3) Log and do checkpointing
                ###########################

                # Write to online logs
                onlineWriter.add_scalar('Loss/ContentLoss', contentLoss, i)
                onlineWriter.add_scalar('loss/StyleLoss', styleLoss, i)
                onlineWriter.add_scalar('loss/FinalLoss', loss, i)
                # Write to console
                # print('[%d/%d][%d/%d] Style Loss: %.4f Content Loss: %.4f'
                #     % (epoch, opt.niter, i, len(dataloader),
                #         styleLoss, contentLoss))

                with open('log.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, i, styleLoss, contentLoss])
                if i % 100 == 0:
                    torch.save(reconet.state_dict(), '%s/reconet_epoch_%d.pth' % (opt.outf, i))
                    vutils.save_image(stylizedFrame.data,
                            '%s/stylizedFrame_samples_epoch_%03d.png' % (style_model_path, epoch),
                            normalize=True)
                    onlineWriter.add_image('output/Frame', frame, i)
                    onlineWriter.add_image('output/StylizedFrame', stylizedFrame, i)

            # do checkpointing
            torch.save(reconet.state_dict(), '%s/reconet_epoch_%d.pth' % (style_model_path, epoch))

    onlineWriter.close()
    # -----------------------------------------------------------------------------------
    # Run test
    # -----------------------------------------------------------------------------------
    # else:
        # reconet.eval()
        # netG.eval()
        # loss_G = 1
        # for epoch in range(opt.niter):
        #     for i, (colorImg,greyscaleImg) in enumerate(dataloader, 0):

        #         if i == 160:
        #             break
                
        #         ############################
        #         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #         ###########################
        #         batch_size = greyscaleImg.size(0)
        #         if opt.cuda:
        #             greyscaleImg = greyscaleImg.cuda()
        #             colorImg = colorImg.cuda()
        #         greyscale.resize_as_(greyscaleImg).copy_(greyscaleImg)
        #         color.resize_as_(colorImg).copy_(colorImg)
        #         label.resize_(batch_size).fill_(real_label)
        #         greyscaleVar = Variable(greyscale)
        #         colorVar = Variable(color)
        #         labelv = Variable(label).cuda()
        #         # Test with real
        #         output = reconet(greyscaleVar, colorVar)
        #         errD_real = criterion(output, labelv)
        #         errD_real.backward()
        #         D_x = output.data.mean()
                
        #         # Generate stylizedFrame
        #         stylizedFrame = netG(greyscaleVar.detach())

        #         # Test with stylizedFrame            
        #         output = reconet(greyscaleVar.detach(),stylizedFrame.detach())
        #         errD_stylizedFrame = criterion(output, labelv)
        #         errD = errD_real + errD_stylizedFrame
        #         labelv = Variable(torch.rand(label.size())*0.0).cuda()
        #         errG = criterion(output, labelv)
        #         D_G_z = output.data.mean()

        #         print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
        #             % (epoch, opt.niter, i, len(dataloader),
        #                 errD.data[0], errG.data[0], D_x, D_G_z))

        #         with open('log_eval.csv', 'a') as f:
        #             writer = csv.writer(f)
        #             writer.writerow([epoch, i, errD.data[0], errG.data[0], D_x, D_G_z])
        #         if i % 50 == 0:
        #             vutils.save_image(stylizedFrame.data,
        #                     '%s/stylizedFrame_eval_samples_epoch_%03d_batch_%03d.png' % (opt.outf, epoch, i),
        #                     normalize=True)
        
if __name__ == "__main__":
    main()