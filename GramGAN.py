# coding=UTF-8<code>
import time, itertools
from typing import Mapping
from torch.nn.modules.loss import L1Loss
from torch.nn.modules.module import T
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from network_GramGAN import *
from utils import *
from glob import glob
from tensorboardX import SummaryWriter

vis_dir='/tensorboardX1/weight_nonlinear'
writer = SummaryWriter(log_dir=vis_dir)
class GramGAN(object):
    def __init__(self,args):

        self.gpu_ids = args.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.aux_weight = args.aux_weight
        self.alpha_entropy_weight = args.alpha_entropy_weight
        self.L1_weight = args.L1_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.resume = args.resume

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()
        print("##### Generator #####")
        print("# blocks : ", 8)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# aux_weight: ",self.aux_weight)
        print("alpha_entropy_weight: ",self.alpha_entropy_weight)
        print("L1_weight: ",self.L1_weight)
    
    ##################################################################################
    # Model
    ##################################################################################
    def bulid_model(self):
        """DataLoader"""
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        #Notice: The images should be named XXX_label.(png, .jpg, etc.), while label is the staining style of the input images, such as H&E, PAS.
        self.trainA = ImageFolder('/train/',train_transform)
        self.testA = ImageFolder('/validA/', test_transform)
        self.testB = ImageFolder('/validB/', test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True,num_workers=2)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False,num_workers=2)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False,num_workers=2)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size).to(self.device)

        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disMA = Discriminator(input_nc=3, ndf=self.ch, n_layers=6).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(),self.disMA.parameters(), self.disLA.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.disGA.train(),self.disMA.train(), self.disLA.train()

        start_iter = 0
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, label_A = trainA_iter.next()
                _, label_A_target = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, label_A = trainA_iter.next()
                _, label_A_target = trainA_iter.next()


            real_A = real_A.to(self.device)
            label_A=label_A.to(self.device)
            label_A_target=label_A_target.to(self.device)


            # Update D
            self.set_requires_grad([self.disGA,self.disMA,self.disLA], True)  # Ds require no gradients when optimizing Gs
            self.D_optim.zero_grad()
            
            fake_A2B,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = self.genA2B(real_A, label_A_target, self.device)

            real_GA_logit, real_GA_cam_logit, real_GA_logit_2, real_GA_cam_logit_2,_,_,_,_,_,_ = self.disGA(real_A,self.device)
            real_MA_logit, real_MA_cam_logit, real_MA_logit_2, real_MA_cam_logit_2,_,_,_,_,_,_ = self.disMA(real_A,self.device)
            real_LA_logit, real_LA_cam_logit, real_LA_logit_2, real_LA_cam_logit_2,_,_,_,_,_,_ = self.disLA(real_A,self.device)

            fake_GA_logit, fake_GA_cam_logit,fake_GA_logit_2, fake_GA_cam_logit_2,_,_,_,_,_,_ = self.disGA(fake_A2B.detach(),self.device)
            fake_MA_logit, fake_MA_cam_logit,fake_MA_logit_2, fake_MA_cam_logit_2,_,_,_,_,_,_ = self.disMA(fake_A2B.detach(),self.device)
            fake_LA_logit, fake_LA_cam_logit,fake_LA_logit_2, fake_LA_cam_logit_2,_,_,_,_,_,_ = self.disLA(fake_A2B.detach(),self.device)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GA_2 = self.MSE_loss(real_GA_logit_2[:,label_A], torch.ones_like(real_GA_logit_2[:,label_A]).to(self.device))
            D_ad_cam_loss_GA_2 = self.MSE_loss(real_GA_cam_logit_2[:,label_A], torch.ones_like(real_GA_cam_logit_2[:,label_A]).to(self.device))
            D_ad_loss_LA_2 = self.MSE_loss(real_LA_logit_2[:,label_A], torch.ones_like(real_LA_logit_2[:,label_A]).to(self.device))
            D_ad_cam_loss_LA_2 = self.MSE_loss(real_LA_cam_logit_2[:,label_A], torch.ones_like(real_LA_cam_logit_2[:,label_A]).to(self.device))

            D_ad_loss_MA = self.MSE_loss(real_MA_logit, torch.ones_like(real_MA_logit).to(self.device)) + self.MSE_loss(fake_MA_logit, torch.zeros_like(fake_MA_logit).to(self.device))
            D_ad_cam_loss_MA = self.MSE_loss(real_MA_cam_logit, torch.ones_like(real_MA_cam_logit).to(self.device)) + self.MSE_loss(fake_MA_cam_logit, torch.zeros_like(fake_MA_cam_logit).to(self.device))
            D_ad_loss_MA_2 = self.MSE_loss(real_MA_logit_2[:,label_A], torch.ones_like(real_MA_logit_2[:,label_A]).to(self.device))
            D_ad_cam_loss_MA_2 = self.MSE_loss(real_MA_cam_logit_2[:,label_A], torch.ones_like(real_MA_cam_logit_2[:,label_A]).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_loss_LA + D_ad_loss_GA_2 + D_ad_loss_LA_2 + D_ad_loss_MA + D_ad_loss_MA_2)\
                      +self.aux_weight*(D_ad_cam_loss_GA + D_ad_cam_loss_LA + D_ad_cam_loss_GA_2 + D_ad_cam_loss_LA_2 + D_ad_cam_loss_MA + D_ad_cam_loss_MA_2)


            Discriminator_loss = D_loss_A
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.set_requires_grad([self.disGA,self.disMA,self.disLA], False)  # Ds require no gradients when optimizing Gs
            self.G_optim.zero_grad()

            fake_A2B,S1,S2,S3,S4,S5,S6,S7,S8,L1,L2,L3,L4,L5,L6,L7,L8 = self.genA2B(real_A,label_A_target, self.device)


            fake_A2B2A,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_= self.genA2B(fake_A2B,label_A,self.device)#Rec

            fake_A2A,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = self.genA2B(real_A,label_A,self.device)#identity

            fake_GA_logit, fake_GA_cam_logit,fake_GA_logit_2, fake_GA_cam_logit_2,_,_,_,_,_,_ = self.disGA(fake_A2B,self.device)
            fake_MA_logit, fake_MA_cam_logit,fake_MA_logit_2, fake_MA_cam_logit_2,_,_,_,_,_,_ = self.disMA(fake_A2B,self.device)
            fake_LA_logit, fake_LA_cam_logit,fake_LA_logit_2, fake_LA_cam_logit_2,_,_,_,_,_,_ = self.disLA(fake_A2B,self.device)


            
            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GA_2 = self.MSE_loss(fake_GA_logit_2[:,label_A_target], torch.ones_like(fake_GA_logit_2[:,label_A_target]).to(self.device))
            G_ad_cam_loss_GA_2 = self.MSE_loss(fake_GA_cam_logit_2[:,label_A_target], torch.ones_like(fake_GA_cam_logit_2[:,label_A_target]).to(self.device))
            G_ad_loss_LA_2 = self.MSE_loss(fake_LA_logit_2[:,label_A_target], torch.ones_like(fake_LA_logit_2[:,label_A_target]).to(self.device))
            G_ad_cam_loss_LA_2 = self.MSE_loss(fake_LA_cam_logit_2[:,label_A_target], torch.ones_like(fake_LA_cam_logit_2[:,label_A_target]).to(self.device))

            G_ad_loss_MA = self.MSE_loss(fake_MA_logit, torch.ones_like(fake_MA_logit).to(self.device))
            G_ad_cam_loss_MA = self.MSE_loss(fake_MA_cam_logit, torch.ones_like(fake_MA_cam_logit).to(self.device))
            G_ad_loss_MA_2 = self.MSE_loss(fake_MA_logit_2[:,label_A_target], torch.ones_like(fake_MA_logit_2[:,label_A_target]).to(self.device))
            G_ad_cam_loss_MA_2 = self.MSE_loss(fake_MA_cam_logit_2[:,label_A_target], torch.ones_like(fake_MA_cam_logit_2[:,label_A_target]).to(self.device))
   

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            
            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_loss_LA + G_ad_loss_GA_2 + G_ad_loss_LA_2 + G_ad_loss_MA + G_ad_loss_MA_2)+\
                        self.aux_weight*(G_ad_cam_loss_GA + G_ad_cam_loss_LA + G_ad_cam_loss_GA_2 + G_ad_cam_loss_LA_2 + G_ad_cam_loss_MA + G_ad_cam_loss_MA_2)+\
                        self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + \
                        self.alpha_entropy_weight * (S1+S2+S3+S4+S5+S6+S7+S8-4) + self.L1_weight * (L1+L2+L3+L4+L5+L6+L7+L8) 
            Generator_loss = G_loss_A
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            if step % self.print_freq == 0:
                train_sample_num = 0
                test_sample_num = 16
                A2B = np.zeros((self.img_size * 3, 0, 3))

                self.genA2B.eval(), self.disGA.eval(), self.disMA.eval() , self.disLA.eval()

                for _ in range(test_sample_num):
                    try:
                        real_A, label_A = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, label_A = testA_iter.next()

                    try:
                        real_B, label_B = testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, label_B = testB_iter.next()
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    label_A, label_B = label_A.to(self.device), label_B.to(self.device)

                    with torch.no_grad():
                        fake_A2B,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = self.genA2B(real_A,label_B,self.device)
                        _,_,_,_,heatmap_0,heatmap_1,heatmap_2_0,heatmap_2_1,heatmap_2_2,heatmap_2_3=self.disLA(real_A,self.device)


                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(real_B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                self.genA2B.train(), self.disGA.train(),self.disMA.train(), self.disLA.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 1000 == 0:
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disMA'] = self.disMA.state_dict()
                params['disLA'] = self.disLA.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disMA'] = self.disMA.state_dict()
        params['disLA'] = self.disLA.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        print(step)
        params = torch.load(os.path.join(dir,self.dataset + '_params_%07d.pt' % step), map_location='cuda:0')
        self.genA2B.load_state_dict(params['genA2B'])
        self.disGA.load_state_dict(params['disGA'])
        self.disMA.load_state_dict(params['disMA'])
        self.disLA.load_state_dict(params['disLA'])

    def finetune(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        genA2B_state_dict = self.genA2B.state_dict(params['genA2B'])
        disGA_state_dict = self.disGA.state_dict(params['disGA'])
        disMA_state_dict = self.disMA.state_dict(params['disMA'])
        disLA_state_dict = self.disLA.state_dict(params['disGA'])

        pretrained_genA2B_dict = {k: v for k, v in params.items() if k in genA2B_state_dict.items()}
        pretrained_disGA_dict = {k: v for k, v in params.items() if k in disGA_state_dict.items()}
        pretrained_disMA_dict = {k: v for k, v in params.items() if k in disMA_state_dict.items()}
        pretrained_disLA_dict = {k: v for k, v in params.items() if k in disLA_state_dict.items()}

        print(pretrained_genA2B_dict)
        print(params.items())
        print(genA2B_state_dict.items())

        genA2B_state_dict.update(pretrained_genA2B_dict)
        self.genA2B.load_state_dict(genA2B_state_dict)

        disGA_state_dict.update(pretrained_disGA_dict)
        self.disGA.load_state_dict(disGA_state_dict)

        disMA_state_dict.update(pretrained_disMA_dict)
        self.disMA.load_state_dict(disMA_state_dict)

        disLA_state_dict.update(pretrained_disLA_dict)
        self.disLA.load_state_dict(disLA_state_dict)
        print('finetune')

    def test(self):
        newlabel_target_PASM=torch.tensor([[[1.0,0.0,0.0,0.0]]]).to(self.device)
        newlabel_target_HE=torch.tensor([[[0.0,1.0,0.0,0.0]]]).to(self.device)
        newlabel_target_PAS=torch.tensor([[[0.0,0.0,1.0,0.0]]]).to(self.device)
        newlabel_target_MAS=torch.tensor([[[0.0,0.0,0.0,1.0]]]).to(self.device)

        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.testA_big_picture = ImageFolder(os.path.join(f'/Kidney'), test_transform)
        self.testA_big_picture_loader = DataLoader(self.testA_big_picture, batch_size=1, shuffle=False,num_workers=2)
        model_list = glob(os.path.join(self.result_dir, self.dataset,'model', '*.pt'))#self.result_dir, self.dataset
        if not len(model_list) == 0:
            model_list.sort()
            iter=300000
            self.load(os.path.join(self.result_dir, self.dataset,'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval()

        A2B = np.zeros((self.img_size*6 , 0, 3))

        for n, (real_A, coord) in enumerate(self.testA_big_picture_loader):
            real_A = real_A.to(self.device)
                
            fake_A2B_1_8,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_= self.genA2B(real_A,newlabel_target_PASM,self.device)

            _, _, _, _, heatmap_G_0, heatmap_G_1, heatmap_G_2_PASM, heatmap_G_2_HE, heatmap_G_2_PAS, heatmap_G_2_MAS = self.disGA(real_A,self.device)
            _, _, _, _, heatmap_G_0, heatmap_G_1, heatmap_G_2_PASM, heatmap_G_2_HE, heatmap_G_2_PAS, heatmap_G_2_MAS = self.disMA(real_A,self.device)
            _, _, _, _, heatmap_L_0, heatmap_L_1, heatmap_L_2_PASM, heatmap_L_2_HE, heatmap_L_2_PAS, heatmap_L_2_MAS = self.disLA(real_A,self.device)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', str(n) + '20.png'), RGB2BGR(tensor2numpy(denorm(fake_A2B_1_8[0]))) * 255.0)
            print(''.join(coord) + '.png')
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
