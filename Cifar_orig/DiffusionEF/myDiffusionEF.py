import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from myutils import  TConditionalBatchNorm1d, CustomSequential, DDPMDiffuser, LearnedSinusoidalPosEmb, adjust_learning_rate2
from torchvision import transforms
from torchvision.utils import save_image
from EncodeUnet import EncodeUnetModel
from image_datasets import CIFAR10Dataset
from torch.utils.data import DataLoader
import os


class DiffusionEF(nn.Module):
    def __init__(self, args, image_size, num_timesteps, num_classes, learned_sinusoidal_dim=16):

        super().__init__()
        self.encode_unet = EncodeUnetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=256,
        out_channels=6, #learn sigma
        num_res_blocks = 2,
        attention_resolutions=tuple([32//32, 32//16, 32//8]),
        dropout=0.10,
        channel_mult=(1, 2, 3, 4 ),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False)
        self.encode_unet.convert_to_fp16()
        # unet_state_dict = torch.load('/home/lhy/Projects/DiffusionEF/256x256_diffusion_uncond.pt')
        # self.encode_unet.load_state_dict(unet_state_dict)   
        #self.encode_unet.convert_to_fp32()
        for param in self.encode_unet.parameters():
            param.requires_grad = True

        self.args = args
        self.momentum = args.momentum
        self.time_dependent = args.time_dependent

        if self.time_dependent:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1

            self.t_embed_fn = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, args.hidden_channels),
                nn.GELU(),
                nn.Linear(args.hidden_channels, args.hidden_channels),
            )
        else:
            self.t_embed_fn = lambda x: 0

        layers = []
        for _ in range(args.num_layers - 1):
            if _== 0:
                layers.append(nn.Conv2d(1024, 2048, kernel_size=3, padding=1))
                if args.act_type == 'relu':
                    layers.append(nn.ReLU())
                elif args.act_type == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise NotImplementedError
                layers.append(nn.AvgPool2d(kernel_size=2))
                layers.append(nn.Conv2d(2048, 2048, kernel_size=3, padding=1))
                if args.act_type == 'relu':
                    layers.append(nn.ReLU())
                elif args.act_type == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise NotImplementedError    
                layers.append(nn.Flatten())
                layers.append(nn.Linear(2048*2*2, 1024))
                if args.act_type == 'relu':
                    layers.append(nn.ReLU())
                elif args.act_type == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise NotImplementedError
            if _ > 0:
                layers.append(nn.Linear(1024 if _ == 1 else args.hidden_channels, args.hidden_channels, bias=args.norm_type is None))
            if args.norm_type is not None:
                if args.norm_type == 'ln':
                    layers.append(nn.LayerNorm(args.hidden_channels))
                elif args.norm_type == 'bn':
                    layers.append(TConditionalBatchNorm1d(num_timesteps, args.affine_condition_on_time, 1024 if _ == 0 else args.hidden_channels))
                elif args.norm_type == 'gn':
                    layers.append(nn.GroupNorm(16, args.hidden_channels))
                else:
                    raise NotImplementedError
            if args.act_type == 'relu':
                layers.append(nn.ReLU())
            elif args.act_type == 'gelu':
                layers.append(nn.GELU())
            else:
                raise NotImplementedError

        self.layers = CustomSequential(*layers, num_hidden_channels= args.hidden_channels)
        self.head = nn.Linear(args.hidden_channels, args.k)
        self.online_clf_head = nn.Linear(args.k, num_classes)

        self.register_buffer('eigennorm_sqr', torch.zeros(num_timesteps, args.k))
        self.register_buffer('eigenvalues', torch.zeros(num_timesteps, args.k))
        self.register_buffer('num_calls', torch.Tensor([0]))

    def forward(self, x1, x2, t, logsnr, labels=None, return_psi=False):
        with torch.no_grad():
            h1=self.encode_unet.forward(x1, timesteps=t, y=None)
            h2=self.encode_unet.forward(x2, timesteps=t, y=None)
        assert h1.shape[0] == h2.shape[0]
        h1= h1.to(torch.float32)
        h2= h2.to(torch.float32)
        t = t.item()

        t_embedding = self.t_embed_fn(logsnr)
        psi = self.head(self.layers(torch.cat([h1, h2], 0), t, t_embedding))
        
        if self.training:
            norm_sqr = psi.norm(dim=0) ** 2 / psi.shape[0]
            with torch.no_grad():
                if self.num_calls == 0:
                    self.eigennorm_sqr[t].copy_(norm_sqr.data)
                else:
                    self.eigennorm_sqr[t].mul_(self.momentum).add_(
                        norm_sqr.data, alpha = 1-self.momentum)
            
            norm_ = norm_sqr.clamp(min=0).sqrt().clamp(min=1e-6)
        else:
            norm_ = self.eigennorm_sqr[t].clamp(min=0).sqrt().clamp(min=1e-6)

        psi = psi.div(norm_)
        psi1, psi2 = psi.chunk(2, dim=0)
        if return_psi: return psi1, psi2

        psi_K_psi_diag = (psi1 * psi2).sum(0).view(-1, 1)     
        if self.training:
            with torch.no_grad():
                eigenvalues_ = psi_K_psi_diag.view(-1) / psi1.shape[0]
                if self.num_calls == 0:
                    self.eigenvalues[t].copy_(eigenvalues_.data)
                else:
                    self.eigenvalues[t].mul_(self.momentum).add_(
                        eigenvalues_.data, alpha = 1-self.momentum)
                self.num_calls += 1 

        psi2_d_K_psi1 = psi2.detach().T @ psi1
        psi1_d_K_psi2 = psi1.detach().T @ psi2

        scale = 1. / psi.shape[0]
        loss = - (psi_K_psi_diag * scale).sum() * 2
        reg = ((psi2_d_K_psi1) ** 2 * scale).triu(1).sum() \
            + ((psi1_d_K_psi2) ** 2 * scale).triu(1).sum()

        loss /= psi_K_psi_diag.numel()
        reg /= psi_K_psi_diag.numel()

        # train the classification head only on clean data
        if t < 1 :
            logits = self.online_clf_head(psi1.detach())
            cls_loss = F.cross_entropy(logits, labels.to(torch.long))
            acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        else:
            cls_loss, acc = torch.Tensor([0.]).to(x1.device), None

        return loss, reg, cls_loss, acc
    
    def test_acc(self, x1, x2, t, logsnr, labels):

        with torch.no_grad():
            psi1, psi2 = self.forward(x1, x2, t, logsnr, return_psi=True)
            logits1 = self.online_clf_head(psi1.detach())
            logits2 = self.online_clf_head(psi2.detach()) 

        acc = (torch.sum(torch.eq(torch.argmax(logits1, dim=1), labels))+ 
                torch.sum(torch.eq(torch.argmax(logits2, dim=1), labels)))/ (logits1.size(0)+logits2.size(0))

        return acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn diffusion eigenfunctions on ImageNet ')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=50)

    # for the size of image
    parser.add_argument('--resolution', type=int, default=32)

    # for specifying MLP
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--norm_type', default='bn', type=str)
    parser.add_argument('--act_type', default='relu', type=str)
    parser.add_argument('--affine_condition_on_time', default=False, action='store_true')

    # opt configs
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N', help='iterations to warmup LR')
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam')

    # for neuralef
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--time_dependent', default=False, action='store_true')

    args = parser.parse_args()
    if args.min_lr is None:
        args.min_lr = args.lr * 0.001
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True


    # mydiffusionEF implement
    # define model and initialize with pre-trained model
    diffuser = DDPMDiffuser().to(device) #the noise_schedule is linear
    model=DiffusionEF(image_size=(args.batch_size, 3, args.resolution, args.resolution), args=args, num_timesteps=1000, num_classes=10).to(device)
    print(model)
    # print("# params: {}".format(sum([p.numel() for p in model.parameters() if p.requires_grad])))

    ckpt_path = './ckpt/'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    cifar_dataset_train = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=True, transform=transform)
    cifar_dataset_test = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=False, transform=transform)
    train_data_loader= DataLoader(cifar_dataset_train, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(cifar_dataset_test, batch_size=args.test_batch_size, shuffle=False)
    # construct train dataset
    # train_image_path="/data/LargeData/Large/ImageNet/train"
    # paths, classes, classes_name= list_image_files_and_class_recursively(train_image_path)
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_data = ImageDataset(image_path=train_image_path, paths=paths, image_size=args.resolution, classes=classes, classes_name=classes_name,transform=transform)
    # train_data_loader= DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # # construct test dataset
    # test_image_path="/data/LargeData/Large/ImageNet/val"
    # test_paths, test_classes, test_classes_name= list_image_files_and_class_recursively(test_image_path)
    # test_data = ImageDataset(image_path=test_image_path, paths=test_paths, image_size=args.resolution, classes=test_classes, classes_name=test_classes_name,transform=transform)
    # test_data_loader= DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    # train the DiffusionEF model
    model.train()
    losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
    if args.opt == 'lars':
        optimizer = LARS(model.parameters(),
                         lr=args.lr, weight_decay=args.weight_decay,
                         weight_decay_filter=exclude_bias_and_norm,
                         lars_adaptation_filter=exclude_bias_and_norm)
    elif args.opt == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.lr, momentum=.9, 
                                    weight_decay=args.weight_decay)
    optimizer.zero_grad()

    logfile = open('logt.xt','w')
    
    for epoch in range(args.num_epochs):
        
        for (image1, image2), labels, classes_name in train_data_loader:
            n_steps += 1
            # cosine decay lr with warmup
            lr = adjust_learning_rate2(args.lr, optimizer, n_steps, 
                                    args.num_epochs, args.warmup_steps, args.min_lr)
            t = diffuser.sample_t().to(device)
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # diffuse the training data
            image1 = diffuser(image1, t)
            image2 = diffuser(image2, t)

            loss, reg, cls_loss, acc = model.forward(image1, image2, t, diffuser.logsnr(t), labels)
            (loss + reg * args.alpha + cls_loss).div(args.accum_iter).backward()
            if epoch % args.accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()

            losses += loss.detach().item()
            regs += reg.detach().item()
            if acc is not None:
                cls_losses += cls_loss.detach().item()
                accs += acc.detach().item()
                n_steps_clf += 1
            if n_steps%20 ==0:
                print('Epoch: {}, t: {}, LR: {:.4f}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%'.format(
                    epoch, t.item(), lr, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf), file=logfile)
                print('Epoch: {}, t: {}, LR: {:.4f}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%'.format(
                    epoch, t.item(), lr, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf))


        if epoch % 2 == 0:
            model.eval()
            test_data_loader_iter = iter(test_data_loader)
            (test_image1, test_image2), test_labels, test_classes_names = next(test_data_loader_iter)
            test_image1 = test_image1.to(device, non_blocking=True)
            test_image2 = test_image2.to(device, non_blocking=True)
            test_labels = test_labels.to(device, non_blocking=True)
            t0 = torch.Tensor([0.]).long().to(device)
            test_x1 = diffuser(test_image1, t0)
            test_x2 = diffuser(test_image2, t0)
            test_acc = model.test_acc(test_x1, test_x2, t0, diffuser.logsnr(t0), test_labels)
            n_steps+=1
            print('Epoch: {}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%, Test acc: {:.2f}%'.format(
                epoch, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf, 100 * test_acc), file=logfile)
            print('Epoch: {}, Loss: {:.4f}, Reg: {:.4f}, CLS_Loss: {:.4f}, Train acc: {:.2f}%, Test acc: {:.2f}%'.format(
                epoch, losses / n_steps, regs / n_steps, cls_losses / n_steps_clf, 100 * accs / n_steps_clf, 100 * test_acc))
            
            # if 100 * test_acc > best_acc:
            #     best_acc = 100 * test_acc
            #     torch.save(model, ckpt_path + 'best_test_model.pth')

            losses, regs, cls_losses, accs, n_steps, n_steps_clf = 0, 0, 0, 0, 0, 1e-8
            model.train()
            
