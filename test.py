import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
import time
from collections import OrderedDict


from model.PIP import PIP as PIP


import numpy as np
import torch
from metrics import *
from skimage import measure


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument('--ROC_thr', type=int, default=10, help='num')
parser.add_argument("--model_names", default=['PIP'], type=list,
                    help="model_name: 'ACM','ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,                                            #数据集名称
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'NUDT-SIRST1', 'NUDT-SIRST-Sea', 'Image'")




parser.add_argument("--pth_dirs", default=['/home/xxx/log/xxx/PIP_630_best.pth.tar'], type=list)


parser.add_argument("--dataset_dir", default='/home/xxx/IOVarNet/Dataset')

parser.add_argument("--save_img", default= False, type=bool, help="save image of or not")  #是否保存

parser.add_argument("--img_norm_cfg", default=None, type=dict,help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")



parser.add_argument("--save_img_dir", type=str, default='/home/xxx/IOVarNet/test_result/',help="path of saved image")


parser.add_argument("--save_log", type=str, default='/home/xxx/IOVarNet/log/NUAA', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 2,3. use -1 for CPU')

global opt
opt = parser.parse_args(args=[])


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # ************************************************loss*************************************************#
        self.cal_loss = nn.BCELoss(size_average=True)
        if model_name == 'PIP':
            if mode == 'train':
                self.model = PIP(1, 1, mode='train', deepsuper=True)
            else:
                self.model = PIP(1, 1, mode='test', deepsuper=True)


    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = sum(a)
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss
    

def Test():
    if opt.test_dataset_name == 'NUAA-SIRST':
        test_set = TestSetLoader_NUAA(opt.dataset_dir, opt.test_dataset_name, opt.test_dataset_name, img_norm_cfg=opt.img_norm_cfg)
        
    elif opt.test_dataset_name == 'NUDT-SIRST':
        test_set = TestSetLoader_NUDT(opt.dataset_dir, opt.test_dataset_name, opt.test_dataset_name, img_norm_cfg=opt.img_norm_cfg)

    elif opt.test_dataset_name == 'IRSTD-1K':
        test_set = TestSetLoader_IRSTD_1K(opt.dataset_dir, opt.test_dataset_name, opt.test_dataset_name, img_norm_cfg=opt.img_norm_cfg)

    else:
        raise NotImplementedError
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    IOU = mIoU()
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)

    eval_05 = PD_FA()
    ROC_05 = ROCMetric05(nclass=1, bins=15)

    net = Net(model_name=opt.model_name, mode='test').cuda() #


    ckpt = torch.load(opt.pth_dir)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(tbar):
            img = Variable(img.cuda())


            pred, m1,m2,m3,m4,m4f,m3f,m2f,m1f,s2_1,s2_2,s2_2d,s2_1d = net.forward(img)
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]].cuda()


            IOU.update((pred > 0.5), gt_mask)

            nIoU_metric.update(pred, gt_mask)
            
            eval_05.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)

            ROC_05.update(pred, gt_mask)


            # save img
            if opt.save_img == True:

                img_save = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())

                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)

                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')






        pixAcc, mIOU = IOU.get()
        nIoU = nIoU_metric.get()
        results2 = eval_05.get()
        ture_positive_rate, false_positive_rate, recall, precision, FP, F1_score = ROC_05.get()

        print('TP: \t', ture_positive_rate)
        print('FP: \t', false_positive_rate)

        print('pixAcc: %.4f| mIoU: %.4f | nIoU: %.4f | Pd: %.4f| Fa: %.4f |F1: %.4f'
              % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F1_score * 100))




if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                Test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    opt.test_dataset_name = dataset_name
                    opt.model_name = model_name
                    opt.train_dataset_name = dataset_name
                    print(pth_dir)
                    opt.f.write(pth_dir)
                    print(opt.test_dataset_name)
                    opt.f.write(opt.test_dataset_name + '\n')
                    opt.pth_dir = pth_dir
                    # opt.pth_dir = opt.save_log + pth_dir
                    Test()
                    print('\n')
                    opt.f.write('\n')
        opt.f.close()
