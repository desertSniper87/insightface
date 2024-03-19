import time
import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
from torch import nn
from PIL import Image
from torchvision import transforms

from backbones import get_model
from utils.utils_config import get_config

# FOR DESKTOP TESTS

IMG_ROOT = '/run/media/torsho/87_portable/edu/dat/iCartoonFace/personai_icartoonface_rectest/icartoonface_rectest'
RECTEST_INFO = '/home/torsho/dev/insightface/recognition/arcface_torch/eval/personai_icartoonface_rectest/icartoonface_rectest_info.txt'
MODEL_PATH = '/run/media/torsho/87_portable/edu/dat/iCartoonFace/icartoonface_r50_onegpu/model.pt'

# FOR SERVER TEST

# IMG_ROOT = '/root/face-rnd/dat/personai_icartoonface_rectest/icartoonface_rectest'
# RECTEST_INFO = '/root/face-rnd/dat/personai_icartoonface_rectest/icartoonface_rectest_info.txt'
# MODEL_PATH = '/root/face-rnd/insightface/recognition/arcface_torch/work_dirs/icartoonface_r50_onegpu/model.pt'


class IcartoonFaceScore:
    model = None
    cfg = None

    def __init__(self, icartoonface_rectest_info_txt, feat_size=22500):
        super().__init__()
        self.icartoonface_rectest_info_txt = icartoonface_rectest_info_txt
        self.feat_size = feat_size

    def compute_score(self, model_path):

        self.cfg = get_config('configs/icartoonface_r50_onegpu.py')

        resnet = get_model('r50', dropout=0, fp16=False).cpu()

        if not torch.cuda.is_available():
            resnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            resnet.load_state_dict(torch.load(model_path))

        model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()

        cos = nn.CosineSimilarity()

        # distance = pairwise_distances(feats, feats, metric='cosine', n_jobs=-1)
        # dis_mat = np.argsort(distance, axis=0).T
        imgpaths, imgpath_classids = [], []
        correct_num, total_num = 0, 0
        with open(self.icartoonface_rectest_info_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # print(line)
                line_info = line.strip().split()

                if len(line_info) == 6:
                    imgpaths.append(line_info[0])
                    imgpath_classids.append(line_info[-1])
                if len(line_info) == 2:
                    imgpath1, imgpath2 = line_info[0], line_info[1]
                    idx1, idx2 = imgpaths.index(imgpath1), imgpaths.index(imgpath2)
                    idx1_classid = imgpath_classids[idx1]
                    idx2_classid = imgpath_classids[idx2]

                    # print(idx1_classid, idx2_classid)

                    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((112, 112))])
                    # transform = transforms.Compose([transforms.ToTensor()])

                    p1 = f'{IMG_ROOT}/{imgpath1}'
                    p2 = f'{IMG_ROOT}/{imgpath2}'

                    # p1 = f'eval/personai_icartoonface_rectest/icartoonface_rectest/{imgpath1}'
                    # p2= f'eval/personai_icartoonface_rectest/icartoonface_rectest/{imgpath2}'

                    img1 = torch.unsqueeze(transform(Image.open(p1)), 0)
                    img2 = torch.unsqueeze(transform(Image.open(p2)), 0)

                    m1 = self.model(img1)
                    m2 = self.model(img2)

                    sim = float(cos(m1, m2).cpu().detach().numpy()[0])

                    if sim >= 0.5 and idx1_classid == idx2_classid:
                        correct_num += 1
                    elif sim < 0.5 and idx1_classid != idx2_classid:
                        correct_num += 1

                    # for idx_var in dis_mat[idx1]:
                    #     idx_classid = imgpath_classids[idx_var]
                    #     if idx_classid == idx1_classid and idx_var == idx2:
                    #         correct_num += 1
                    #     elif not idx_classid == '-1':
                    #         continue
                    #     else:
                    #         break
                    total_num += 1
                    # if total_num:
                    print('{}/{}, accuracy: {}'.format(correct_num, total_num, 100.0 * correct_num / total_num) + ' ' +
                          imgpath1, imgpath2, sim, 'same' if idx1_classid == idx2_classid else 'different')
        if total_num == 0:
            return 0
        return 100.0 * correct_num / total_num


if __name__ == '__main__':
    # initialization
    icartoonFaceScore = IcartoonFaceScore(RECTEST_INFO)

    # compute score
    s_time = time.time()
    input_bin_path = MODEL_PATH
    print(icartoonFaceScore.compute_score(input_bin_path))
    print('total time cost: {}s'.format(time.time() - s_time))

