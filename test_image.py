import torch
from net import RepVGGFuse_net
import utils
from args import Args
import numpy as np
import os
from train import train
import train
import time


def load_model(path):
    model_or = RepVGGFuse_net(Args.s, Args.n, Args.channel, Args.stride)
    # if num_block <= 4:
    model = model_or
    # else:
    #    model = torch.nn.DataParallel(model_or, list(range(torch.cuda.device_count())))
    model.load_state_dict(torch.load(path))
    # decoupling
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    print(model)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    total = sum([param.nelement() for param in model.parameters()])
    print('Number	of	parameter: {:4f}M'.format(total / 1e6))

    model.eval()
    model.cuda()

    return model


def run(model, infrared_path, visible_path, output_path, img_name, fusion_type, network_type, mode):
    ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
    vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)

    if Args.cuda:
        ir_img = ir_img.cuda()
        vis_img = vis_img.cuda()

    ir_img = utils.normalize_tensor(ir_img)
    vis_img = utils.normalize_tensor(vis_img)

    img_fusion = model(ir_img, vis_img)
    # multi outputs
    file_name = 'fusion_' + fusion_type + '_' + network_type + '_' + img_name + '.png'
    output_path = output_path + file_name
    utils.save_image(img_fusion, output_path)

    print(output_path)
    print(' ')


def main():
    # run demo
    test_path = "images/40test/ir/"
    # test_path = "images/40test/ir/"
    network_type = 'RepVGGFuse'
    fusion_type = 'cat'
    imgs_paths_ir, names = utils.list_images(test_path)
    num = len(imgs_paths_ir)
    output_path = './outputs/40images/'
    # output_path = './outputs/40images/'
    mode = 'L'

    lam2_str = str(Args.lam2_vi)
    wir_str = str(Args.w_ir)
    lam3_gram = str(Args.lam3_gram)
    path = './models/final_RepVGGfuse_net_lam2_' + lam2_str + '_wir_' + wir_str + \
           '_lam3_gram_' + lam3_gram + '.model'
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    with torch.no_grad():
        model = load_model(path)

        start_time = time.time()
        for i in range(num):
            img_name = names[i]
            infrared_path = imgs_paths_ir[i]
            visible_path = infrared_path.replace('ir/', 'vis/')
            if visible_path.__contains__('IR'):
                visible_path = visible_path.replace('IR', 'VIS')
            else:
                visible_path = visible_path.replace('i.', 'v.')
            # infrared_path = test_path + 'IR' + str(index) + '.jpg'
            # visible_path = test_path + 'VIS' + str(index) + '.jpg'
            print('Infrared image path: ' + infrared_path)
            print('Visible image path: ' + visible_path)
            run(model, infrared_path, visible_path, output_path, img_name, fusion_type, network_type, mode)
        T_end_time = time.time()
        T_time = T_end_time - start_time
        print(f"Time: {T_time:.2f} seconds")
    print('Done......')


if __name__ == '__main__':
    main()
