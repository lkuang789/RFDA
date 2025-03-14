import os
import time
import yaml
import argparse
import torch
import os.path as op
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import utils  # my tool box
import dataset
from net_rfda import RFDA
from pytorch_msssim import ssim, ms_ssim  # Import functional SSIM and MS-SSIM
import lpips

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def receive_arg():
    """Process all hyper-parameters and experiment settings.

    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='config/STDR.yml',
        help='Path to option YAML file.'
    )
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] is None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
    )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
    )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
    )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"  
        f"{opts_dict['test']['restore_iter']}"  
        '.pt'
    )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['test']['criterion'].get('unit', 'Unknown')

    # ==========
    # open logger
    # ==========

    log_fp = open(opts_dict['train']['log_path'], 'w')
    msg = (
        f"{'<' * 10} Test {'>' * 10}\n"  
        f"Timestamp: [{utils.get_timestr()}]\n"  
        f"\n{'<' * 10} Options {'>' * 10}\n"  
        f"{utils.dict2str(opts_dict['test'])}"
    )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ==========
    # Ensure reproducibility or Speed up
    # ==========

    # torch.backends.cudnn.benchmark = False  # if reproduce
    # torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create test data prefetchers
    # ==========

    # create datasets
    test_ds_type = opts_dict['dataset']['test']['type']
    radius = opts_dict['network']['radius']
    assert test_ds_type in dataset.__all__, \
        "Not implemented!"
    test_ds_cls = getattr(dataset, test_ds_type)
    test_ds = test_ds_cls(
        opts_dict=opts_dict['dataset']['test'],
        radius=radius
    )

    test_num = len(test_ds)
    test_vid_num = test_ds.get_vid_num()

    # create datasamplers
    test_sampler = None  # no need to sample test data

    # create dataloaders
    test_loader = utils.create_dataloader(
        dataset=test_ds,
        opts_dict=opts_dict,
        sampler=test_sampler,
        phase='val'
    )
    assert test_loader is not None

    # create dataloader prefetchers
    test_prefetcher = utils.CPUPrefetcher(test_loader)

    # ==========
    # create & load model
    # ==========

    model = RFDA(opts_dict=opts_dict['network'])

    checkpoint_save_path = opts_dict['test']['checkpoint_save_path']
    msg = f'loading model {checkpoint_save_path}...'
    print(msg)
    log_fp.write(msg + '\n')

    checkpoint = torch.load(checkpoint_save_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {checkpoint_save_path} loaded.'
    print(msg)
    log_fp.write(msg + '\n')

    model = model.cuda()
    model.eval()

    # 初始化 LPIPS 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # ==========
    # define criterion
    # ==========

    criterion_psnr = utils.PSNR()
    # No need to initialize SSIM and MS-SSIM modules when using functional interface

    # 初始化 LPIPS 计数器
    per_aver_dict_psnr = dict()
    ori_aver_dict_psnr = dict()
    per_aver_dict_ssim = dict()
    ori_aver_dict_ssim = dict()
    per_aver_dict_msssim = dict()
    ori_aver_dict_msssim = dict()
    per_aver_dict_lpips = dict()
    name_vid_dict = dict()
    for index_vid in range(test_vid_num):
        per_aver_dict_psnr[index_vid] = utils.Counter()
        ori_aver_dict_psnr[index_vid] = utils.Counter()
        per_aver_dict_ssim[index_vid] = utils.Counter()
        ori_aver_dict_ssim[index_vid] = utils.Counter()
        per_aver_dict_msssim[index_vid] = utils.Counter()
        ori_aver_dict_msssim[index_vid] = utils.Counter()
        per_aver_dict_lpips[index_vid] = utils.Counter()
        name_vid_dict[index_vid] = ""

    # ==========
    # validation
    # ==========

    # create timer
    total_timer = utils.Timer()

    pbar = tqdm(
        total=test_num,
        ncols=None
    )

    # fetch the first batch
    test_prefetcher.reset()
    val_data = test_prefetcher.next()

    with torch.no_grad():
        while val_data is not None:
            # get data
            gt_data = val_data['gt'].to(device)  # (B [RGB] H W)
            lq_data = val_data['lq'].to(device)  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!

            b, _, c, _, _ = lq_data.shape
            assert b == 1, "Not supported!"

            input_data = torch.cat(
                [lq_data[:, :, i, ...] for i in range(c)],
                dim=1
            )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            enhanced_data = model(input_data)  # (B [RGB] H W)

            # Ensure data is in [0, 1]
            gt_data = torch.clamp(gt_data, 0, 1)
            lq_data = torch.clamp(lq_data, 0, 1)
            enhanced_data = torch.clamp(enhanced_data, 0, 1)

            # eval
            batch_ori_psnr = criterion_psnr(lq_data[0, radius, ...], gt_data[0])
            batch_perf_psnr = criterion_psnr(enhanced_data[0], gt_data[0])

            batch_ori_ssim = ssim(
                lq_data[0, radius, ...].unsqueeze(0), gt_data[0].unsqueeze(0),
                data_range=1.0, size_average=True
            )
            batch_perf_ssim = ssim(
                enhanced_data[0].unsqueeze(0), gt_data[0].unsqueeze(0),
                data_range=1.0, size_average=True
            )

            batch_ori_msssim = ms_ssim(
                lq_data[0, radius, ...].unsqueeze(0), gt_data[0].unsqueeze(0),
                data_range=1.0, size_average=True
            )
            batch_perf_msssim = ms_ssim(
                enhanced_data[0].unsqueeze(0), gt_data[0].unsqueeze(0),
                data_range=1.0, size_average=True
            )

            # 计算 LPIPS，需要将图像转换到 [-1, 1] 范围
            gt_lpips = gt_data * 2 - 1
            enhanced_lpips = enhanced_data * 2 - 1
            batch_perf_lpips = lpips_fn(
                enhanced_lpips, gt_lpips
            ).mean().item()

            # display
            pbar.set_description(
                "{:s}: PSNR: [{:.3f}->{:.3f}] SSIM: [{:.4f}->{:.4f}] "  
                "MS-SSIM: [{:.4f}->{:.4f}] LPIPS: [{:.4f}]"
                .format(
                    name_vid,
                    batch_ori_psnr, batch_perf_psnr,
                    batch_ori_ssim.item(), batch_perf_ssim.item(),
                    batch_ori_msssim.item(), batch_perf_msssim.item(),
                    batch_perf_lpips
                )
            )
            pbar.update()

            # log
            per_aver_dict_psnr[index_vid].accum(volume=batch_perf_psnr)
            ori_aver_dict_psnr[index_vid].accum(volume=batch_ori_psnr)

            per_aver_dict_ssim[index_vid].accum(volume=batch_perf_ssim.item())
            ori_aver_dict_ssim[index_vid].accum(volume=batch_ori_ssim.item())

            per_aver_dict_msssim[index_vid].accum(volume=batch_perf_msssim.item())
            ori_aver_dict_msssim[index_vid].accum(volume=batch_ori_msssim.item())

            per_aver_dict_lpips[index_vid].accum(volume=batch_perf_lpips)

            if name_vid_dict[index_vid] == "":
                name_vid_dict[index_vid] = name_vid
            else:
                assert name_vid_dict[index_vid] == name_vid, "Something wrong."

            # fetch next batch
            val_data = test_prefetcher.next()

    # end of val
    pbar.close()

    # log
    msg = '\n' + '<' * 10 + ' Results ' + '>' * 10
    print(msg)
    log_fp.write(msg + '\n')
    for index_vid in range(test_vid_num):
        per_psnr = per_aver_dict_psnr[index_vid].get_ave()
        ori_psnr = ori_aver_dict_psnr[index_vid].get_ave()
        per_ssim = per_aver_dict_ssim[index_vid].get_ave()
        ori_ssim = ori_aver_dict_ssim[index_vid].get_ave()
        per_msssim = per_aver_dict_msssim[index_vid].get_ave()
        ori_msssim = ori_aver_dict_msssim[index_vid].get_ave()
        per_lpips = per_aver_dict_lpips[index_vid].get_ave()
        name_vid = name_vid_dict[index_vid]
        msg = "{:s}: PSNR: [{:.3f}->{:.3f}] SSIM: [{:.4f}->{:.4f}] "\
              "MS-SSIM: [{:.4f}->{:.4f}] LPIPS: [{:.4f}]".format(
            name_vid, ori_psnr, per_psnr,
            ori_ssim, per_ssim,
            ori_msssim, per_msssim,
            per_lpips
        )
        print(msg)
        log_fp.write(msg + '\n')

    # Calculate average results
    ave_per_psnr = np.mean(
        [per_aver_dict_psnr[index_vid].get_ave() for index_vid in range(test_vid_num)]
    )
    ave_ori_psnr = np.mean(
        [ori_aver_dict_psnr[index_vid].get_ave() for index_vid in range(test_vid_num)]
    )
    ave_per_ssim = np.mean(
        [per_aver_dict_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)]
    )
    ave_ori_ssim = np.mean(
        [ori_aver_dict_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)]
    )
    ave_per_msssim = np.mean(
        [per_aver_dict_msssim[index_vid].get_ave() for index_vid in range(test_vid_num)]
    )
    ave_ori_msssim = np.mean(
        [ori_aver_dict_msssim[index_vid].get_ave() for index_vid in range(test_vid_num)]
    )
    ave_per_lpips = np.mean(
        [per_aver_dict_lpips[index_vid].get_ave() for index_vid in range(test_vid_num)]
    )

    msg = (
        f"Average PSNR: [{ave_ori_psnr:.3f}] -> [{ave_per_psnr:.3f}] (delta: {ave_per_psnr - ave_ori_psnr:.3f})\n"  
        f"Average SSIM: [{ave_ori_ssim:.4f}] -> [{ave_per_ssim:.4f}] (delta: {ave_per_ssim - ave_ori_ssim:.4f})\n"  
        f"Average MS-SSIM: [{ave_ori_msssim:.4f}] -> [{ave_per_msssim:.4f}] (delta: {ave_per_msssim - ave_ori_msssim:.4f})\n"  
        f"Average LPIPS: [{ave_per_lpips:.4f}]"
    )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ==========
    # final log & close logger
    # ==========

    total_time = total_timer.get_interval() / 3600
    msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
    print(msg)
    log_fp.write(msg + '\n')

    msg = (
        f"\n{'<' * 10} Goodbye {'>' * 10}\n"  
        f"Timestamp: [{utils.get_timestr()}]"
    )
    print(msg)
    log_fp.write(msg + '\n')

    log_fp.close()


if __name__ == '__main__':
    main()
