import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


# 将输入逐个小批量输入到网络模型中
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):  # inputs: [N_rays*N_samples, 2L*3(xyz)] or [[N_rays*N_samples, 2L*3(xyz)+2L*3(dir)]]
        # 对输入进行小批量(chunk)处理
        # 匿名函数，返回：[N_rays*N_samples, 4(RGBA)]
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    Prepares inputs and applies network 'fn'.
    :param inputs: 可推断出为 pts: [N_rays, N_samples, 3(xyz)]
    :param viewdirs: 视图方向  [N_rays, 3]
    :param fn: course网络模型，输出为(RGBA)
    :param embed_fn: 对坐标编码，功能是将输入的-1维×2L
    :param embeddirs_fn: 对方向编码, 功能是将输入的-1维×2L
    :param netchunk: 小批量输入到网络中
    :return: (RGBA)
    """

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [N_rays*N_samples, 3(xyz)]
    # 对 inputs 进行编码
    embedded = embed_fn(inputs_flat)  # [N_rays*N_samples, 2L*3(xyz)]
    # 视图不为 None，即输入了视图方向，将方向编码后与坐标编码拼接得到最终编码
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)  # 方向维度扩展成inputs形状 [N_rays, N_samples, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  # [N_rays*N_samples, 3]
        # 对输入方向进行编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # [N_rays*N_samples, 2L*3]
        embedded = torch.cat([embedded, embedded_dirs], -1)  # [N_rays*N_samples, 2L*3(xyz)+2L*3(dir)]

    # 将编码过的点以批处理的形式输入到网络模型中（每批依次从shape[0]中取netchunk），得到输出（RGBA）
    outputs_flat = batchify(fn, netchunk)(embedded)  # [N_rays*N_samples, 4(RGBA)]
    # [N_rays, N_samples, 4(RGBA)]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs  # [N_rays, N_samples, 4(RGBA)]


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    :param rays_flat: [batch, 11(3(ro)+3(rd)+2(near, far)+3(viewDir))]
    :param chunk: batch
    :param kwargs: render_kwargs_train
    :return:
    """
    # 对每个batch的训练结果进行累积
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        # 对光线进行批渲染 {'rgb_map': rgb_map([N_rays, 3(RGB)]), 'disp_map': disp_map, 'acc_map': acc_map}
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            # [3, chunk, N_rays, 3]
            all_ret[k].append(ret[k])

    # {'rgb_map': rgb_map([chunk*N_rays, 3(RGB)]), 'disp_map': disp_map, 'acc_map': acc_map}
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):  # render_kwargs_train
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch. 即[2(ro+rd), N_rand, 3]
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray. 描述了volume的边界
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    # 如果使用视图方向，将 rays_d 归一化后的单位向量作为 view_dirs
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # torch.norm：求指定维度的范数
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # [N_rand, 3]
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    # 根据是否使用c2w进行rays的获取，得到的rays的维度是不一样的，[H,W,3] 或 [N_rand, 3]，接下来将其统一为[batch, 3]的格式
    sh = rays_d.shape  # [..., 3]
    # NDC coordinates.
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # 生成光线的远近端，用于确定边界框，用于确定边界的方向是x方向
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    # [batch, 8]
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    # 视图方向聚合到光线中
    if use_viewdirs:
        # [batch, 11]
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    # 批计算光线属性
    # {'rgb_map': rgb_map([chunk*batch, 3(RGB)]), 'disp_map': disp_map, 'acc_map': acc_map}
    all_ret = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    # [3(rgb_map,disp_map,acc_map), chunk*batch, 3(RGB)]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


# NeRF模型初始化，得到NeRF中一系列的模型和参数
def create_nerf(args):
    """Instantiate NeRF's MLP model.
    args.multires: log2 of max freq for positional encoding (3D location) 即 L
    args.multires_views: log2 of max freq for positional encoding (2D direction)
    args.i_embed: set 0 for default positional encoding, -1 for none
    args.N_importance: number of additional fine samples per ray

    embed_fn：函数，将输入的-1维度(假设为3)通过编码扩展为2L*3
    input_ch：对输入编码后的-1维度的大小：2L*3(当输入的-1维是3维度的xyz时)
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    # 方向编码器
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    # 第五层网络结构有所改动，需要再加入一次编码后的输入数据
    skips = [4]

    # course 网络
    '''
    netdepth: layers in network 
    netwidth: channels per layer
    '''
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    # 模型中的梯度变量
    grad_vars = list(model.parameters())

    # fine 网络
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # 匿名函数，提供输入，方向，网络模型，输出 [N_rays, N_samples, 4(RGBA)]
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,  # 坐标编码器
                                                                        embeddirs_fn=embeddirs_fn,  # 方向编码器
                                                                        netchunk=args.netchunk)  # 网络批处理查询点的数量

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    # train
    render_kwargs_train = {
        'network_query_fn': network_query_fn,  # 匿名函数，输入位置坐标，方向坐标，以及神经网络，就可以利用神经网络返回该点对应的颜色和密度
        'perturb': args.perturb,  # 扰动，对整体算法理解没有影响
        'N_importance': args.N_importance,  # 每条光线上fine采样点的数量
        'network_fine': model_fine,  # fine 网络
        'N_samples': args.N_samples,  # 每条光线上course采样点的数量
        'network_fn': model,  # course 网络
        'use_viewdirs': args.use_viewdirs,  # 是否使用视点方向，影响到神经网络是否输出颜色
        'white_bkgd': args.white_bkgd,  # 如果为 True 将输入的 png 图像的透明部分转换成白色
        'raw_noise_std': args.raw_noise_std,  # 归一化密度
    }

    # NDC only good for LLFF-style forward facing data
    # NDC 空间，只对前向场景有效，具体解释可以看论文
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # test
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# 将每条射线上离散的点进行积分，得到对应的像素颜色
def raw2outputs(raw,  # [N_rays, N_samples, 4(RGBA)]
                z_vals,  # 每条射线上采样点的分布 [N_rays, N_samples]
                rays_d,  # ray 的方向 [N_rays, 3]
                raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4(RGBA)]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    # 渲染公式中的 1−exp(−σ δ) 部分 某个采样点的光线遮挡比率 即 某个采样点的透明度alpha
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
    # 相邻采样点之间的距离 # [N_rays, N_samples-1]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays, N_samples] 最后一维全是1e10
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    # torch.norm()默认求2范数，将dist投影到世界坐标系，采样点之间的距离转换为实际距离
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    # 每个点的 RGB 值 [N_rays, N_samples, 3]
    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # 每个点的透明度alpha [N_rays, N_samples]
    alpha = raw2alpha(raw[..., 3] + noise, dists)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    '''
    w = alpha * exp(-Σσδ)
    exp(-Σσδ) = Πexp(-σδ) = Π(1-alpha): 将求和转换为连乘
    t = torch.cat([torch.ones(alpha.shape[0], 1), 1. - alpha + 1e-10], -1): [N_rays, N_samples+1]
    T = torch.cumprod(t, -1): 保持第一列不变，后面的列依次累乘前列
    weights = alpha * T[:, :-1]: 公式中计算的是前 i-1 列的累积结果，所以舍去最后一列取 T[:, :-1]
    '''
    # 求出每一个采样点的权重w [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    # C = Σwc 整条 ray 最终的渲染颜色 [N_rays, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    # 额外信息输入
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,  # [chunk, 11(3(ro)+3(rd)+2(near, far)+3(viewDir))]
                network_fn,  # course网络，预测volume中每一个点的 RGB 和 体密度
                network_query_fn,  # 具有编码器的功能
                N_samples,  # 一条 ray 上的采样点个数
                retraw=False,  # 是否返回未经处理的预测值
                lindisp=False,  # depth 越深采样点越多，或相反
                perturb=0.,  # 若果不为零，对点进行随机采样
                N_importance=0,  # 在一条 ray 上额外的采样次数
                network_fine=None,  # fine
                white_bkgd=False,  # 是否使用白色背景
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # N_rays
    N_rays = ray_batch.shape[0]
    # rays_o, rays_d, viewdirs, near, far 的分离
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None  # [N_rays, 3]
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:  # 采样点均匀分布 near + (far-near) * t_vals
        z_vals = near * (1. - t_vals) + far * t_vals  # [N_samples,]
    else:  # depth越深采样点越密集
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

    # 每条ray上采样点的分布情况 [N_rays, N_samples]
    z_vals = z_vals.expand([N_rays, N_samples])

    # 在以上采样点分布的基础上增加扰动
    if perturb > 0.:
        # get intervals between samples
        # 计算ray上所有采样点之间中点的坐标 [N_rays, N_samples-1]
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # ray上所有采样点的上边界 [N_rays, N_samples]
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        # ray上所有采样点的下边界 [N_rays, N_samples]
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals 生成[0,1)之间的均匀分布
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # 将采样分布情况投影到世界坐标系下的 ray, 即 r=o+td。根据广播机制进行运算得
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3(xyz)]

    # raw = run_network(pts)
    # 将光线上的每个点投入到 MLP 网络 network_fn 中前向传播得到每个采样点对应的
    '''
    pts: [N_rays, N_samples, 3(xyz)]
    viewdirs: [N_rays, 3]
    network_fn: course
    raw: [N_rays, N_samples, 4(RGB，A)]
    '''
    raw = network_query_fn(pts, viewdirs, network_fn)

    # 对这些离散点进行体积渲染，即进行积分操作
    '''
    rgb_map: 每条ray最终渲染的颜色 [N_rays, 3(RGB)]
    weights: 每个点的权重 [N_rays, N_samples]
    '''
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest)

    # 分层采样的细采样阶段 fine
    if N_importance > 0:
        # [N_rays, 3(RGB)]
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [N_rays, N_samples]
        # 根据权重 weight 判断这个点在物体表面附近的概率，重新采样
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        # 添加detach(),使tensor的requires_grad为False
        z_samples = z_samples.detach()

        # 将course采样与fine采样结果结合 # [N_rays, N_samples + N_importance]
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 将采样分布情况投影到世界坐标系下的ray [N_rays, N_samples + N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        # 在course 网络上继续进行训练还是在新的 fine 网络上进行训练
        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)
        # 渲染射线颜色
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    # 创建解析对象
    parser = configargparse.ArgumentParser()
    # 给解析对象增加属性
    # 属性名前加 -- 表示可选参数
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')  # 在图片的中心区域训练迭代的次数
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')  # 图片的中心区域

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    parser = config_parser()
    # 将解析对象中的属性赋予到实例中
    args = parser.parse_args()

    # 1 Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    # 图片，位姿，参数，数据集标定 的获取
    elif args.dataset_type == 'blender':
        """
        images：图像 [train_n + val_n + test_n, H, W, 4(RGBA)] 
        poses：camera2world 外参数矩阵 [train_n + val_n + test_n，4，4]
        render_poses：用于测试训练效果的渲染pose [40，4，4]
        hwf：宽高焦距
        i_split：对images进行train，val，test的标定分割 [[0: train], [train: val], [val: test]]
        """
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # 设定volume边界框的远近边界
        near = 2.
        far = 6.

        # 将 RGBA 转换成 RGB 图像
        if args.white_bkgd:
            # 如果使用白色背景,使用透明度对图像进行处理
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            # 前三项分别是R,G,B, 第四项是透明度A
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 相机内参数
    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],  # 0.5W,0.5H是将像平面投影到像素平面后坐标的偏移
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 2 Create nerf model
    # 初始化 NeRF 网络模型
    """
    render_kwargs_train： 一个字典，包含了用于训练的各个参数值。
    render_kwargs_test： 同上
    start:
    grad_vars: 整个网络的梯度变量
    optimizer: 整个网络的优化器
    """
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    # 更新字典，加入 volume 的边界框 bounding box
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # 3 Prepare raybatch tensor if batching random rays 开始读取光线以及光线对应的像素值
    # batch size (number of random rays per gradient step) 定义批大小
    N_rand = args.N_rand
    # 是否以批处理的形式生成光线
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # rays：[N, 2(ro+rd), H, W, 3]
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)
        print('done, concats')
        # None 表示在所在位置增加一维，同 np.newaxis
        # [N, 3(ro+rd+rgb), H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, 3(ro+rd+rgb), 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # train images only
        # [train_n, H, W, 3(ro+rd+rgb), 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        # [train_n*H*W, 3(ro+rd+rgb), 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        # 打乱顺序
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    # 4 train
    start = start + 1
    for i in trange(start, N_iters):  # 使用tqdm模块 显示进度条
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # 分批加载光线，每个批大小为 N_rand
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [N_rand, 3(ro+rd+rgb), 3]
            batch = torch.transpose(batch, 0, 1)  # [3(ro+rd+rgb), N_rand, 3]
            batch_rays, target_s = batch[:2], batch[2]  # [2(ro+rd), N_rand, 3]  [N_rand, 3] 将颜色 和 射线原点方向 分离

            i_batch += N_rand
            # 经过一定批次的处理后，所有的图片都经过了一次。这时候要对数据打乱，重新再挑选
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                # torch.randperm(): 将整个张量打乱顺序后返回
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            # Random from one image
            # 从所有图像中随机选择一张图像进行训练
            img_i = np.random.choice(i_train)
            # [H, W, 3]
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]  # R, T

            if N_rand is not None:
                # get_rays_np是生成批大小的结果返回，要生成当前图像的射线结果要用get_rays
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                # 生成每个像素点的笛卡尔坐标，前 precrop_iters 次迭代先生成图像中心的像素坐标坐标
                # 即先对图像中心部分进行拟合训练
                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    # [2*dH, 2*dW, 2]
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    # 生成图像中每个像素的坐标
                    # (H, W, 2)
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                # 在训练的时候并不是给图像中每个像素都打光线，而是加载一批像素对应的光线，批大小为 N_rand
                # np.random.choice 返回的是随机选择的下标组成的数组
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                #                         横坐标组成的列表        纵坐标组成的列表
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                # ro,rd集合
                batch_rays = torch.stack([rays_o, rays_d], 0)  # [2(ro+rd), N_rand, 3]
                # 该图像的像素集合
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        '''
        rgb:[chunk*N_rand, 3(RGB)]
        '''
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        # 计算 MSE 损失
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        # 将损失转换为 PSNR 指标
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate 动态更新学习率   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
