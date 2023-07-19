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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
np.random.seed(0)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    # 配置文件地址
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # 实验名称，用于log中文件夹的名称
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    # 输出目录
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    # 数据集位置
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # dir_: 1.训练操作
    # training options
    # 网络深度
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    # 网络宽度
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    # 精细网络深度
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    # 精细网络宽度
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    # 每一次训练的光线数目
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    # 学习率
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 学习率衰减，每250*1000轮衰减一次
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    # 并行处理的光线数目（如果OOM，可以减小）
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 并行处理的三维点数目（如果OOM，可以减小）
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # 是否混合图片
    # 合成的数据集一般都是True, 每次只从一张图片中选取随机光线
    # 真实的数据集一般都是False, 图形先混在一起
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    # 不加载权重
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # 粗网络的权重文件的位置
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # dir_: 2.渲染操作
    # rendering options
    # 采样数目
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    # 精细网络采样数目（一般会设置为128）
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    # 采样点扰动
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 使用视角数据
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    # 使用位置编码（0使用，-1不使用）
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    # L = 10
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    # L = 4
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    # 是否在计算透明度时添加噪声
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    # 仅进行渲染
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染测试集数据，而不是render_poses
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    # 下采样的倍数（缩放因子）
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dir_: 3.训练操作
    # training options
    # 中心裁剪的训练轮数
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    # 中心区域的定义
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dir_: 4.数据集操作
    # dataset options
    # 数据集类型
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    # 对于大型数据集，只取其中的一部分test/val sets
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    # TODO: 该种数据格式未实现
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    # TODO: 该种数据格式未实现
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    # 下采样倍数（缩放因子）
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    # 不使用规格化的设备坐标
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # 在视差而不是深度上线性采样
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    # 球形环绕位姿
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    # 只取其中的一部分
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # dir_: 5.日志操作
    # logging/saving options
    # 打印频率
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    # 该参数在run_nerf中未用到，被注释
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    # 权重保存频率
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    # 测试集保存频率
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    # 环绕位姿保存频率
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


# FUNCTION:1_START 模型创建与训练
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        # 以chunk分批进入网络，防止OOM，之后再进行拼接
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    # inputs: 光线上的点，如 [1024,64,3]，1024条光线，一条光线上64个点
    # viewdirs: 光线起点的方向
    # fn: 网络模型
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [N_rand*64,3]
    # 位置编码
    embedded = embed_fn(inputs_flat) # [N_rand*64,63]

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 对方向进行位置编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 在这里经过神经网络
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # dir_:1. 数据预处理
    # 位置编码函数
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    # 视角编码函数
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    # dir_:2. 建立网络模型
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4] # 跳跃连接
    # 粗网络
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters()) # 记录参数
    # 精细网络
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters()) # 记录参数
    # 网络训练函数
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # dir_:3. 设置网络相关属性
    # Create optimizer
    # 优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    # 开始位置
    start = 0
    basedir = args.basedir # 输出目录
    expname = args.expname # 实验名称

    ##########################
    # dir_:4. 加载模型
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path] # 保存模型位置
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    print('Found ckpts', ckpts) # 打印模型位置
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1] # 直接加载最后一个
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        # 重置开始位置和优化器
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # 加载模型
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None: # 如果精细模型存在，一起加载
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    ##########################

    # dir_:5. 训练数据封装
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }
    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # dir_:6. 测试数据封装
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
# FUNCTION:1_END


# FUNCTION:2_START 场景渲染
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # dir_:1. 数据预处理
    # 采样点之间的距离
    dists = z_vals[...,1:] - z_vals[...,:-1]
    # 将无穷远点拼接到末尾
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    # 将距离乘以方向向量的2范式，转成真正的距离
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    # RGB经过sigmoid处理
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    # 设置噪声
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # dir_:2. 计算数据
    # 计算透明度
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # 计算颜色权重
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    # dir_:3. 输出结果
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # 深度图
    depth_map = torch.sum(weights * z_vals, -1)
    # 视差图
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # 权重和
    acc_map = torch.sum(weights, -1)

    if white_bkgd: # blender数据
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
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
      perturb: .float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time
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
    # dir_:1. 分离数据
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # dir_:2. 采样
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp: # 插值采样
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else: # 逆 采样
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    # dir_:3. 随机分层采样
    if perturb > 0.:
        # get intervals between samples
        # 计算中点
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # 拼接最后一个值
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        # 拼接第一个值
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        # 随机扰动因子
        t_rand = torch.rand(z_vals.shape)
        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest: # 用固定的随机数对其进行覆盖，可以用于重现结果
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        # 随机扰动
        z_vals = lower + (upper - lower) * t_rand

    # dir_:4. 得到三维坐标(o+td)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    # dir_:5. 神经网络输出的结果
#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    # dir_:6. 体渲染
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # dir_:7. 精细网络部分
    if N_importance > 0:
        # 粗网络输出数据
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # 计算中点
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # 找到精细网络中新加的采样点
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()
        # 将采样点与粗网络中原有采样点相拼接
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 求出采样点对应的三维空间坐标
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        # 选择网络模型（粗网络或精细网络）
        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        # 神经网络输出的结果
        raw = network_query_fn(pts, viewdirs, run_fn)
        # 体渲染
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # dir_:8. 保存结果
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
    # 检查是否有异常的值
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    # 小批量渲染
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    # 将分批的再叠加在一起
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.  长度
      W: int. Width of image in pixels.   宽度
      focal: float. Focal length of pinhole camera.  焦距
      # 分块训练，防止OOM
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      # 光线，格式为[2, batch_size, 3]
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      # c2w矩阵
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      # 将光线转至ndc坐标系
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
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
    # SECTION:数据预处理
    if c2w is not None: # 如果c2w非空，重新生成光线
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else: # 使用输入的光线
        # use provided ray batch
        rays_o, rays_d = rays
    # dir_:1. 是否使用视角数据
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # 静态相机，使用其生成光线
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # 方向单位向量
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    # dir_:2. 是否变换到ndc坐标系
    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    # dir_:3. 调整数据格式
    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    # dir_:4. 场景边界
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # dir_:5. 拼接以获得输入数据
    rays = torch.cat([rays_o, rays_d, near, far], -1) # 8=3+3+1+1
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # 11=8+3

    # SECTION: 渲染
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        # 对所有返回值进行reshape
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]
# FUNCTION:2_END


# FUNCTION:3_START 场景渲染（围绕render_poses）
def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    # dir_:1. 数据预处理
    H, W, focal = hwf
    if render_factor!=0: # 缩放
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    rgbs = []
    disps = []

    # dir_:2. 根据位姿渲染图片
    t = time.time() # 记录时间
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        if savedir is not None: # 如果保存位置非空，将图片进行保存
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    # dir_:3. 整理图片并返回
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    return rgbs, disps
# FUNCTION:3_END

def train():
    # SECTION:获取数据
    # dir_:1. 获取系统参数
    parser = config_parser()
    config_file_path = "./configs/cup.txt" # 通过路径获取，不需要命令行得到
    args = parser.parse_args(['--config', config_file_path])
    # dir_:2. 加载数据
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1] # 长宽焦距
        poses = poses[:, :3, :4] # 相机位姿
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        # 确定i_test
        if not isinstance(i_test, list): # 如果不是list，转为list
            i_test = [i_test]
        if args.llffhold > 0: # 跳跃选取
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        # 确定i_val
        i_val = i_test
        # 确定i_train
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
        # 确定场景边界
        print('DEFINING BOUNDS')
        if args.no_ndc: # 不使用规格化的设备坐标
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    # dir_:3. 计算相机参数
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    # dir_:4. 确定渲染位姿
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # SECTION:创建日志目录并保存配置文件
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

    # SECTION:预准备
    # dir_:1. 创建NeRF模型
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start # 用于中间点恢复
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    # dir_:2.移动测试数据到设备上
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    # dir_:3. 是否仅需要渲染，不需要训练
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            # 3.1 选择渲染的数据
            if args.render_test: # 测试数据
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None
            # 3.2 创建保存文件夹
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            # 3.3 渲染
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return
    # dir_:4. 设置批处理方式
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand # 光线数目
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, 2(ro+rd), H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, 3(ro+rd+rgb), H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb) # 打乱
        print('done')
        i_batch = 0
    # dir_:5.移动训练数据到设备上
    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # SECTION:主循环训练
    # dir_:1.设置循环属性
    N_iters = 200000 + 1 # 总轮数
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    # Summary writers # 源代码同为注释，未使用
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    start = start + 1 # 开始位置
    # dir_:2. 循环训练
    for i in trange(start, N_iters):
        time0 = time.time() # 记录时间
        # dir_:2.1 选取光线
        # Sample random ray batch
        if use_batching: # 从所有图片上选取光线
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2] # 依次为ray_o, ray_d, rgb
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]: # 处理到末尾，重新打乱并开始
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else: # 从一张图片上选取光线
            # Random from one image
            img_i = np.random.choice(i_train) # 随机选择一张图片
            target = images[img_i] # 目标颜色
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4] # 位姿
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                if i < args.precrop_iters: # 优先训练中心区域
                    dH = int(H//2 * args.precrop_frac) # 裁剪掉的区域
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else: # 训练全图
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                # 坐标点
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                # 随机选择N_rand条光线
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        # dir_:2.2 网络学习
        #####  Core optimization loop  #####
        # 渲染
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)
        # 梯度清零
        optimizer.zero_grad()
        # 计算损失与psnr指标
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1] # trans未被使用
        loss = img_loss
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s) # 粗网络损失
            loss = loss + img_loss0 # 将精细网络损失和粗网络损失相加
            psnr0 = mse2psnr(img_loss0)
        # 梯度回传与参数更新
        loss.backward()
        optimizer.step()
        # NOTE: IMPORTANT! # 重要的!
        # 学习率更新
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps)) # 指数衰减
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}") # 保留源代码注释
        #####           end            #####

        # dir_:2.3 保存模型
        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                # 运行的轮次数目
                'global_step': global_step,
                # 粗网络的权重
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 精细网络的权重
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                # 优化器的状态
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # dir_:2.3 生成测试视频，使用render_poses
        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs: # 保留源代码注释
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # dir_:2.4 执行测试，使用测试数据
        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        # dir_:2.5 打印log信息
        if i%args.i_print==0:
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

if __name__=='__main__':
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.FloatTensor')

    train()