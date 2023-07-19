import numpy as np
import os, imageio

# FUNCTION:1_START 加载数据
def _minify(basedir, factors=[], resolutions=[]):
    # EXPLAIN:对basedir中的图片进行下采样，下采样后的图片保存在新的文件夹中
    # SECTION:导入文件操作相关库
    # LINK:2_START
    # 源代码方法: subprocess
    # from subprocess import check_output
    # 个人代码方法: shutil（非命令行方式）
    # from shutil import copy
    import shutil
    from PIL import Image

    # SECTION:下采样预处理
    # LINK:1_START
    # 源代码方法: 如果已经存在此文件夹，认为已经完成下采样，直接返回
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):  # 如果路径不存在
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    # 个人代码方法: 如果已经存在此文件夹，删除该文件夹，重新下采样
    # for r in factors:
    #     if not isinstance(r, int): continue # 缩放因子应为整数
    #     imgdir = os.path.join(basedir, 'images_{}'.format(r))
    #     if os.path.exists(imgdir):
    #         shutil.rmtree(imgdir)
    # for r in resolutions:
    #     imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
    #     if os.path.exists(imgdir):
    #         shutil.rmtree(imgdir)

    # SECTION:下采样
    # dir_:1. 整理以('JPG', 'jpg', 'png', 'jpeg', 'PNG')结尾的原始images目录
    # 源代码:
    # imgdir = os.path.join(basedir, 'images')
    # imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    # imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    # 个人代码:
    imgs = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith(('JPG', 'jpg', 'png', 'jpeg', 'PNG'))]
    # dir_:2. 获取原始图片文件夹地址
    imgdir_orig = os.path.join(basedir, 'images')
    # dir_:3. 获取当前工作路径
    wd = os.getcwd()
    # dir_:4. 循环下采样
    for r in factors + resolutions:
        # dir_:4.1 判断下采样方式
        if isinstance(r, int): # 如果是整数，认为是通过缩放因子进行下采样
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else: # 如果不是整数，认为是通过限制宽高进行下采样
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        # LINK:1_END
        # 源代码:
        if os.path.exists(imgdir):
            continue
        # 个人代码未使用，因为不会因为出现已经存在了该文件夹的情况（如果存在也已经被删除）
        print('Minifying', r, basedir)

        # dir_:4.2 复制文件到下采样文件夹
        # LINK:2_END
        # 源代码方法:
        # os.makedirs(imgdir)
        # check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        # 个人代码方法:
        shutil.copytree(imgdir_orig, imgdir)

        # dir_:4.3 对每个图片进行下采样
        # 打印信息
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        # LINK:1_END
        # 源代码方法:
        # os.chdir(imgdir) # 切换工作路径
        # check_output(args, shell=True)
        # os.chdir(wd) # 返回工作路径
        # if ext != 'png':
        #     check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
        #     print('Removed duplicates')
        # 个人代码方法:
        for filename in os.listdir(imgdir):
            if filename.endswith(('JPG', 'jpg', 'png', 'jpeg', 'PNG')):
                input_file = os.path.join(imgdir, filename)
                with Image.open(input_file) as img:
                    resized_img = img.resize((int(img.width/r), int(img.height/r)))
                    output_file = os.path.join(imgdir, os.path.splitext(filename)[0] + ".png")
                    resized_img.save(output_file, format="PNG")
                if os.path.splitext(filename)[1] != 'png':
                    os.remove(input_file)
        print('Done')

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    # EXPLAIN:返回下采样后的poses, bds, imgs (bds未变，poses的图片尺寸与相机焦距改变，imgs改变)
    # SECTION:加载数据
    # dir_:1. 读取poses.npy文件
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # dir_:2. 获取相机位姿poses，格式为3*5*N
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    # dir_:3. 获取场景边界bds，格式为2*N
    bds = poses_arr[:, -2:].transpose([1, 0])

    # SECTION:下采样
    # dir_:1. 整理以('JPG', 'jpg', 'png')结尾的原始images目录
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith(('JPG', 'jpg', 'png'))][0]
    # dir_:2. 获取原始图片尺寸
    sh = imageio.imread(img0).shape
    # dir_:3. 缩放
    sfx = '' # 文件夹名称，会和images进行连接，连接完成后的文件夹保存下采样的图片
    if factor is not None:      # 如果缩放因子非空，按照缩放因子进行下采样
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:    # 如果高度非空，按照高度缩放
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:     # 如果宽度非空，按照宽度缩放
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    imgdir = os.path.join(basedir, 'images' + sfx) # 下采样后的文件夹名称
    # dir_:4. 下采样检验措施
    if not os.path.exists(imgdir): # 是否含有此文件夹
        print( imgdir, 'does not exist, returning' )
        return
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith(('JPG', 'jpg', 'png'))]
    if poses.shape[-1] != len(imgfiles): # 下采样后的图片数目和原始图片数目是否相同
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return
    # dir_:5. 更新数据
    sh = imageio.imread(imgfiles[0]).shape # 下采样后的图片尺寸
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # 图片尺寸
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor # 相机焦距

    # SECTION:返回结果
    # dir_:1. 如果不需要加载图片，直接返回poses和bds
    if not load_imgs:
        return poses, bds
    # dir_:2. 如果需要加载图片，则返回poses, bds, imgs(下采样后的图片)
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles] # 对图片归一化
    imgs = np.stack(imgs, -1)
    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs
# FUNCTION:1_END


# FUNCTION:2_START 中心化相机位姿
def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    # EXPLAIN:生成3*4相机矩阵
    vec2 = normalize(z) # 对z轴进行归一化
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2)) # 叉乘求出x轴
    vec1 = normalize(np.cross(vec2, vec0)) # 叉乘求出y轴（防止y和z不垂直，再求一次叉乘）
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m
# LINK:3_END
# 源代码未使用的函数
# def ptstocam(pts, c2w):
#     tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
#     return tt
def poses_avg(poses):
    # EXPLAIN:根据poses求出平均位姿
    hwf = poses[0, :3, -1:] # 长宽焦距（所有都相同）
    center = poses[:, :3, 3].mean(0) # 中心（求平均）
    vec2 = normalize(poses[:, :3, 2].sum(0)) # z轴（求平均）
    up = poses[:, :3, 1].sum(0) # y轴（求平均）
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses):
    # EXPLAIN:将相机位姿中心化
    # dir_: 数据提前保存
    poses_ = poses+0 # 相当于copy
    # dir_:1. 计算平均位姿
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses) # 平均位姿
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    # dir_:2. 将所有的位姿以平均位姿为标准进行统一
    poses = np.linalg.inv(c2w) @ poses
    # dir_: 数据恢复并更新
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses # N*3*5
# FUNCTION:2_END


# FUNCTION:3_START 生成渲染位姿
def spherify_poses(poses, bds):
    # EXPLAIN:生成球形环绕视角的渲染相机位姿
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)
    rays_d = poses[:, :3, 2:3] # z轴
    rays_o = poses[:, :3, 3:4] # center

    # SECTION: 求出原始渲染位姿
    # dir_:1. 计算最优中心
    # TODO: 找到离所有相机中心射线距离之和最短的点（为什么这样能求出来？） ###########
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist
    pt_mindist = min_line_dist(rays_o, rays_d)
    # dir_:2. 计算标准相机位姿
    # 以center朝向为z轴，[.1, .2, .3]为y轴求出
    # [.1, .2, .3]目前没有发现实际意义，可能只是为了需要y轴而随便设置的数值，反正最后会通过x轴和z轴的叉乘重新确定y轴
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = normalize(up) # z
    vec1 = normalize(np.cross([.1, .2, .3], vec0)) # x
    vec2 = normalize(np.cross(vec0, vec1)) # y
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    # dir_:3. 将所有的位姿以标准相机位姿为标准进行统一
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    # SECTION: 计算环绕位姿
    # dir_:1. 以单位圆对原始渲染位姿进行缩放（所有center的平方根）
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc # center
    bds *= sc # 场景边界
    rad *= sc # 恢复为1
    # dir_:2. 计算球形环绕相机位姿
    centroid = np.mean(poses_reset[:, :3, 3], 0) # 所有center的中心
    zh = centroid[2] # z轴
    radcircle = np.sqrt(rad ** 2 - zh ** 2) # 1 ** 2 - zh ** 2
    new_poses = [] # 保存位姿
    for th in np.linspace(0., 2. * np.pi, 120):
        # 假设一球形坐标系，以camorigin方向为z轴
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])
        vec2 = normalize(camorigin) # z
        vec0 = normalize(np.cross(vec2, up)) # x
        vec1 = normalize(np.cross(vec2, vec0)) # y
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)
        new_poses.append(p)
    new_poses = np.stack(new_poses, 0)
    # dir_:3. 添加图片尺寸及相机焦距
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5] # 长宽焦距
    # TODO: 为什么要这样计算？ ###########
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def spiral_poses(poses, bds, path_zflat=False):
    # EXPLAIN:生成轴向环绕视角的渲染相机位姿
    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3, :4])

    # dir_:1. 寻找一个合适的焦点深度
    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # dir_:2. 计算螺旋路径的半径
    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3] # center
    # LINK:3_START
    # 源代码注释: ptstocam(poses[:3,3,:].T, c2w).T，该函数未使用
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120 # 视角数目
    N_rots = 2 # 螺旋圈数
    if path_zflat: # 螺旋改为一圈
        #             zloc = np.percentile(tt, 10, 0)[2]
        zloc = -close_depth * .1
        c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        rads[2] = 0.
        N_rots = 1
        N_views /= 2

    # dir_:3. 计算轴向环绕相机位姿
    # Generate poses for spiral path
    up = normalize(poses[:, :3, 1].sum(0))
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    return poses, render_poses, bds
# FUNCTION:3_END

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    # SECTION:加载数据(poses, bds, imgs)(下采样后的poses, bds, imgs)
    # bds未变，poses的图片尺寸与相机焦距改变，imgs改变
    poses, bds, imgs = _load_data(basedir, factor=factor)
    print('Loaded', basedir, bds.min(), bds.max())

    # SECTION:调整数据
    # Correct rotation matrix ordering and move variable dim to axis 0
    # dir_:1. 调整坐标轴 (x,y轴交换，并将y轴反向)
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # dir_:2. 调整数据格式
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # 3*5*N -> N*3*5
    images = np.moveaxis(imgs, -1, 0).astype(np.float32) #
    bds = np.moveaxis(bds, -1, 0).astype(np.float32) # 2*N -> N*2
    # Rescale if bd_factor is provided
    # dir_:3. 如果设置了边界缩放因子，调整数据 (图片尺寸和场景边界)
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc # 图片尺寸
    bds *= sc # 场景边界
    # dir_:4. 中心化相机位姿
    if recenter:
        poses = recenter_poses(poses)

    # SECTION:生成渲染位姿（用于渲染视频）
    if spherify: # 球形环绕视角
        poses, render_poses, bds = spherify_poses(poses, bds)
    else: # 轴向环绕视角
        poses, render_poses, bds = spiral_poses(poses, bds, path_zflat)

    # SECTION:返回结果
    render_poses = np.array(render_poses).astype(np.float32) # 渲染位姿
    c2w = poses_avg(poses) # 平均位姿
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1) # center之间的距离
    i_test = np.argmin(dists) # 距离最短的用于测试
    print('HOLDOUT view is', i_test)
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    return images, poses, bds, render_poses, i_test