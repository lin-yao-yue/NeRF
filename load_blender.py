import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


# z方向平移
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


# 绕着x轴旋转
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


# 绕着y轴旋转
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    # z方向的平移
    c2w = trans_t(radius)
    # @ 矩阵乘法
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    # 取反+换行
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


# 得到指定文件夹下的所有 图像、pose、测试渲染的pose、宽高焦距、分割数组。
def load_blender_data(basedir, half_res=False, testskip=1):
    """
    :param basedir: args.dataDir 数据文件夹路径
    :param half_res: 是否对图像进行半裁剪
    :param testskip: 挑选测试数据集的跳跃步长
    :return:
    """
    splits = ['train', 'val', 'test']
    metas = {}
    # 分别加载三个(train,val,test) .json 文件，保存到字典中
    for s in splits:
        # 文件路径拼接
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            # 将整个json文件中的字符串转换为字典，最终metas为字典的字典
            # 加载不同数据集中的参数
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    # 分别加载train,val,test .json 文件中所对应的数据内容
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # 如果是 train 文件夹，连续读取图像数据
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        # 有无,表示的含义不同
        # a[::-1] 相当于 a[-1:-len(a)-1:-1]
        # a[::1] 相当于 a[0:len(a)+1:1]

        # frame是[{}]的格式，包含每一张图片的路径、所对应的摄像机的外参数矩阵，以指定步长读取列表中的字典
        for frame in meta['frames'][::skip]:
            # 从指定路径读取一张图片
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            # 图片以np.array矩阵返回，将返回结果添加到图像列表中
            imgs.append(imageio.imread(fname))
            # 读取相机 camera2word 外参数矩阵
            poses.append(np.array(frame['transform_matrix']))
        # astype: 转换np.array的数据类型
        # imgs: [n, H, W, 4] 4通道(RGBA)
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        # [n, 4, 4]
        poses = np.array(poses).astype(np.float32)
        # 通过counts 记录 all_imgs 中存储的 train，val，test 图片数量
        counts.append(counts[-1] + imgs.shape[0])
        # 包含了 train、test、val 的图像的二维列表
        # [3, :, H, W, 4]
        all_imgs.append(imgs)
        all_poses.append(poses)

    # np.arrange(start, end, step) : 均匀在 [start, end) 范围内生成步长为step的列表
    # [[0: train], [train: val], [val: test]]
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    # 根据指定维度对数列或矩阵进行合并
    # 通过 counts 对 imgs 进行标定分割
    imgs = np.concatenate(all_imgs, 0)  # [train_n + val_n + test_n, h, w, 4]
    poses = np.concatenate(all_poses, 0)  # [train_n + val_n + test_n, 4, 4]

    H, W = imgs[0].shape[:2]
    # 光圈在x轴的成像范围角度
    camera_angle_x = float(meta['camera_angle_x'])
    # 计算焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # np.linspace(start, end, nums): 均匀生成以start和end作为开头和结尾的nums个数
    # 用于测试训练效果的渲染pose [40，4，4]
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    # 为了节省内存开销可以选择加载图片分辨率的一半
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split
