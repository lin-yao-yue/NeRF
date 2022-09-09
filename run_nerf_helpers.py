import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
# lambda 匿名表达式
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    # **kwargs会把多余参数中的关键字参数(s="abc")转化为dict，即dict类型的参数
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # 创建实例后，会直接执行 create_embedding_fn 得到编码方式与编码维度
        self.create_embedding_fn()

    # 得到对坐标的编码 与 编码后的维度
    def create_embedding_fn(self):
        # 最终的编码结果
        embed_fns = []
        # 输入的维度 3
        d = self.kwargs['input_dims']
        # 输出维度
        out_dim = 0
        # 最终的位置编码是否包含输入
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        # 对最高频率使用log2得 L
        max_freq = self.kwargs['max_freq_log2']
        # 0-L中的数量
        N_freqs = self.kwargs['num_freqs']

        # 编码中频率的表现方式
        if self.kwargs['log_sampling']:  # 适用于可log2运算的编码，即要求编码得每一个部分都是2^
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:  # torch.sin，torch.cos
                # x:[N_rays, N_samples, 3(xyz)]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))   # sin(x * 2^n)，cos(x * 2^n)  x是未知量
                out_dim += d

        self.embed_fns = embed_fns  # 2L个匿名函数，运行后输出 [2L, N_rays, N_samples, 3(xyz)]
        self.out_dim = out_dim  # 3*2L

    def embed(self, inputs):
        # embed_fns 中存储着编码方式的匿名函数，未知量是 x
        # [N_rays, N_samples, 2L*3(xyz)]
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)  # 根据最后一维拼接


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,  # 如果为真，最终的编码结果包含原始坐标
        'input_dims': 3,  # 输入给编码器的数据的维度,主要指input[N_rays,N_samples,3]最后一维的3(xyz)
        'max_freq_log2': multires - 1,  # 将编码频率(2^L·x中的2^L)使用log2处理后便得到0-L-1(multires-1)
        'num_freqs': multires,  # 0-L-1 一共有 L(multires) 个编码
        'log_sampling': True,  # 两种编码方式
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # 匿名函数保留未知量 x，得到可对输入进行编码的编码器
    # 从之后的代码可推断出 x 是 pts[N_rays, N_samples, 3(xyz)]
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # [N_rays, N_samples, 2L*3(xyz)]
    # 返回编码器 与 编码维度 2L*3
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        input_ch：编码后，大小为2L*3(xyz)
        """
        super(NeRF, self).__init__()
        self.D = D  # layers in network
        self.W = W  # channels per layer
        self.input_ch = input_ch  # 2L*3
        self.input_ch_views = input_ch_views  # 2L*3
        self.skips = skips  # [4]
        self.use_viewdirs = use_viewdirs
        # 第五层网络的输入还要再加入一次编码后的输入数据-->输入channel增加input_ch
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        # 第九层网络结构，加入编码后的视角
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # 输出体密度 layer
            self.alpha_linear = nn.Linear(W, 1)
            # 输出RGB layer
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):  # x: [chunk, 2L*3] or [chunk, 2*2L*3]
        # 根据输入的-1维度分解为pts和view两个部分
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                # 第五层的layer要加γ(x)
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # 输出体密度
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            # 加γ(d)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            # 输出RGB
            rgb = self.rgb_linear(h)
            # [chunk, 4(RGBA)]
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs  # [chunk, 4(RGBA)]

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Ray helpers
def get_rays(H, W, K, c2w):
    # torch.meshgrid: 行数(列数)为第一(二)个张量的元素个数, 与np.meshgrid相反，所以需要加一个转置
    # [W, H]
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    # [H, W]
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# 返回一种内参与外参矩阵下拍摄的照片中所有像素所对应的射线方向，所有射线只对应一个原点，就在光圈中心位置
# 为解决倒像问题，将透过光圈成像的几何过程 转换为 由光圈发射光线成像的几何过程 (摄像机标定p7)
def get_rays_np(H, W, K, c2w):
    # np.meshgrid 通过横纵坐标向量 生成 网格点坐标矩阵，i(横坐标),j(纵坐标)都是 [H, W] 的二维数组。
    # 从而得到每一个图片每一个像素的笛卡尔坐标: (i[x][y], j[x][y])
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # 利用相机内参 K 计算每个像素坐标相对于光心的单位方向向量，注意倒成像方面的处理
    # [H,W,3]
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # 将射线方向从相机坐标系旋转到世界坐标系 np.newaxis: 在所在位置增加一个维度
    # [H,W,1,3]*[3,3]--广播机制-->[H,W,3,3]*[3,3] = [H,W,3,3], np.sum([H,W,3,3], -1)->[H,W,3]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # c2w[:3, :3] 外参数矩阵中的旋转部分 R
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 将射线原点(0,0,0)即相机光圈平移到世界坐标系,所以平移结果就是平移矩阵T. np.broadcast_to(a, shape) 将张量广播(broadcast)到shape形状
    # 相机光圈并不是成像几何的原点，但是所有射线的原点，因为射线形成过程为：像素坐标与光圈位置相减
    # [H,W,3] 此时所有射线原点相同
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))  # c2w[:3, -1] 外参数矩阵中的平移部分 T
    return rays_o, rays_d


# 把rays移动到near平面
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    :param bins: 即 z_vals_mid, 每条ray上采样点的分布情况 [N_rays, N_samples]
    :param weights: 每个点的权重 [N_rays, N_samples]
    :param N_samples: 即 N_importance=0, fine在一条 ray 上额外的采样次数
    :param det: perturb是否为0
    :param pytest:
    :return:
    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans 防止权重为0
    # 每一条ray上每一个sample点的分布概率
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # [N_rays, N_samples]
    # torch.cumsum: 求指定维度的累加和
    # 因为是对pdf进行累加，所以ray的最后一个sample点的cdf为1
    cdf = torch.cumsum(pdf, -1)  # [N_rays, N_samples]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins)) [N_rays, 1+N_samples]

    # Take uniform samples
    if det:
        # 均匀采样
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])  # [N_rays, N_samples]
    else:
        # 随机采样
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    # torch.searchsorted：相当于upper_bound查找，返回cdf中的索引值，其对应的value比u大
    # 即寻找权重pdf比均匀pdf大的采样点
    inds = torch.searchsorted(cdf, u, right=True)  # [N_rays, N_samples]
    # 索引最小值为0
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 索引最大值为N_samples-1
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # [3(N_rays, N_samples, 1+N_samples),]
    # torch.gather 使用 inds_g 索引 cdf 上特定位置的值
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
