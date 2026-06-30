# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


class FeatureExtractorNetwork(nn.Module):
    """CNN架构用于从图像数据中回归手内立方体的关键点位置。
    该网络使用卷积神经网络来处理RGB、深度和分割图像, 并输出关键点位置。
    """

    def __init__(self):
        super().__init__()
        # 输入通道数为7（RGB 3通道 + 深度 1通道 + 分割 3通道
        num_channel = 7
        # 定义卷积神经网络结构
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 58, 58]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 28, 28]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 13, 13]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 6, 6]),
            nn.AvgPool2d(6),
        )
        # 定义全连接层，输出27维关键点位置
        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )
        # 定义图像标准化变换
        self.data_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x):
        """前向传播函数
        Args:
            x (torch.Tensor): 输入图像张量，形状为(N, H, W, C)
        Returns:
            torch.Tensor: 预测的关键点位置，形状为(N, 27)
        """
        # 将通道维度移到中间位置 (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # 对RGB和分割图像应用标准化
        x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        x[:, 4:7, :, :] = self.data_transforms(x[:, 4:7, :, :])
        # 通过CNN提取特征
        cnn_x = self.cnn(x)
        # 通过全连接层输出预测结果
        out = self.linear(cnn_x.view(-1, 128))
        return out


@configclass
class FeatureExtractorCfg:
    """特征提取器模型的配置类。

    用于配置特征提取器的行为，包括训练模式、检查点加载和图像保存等选项。
    """

    train: bool = True
    """如果为True, 则在滚动过程中训练特征提取器模型。默认为False。"""

    load_checkpoint: bool = False
    """如果为True, 则从检查点加载特征提取器模型。默认为False。"""

    write_image_to_file: bool = False
    """如果为True, 则将相机传感器的图像写入文件。默认为False。"""

    # 只保存第几个 env 的图像
    save_env_id: int = 0

    # 每隔多少 step 保存一次，避免每步都写硬盘
    save_image_interval: int = 100

    # 保存目录
    save_image_dir: str = "camera_debug"


class FeatureExtractor:
    """用于从图像数据中提取特征的类。

    使用CNN从标准化的RGB、深度和分割图像中回归关键点位置。
    如果train标志设置为True, 则在滚动过程中训练CNN。
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str, log_dir: str | None = None):
        """初始化特征提取器模型。

        Args:
            cfg: 特征提取器模型的配置。
            device: 运行模型的设备。
            log_dir: 保存检查点的目录。默认为None, 使用相对于此文件的本地"logs"文件夹。
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = FeatureExtractorNetwork()
        self.feature_extractor.to(self.device)

        self.step_count = 0
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.image_dir = os.path.join(self.log_dir, self.cfg.save_image_dir)
        os.makedirs(self.image_dir, exist_ok=True)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预处理输入图像。

        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
            depth_img (torch.Tensor): 深度图像张量。形状: (N, H, W, 1)。
            segmentation_img (torch.Tensor): 分割图像张量。形状: (N, H, W, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 预处理后的RGB、深度和分割图像
        """
        rgb_img = rgb_img / 255.0
        # process depth image
        depth_img[depth_img == float("inf")] = 0
        depth_img /= 5.0
        depth_img /= torch.max(depth_img)
        # process segmentation image
        segmentation_img = segmentation_img / 255.0
        mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
        segmentation_img -= mean_tensor
        return rgb_img, depth_img, segmentation_img

    def _save_images(self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor):
        """将图像缓冲区写入文件。

        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
            depth_img (torch.Tensor): 深度图像张量。形状: (N, H, W, 1)。
            segmentation_img (torch.Tensor): 分割图像张量。形状: (N, H, W, 3)。
        """
        """只保存一个 env 的 RGB / depth / segmentation 图像。"""

        env_id = self.cfg.save_env_id

        # 防止 env_id 超过 batch 大小
        env_id = min(env_id, rgb_img.shape[0] - 1)

        # 只取一个 env，保持 batch 维度: [1, H, W, C]
        rgb_one = rgb_img[env_id : env_id + 1]
        depth_one = depth_img[env_id : env_id + 1]
        seg_one = segmentation_img[env_id : env_id + 1]

        save_images_to_file(
            rgb_one,
            os.path.join(self.image_dir, f"rgb_env{env_id}_step{self.step_count}.png"),
        )
        save_images_to_file(
            depth_one,
            os.path.join(self.image_dir, f"depth_env{env_id}_step{self.step_count}.png"),
        )
        save_images_to_file(
            seg_one,
            os.path.join(self.image_dir, f"seg_env{env_id}_step{self.step_count}.png"),
        )

    def step(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor, gt_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用图像提取特征并根据训练标志训练模型。
        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
            depth_img (torch.Tensor): 深度图像张量。形状: (N, H, W, 1)。
            segmentation_img (torch.Tensor): 分割图像张量。形状: (N, H, W, 3)。
            gt_pose (torch.Tensor): 真实姿态张量（位置和角点）。形状: (N, 27)。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 姿态损失和预测姿态       。
        """

        rgb_img, depth_img, segmentation_img = self._preprocess_images(rgb_img, depth_img, segmentation_img)

        if self.cfg.write_image_to_file and self.step_count % self.cfg.save_image_interval == 0:
            self._save_images(rgb_img, depth_img, segmentation_img)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    img_input = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            img_input = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
