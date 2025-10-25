from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


# ===== 1) Dataset: 1024pxに揃え、VAE用に[-1,1]、セグマップは[0,1]のRGB =====
class SegDataset(Dataset):
    def __init__(
        self,
        img_dir: str | Path = "data/images",
        seg_dir: str | Path = "data/segs",
        img_suffix: str = ".jpg",
        seg_suffix: str = ".png",
        size: int = 1024,
        num_classes: int = 18,  # 背景を含めたクラス数
    ):
        self.num_classes = num_classes
        # 画像の取得
        img_dir = Path(img_dir)
        seg_dir = Path(seg_dir)
        assert img_dir.exists(), img_dir.as_posix()
        assert seg_dir.exists(), seg_dir.as_posix()
        self.img_paths = sorted(img_dir.glob(f"*{img_suffix}"))
        self.seg_paths = [seg_dir.joinpath(p.stem + seg_suffix) for p in self.img_paths]
        assert len(self.img_paths) > 0, "There is no images. Please check suffix."
        assert len(self.seg_paths) > 0, "There is no segs. Please check name or suffix."
        # 変換を定義
        self.to_img = T.Compose(
            [
                T.Resize(
                    size, interpolation=T.InterpolationMode.BICUBIC, antialias=True
                ),
                T.CenterCrop(size),
                T.ToTensor(),  # [0,1]
                T.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1,1]
            ]
        )
        self.to_seg = T.Compose(
            [
                T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
                T.CenterCrop(size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB")
        seg = Image.open(self.seg_paths[i]).convert("P")
        seg.putpalette(color_map().flatten().tolist())
        img = self.to_img(img)  # (3,H,W), [-1,1]
        seg = self.to_seg(seg.convert("RGB"))  # PIL
        return {"pixel_values": img, "conditioning": seg}


def color_map(N=256, normalized=False) -> np.ndarray:
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
