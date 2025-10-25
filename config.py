from pydantic import BaseModel, model_validator
from pathlib import Path
import json


class DatasetConfig(BaseModel):
    img_dir: str = "path_to_imagedir"
    seg_dir: str = "path_to_maskdir"
    img_suffix: str = ".jpg"
    seg_suffix: str = ".png"
    size: int = 1024
    num_classes: int = 18


class ModelConfig(BaseModel):
    model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    in_channels: int = 3
    channels: list[int] = [320, 640, 1280, 1280]
    num_res_blocks: int = 2
    downscale_factor: int = 8
    adapter_type: str = "full_adapter"


class Config(BaseModel):
    out_dir: str = "./result/"
    lr: float = 1e-4
    batch_size: int = 1
    epochs: int = 1
    grad_accum: int = 4
    sample_every: int = 1000

    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()

    # ====================== Validation Rules (Was ValueError Part) ======================
    @model_validator(mode="after")
    def check_dataset_or_train_dir(cls, values):
        # if values.dataset_name is None and values.train_data_dir is None:
        #     raise ValueError("Specify either `dataset_name` or `train_data_dir`")
        return values

    # ====================== JSON Utility ======================
    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))


if __name__ == "__main__":
    Config().to_json("./configs/defaults.json")
