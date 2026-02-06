"""查找数据中的异常值（outliers）。

这个脚本用于遍历所有数据，找到 action 和 state 中的异常值，并记录对应的 episode 和时间点。
"""

import json
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_raw_dataset_for_metadata(data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig):
    """创建原始数据集以获取元数据（episode_index, frame_index）。"""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return None
    
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    return dataset


def get_metadata_from_raw_sample(raw_sample: dict) -> tuple[int, int]:
    """从原始样本中提取 episode_index 和 frame_index。
    
    Returns:
        (episode_index, frame_index) 元组，如果无法获取则返回 (-1, -1)
    """
    episode_idx = -1
    frame_idx = -1
    
    # 尝试多种可能的键名
    episode_keys = ["episode_index", "episode_idx", "episode"]
    frame_keys = ["frame_index", "frame_idx", "index", "frame"]
    
    for key in episode_keys:
        if key in raw_sample:
            val = raw_sample[key]
            if isinstance(val, (np.ndarray, list)):
                val = val[0] if len(val) > 0 else -1
            episode_idx = int(val)
            break
    
    for key in frame_keys:
        if key in raw_sample:
            val = raw_sample[key]
            if isinstance(val, (np.ndarray, list)):
                val = val[0] if len(val) > 0 else -1
            frame_idx = int(val)
            break
    
    return episode_idx, frame_idx


def find_outliers(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """找出异常值。
    
    Args:
        values: 数据值，形状为 (batch_size, feature_dim)
        mean: 均值，形状为 (feature_dim,)
        std: 标准差，形状为 (feature_dim,)
        threshold: z-score 阈值，默认 3.0（即 3 个标准差）
    
    Returns:
        布尔数组，形状为 (batch_size,)，True 表示异常值
    """
    # 计算 z-score
    z_scores = np.abs((values - mean) / (std + 1e-8))
    # 如果任何特征维度超过阈值，则认为是异常值
    is_outlier = np.any(z_scores > threshold, axis=-1)
    return is_outlier


def main(
    config_name: str,
    max_frames: int | None = None,
    threshold: float = 3.0,
    output_file: str = "bad_data.json",
):
    """查找数据中的异常值。
    
    Args:
        config_name: 配置名称
        max_frames: 最大处理帧数，None 表示处理所有数据
        threshold: z-score 阈值，默认 3.0
        output_file: 输出文件路径
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 创建数据加载器用于计算统计信息
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("RLDS 数据集暂不支持，请使用 torch 数据集")
    
    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    # 计算统计信息
    print("计算统计信息...")
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats[key].get_statistics() for key in keys}
    
    print(f"State mean: {norm_stats['state'].mean}")
    print(f"State std: {norm_stats['state'].std}")
    print(f"Actions mean: {norm_stats['actions'].mean}")
    print(f"Actions std: {norm_stats['actions'].std}")

    # 创建原始数据集以获取元数据
    print("创建原始数据集以获取元数据...")
    raw_dataset = create_raw_dataset_for_metadata(data_config, config.model.action_horizon, config.model)
    
    if raw_dataset is None:
        print("警告: 无法创建原始数据集，将无法记录 episode_index 和 frame_index")
        use_metadata = False
    else:
        use_metadata = True

    # 创建转换后的数据集用于查找异常值（不打乱，以确保索引对应）
    print("创建数据集用于查找异常值...")
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    transformed_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed_dataset = _data_loader.TransformedDataset(
        transformed_dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    # 确定要处理的样本数量
    num_samples = len(transformed_dataset)
    if max_frames is not None:
        num_samples = min(num_samples, max_frames)

    # 查找异常值
    print("查找异常值...")
    bad_data = {
        "state_outliers": [],
        "action_outliers": [],
        "config_name": config_name,
        "threshold": threshold,
    }

    for sample_idx in tqdm.tqdm(range(num_samples), desc="Finding outliers"):
        try:
            # 获取转换后的样本
            sample = transformed_dataset[sample_idx]
            state_values = np.asarray(sample["state"])
            action_values = np.asarray(sample["actions"])
            
            # 获取元数据
            if use_metadata and sample_idx < len(raw_dataset):
                try:
                    raw_sample = raw_dataset[sample_idx]
                    episode_idx, frame_idx = get_metadata_from_raw_sample(raw_sample)
                except Exception:
                    episode_idx = -1
                    frame_idx = -1
            else:
                episode_idx = -1
                frame_idx = -1

            # 检查 state 异常值
            state_flat = state_values.reshape(-1, state_values.shape[-1]) if state_values.ndim > 1 else state_values.reshape(1, -1)
            state_outlier = find_outliers(
                state_flat,
                norm_stats["state"].mean,
                norm_stats["state"].std,
                threshold,
            ).any()
            
            # 检查 action 异常值
            action_flat = action_values.reshape(-1, action_values.shape[-1]) if action_values.ndim > 1 else action_values.reshape(1, -1)
            action_outlier = find_outliers(
                action_flat,
                norm_stats["actions"].mean,
                norm_stats["actions"].std,
                threshold,
            ).any()

            # 记录异常值
            if state_outlier:
                state_val = state_values
                if state_val.ndim > 0:
                    state_summary = {
                        "min": float(np.min(state_val)),
                        "max": float(np.max(state_val)),
                        "mean": float(np.mean(state_val)),
                        "shape": list(state_val.shape),
                    }
                else:
                    state_summary = {"value": float(state_val)}
                
                bad_data["state_outliers"].append({
                    "sample_idx": sample_idx,
                    "episode_index": episode_idx,
                    "frame_index": frame_idx,
                    "state_summary": state_summary,
                })
            
            if action_outlier:
                action_val = action_values
                if action_val.ndim > 0:
                    action_summary = {
                        "min": float(np.min(action_val)),
                        "max": float(np.max(action_val)),
                        "mean": float(np.mean(action_val)),
                        "shape": list(action_val.shape),
                    }
                else:
                    action_summary = {"value": float(action_val)}
                
                bad_data["action_outliers"].append({
                    "sample_idx": sample_idx,
                    "episode_index": episode_idx,
                    "frame_index": frame_idx,
                    "action_summary": action_summary,
                })
        except Exception as e:
            print(f"警告: 处理样本 {sample_idx} 时出错: {e}")
            continue

    # 保存结果
    print(f"\n找到 {len(bad_data['state_outliers'])} 个 state 异常值")
    print(f"找到 {len(bad_data['action_outliers'])} 个 action 异常值")
    print(f"保存结果到: {output_file}")
    
    with open(output_file, "w") as f:
        json.dump(bad_data, f, indent=2)

    # 打印一些统计信息
    if bad_data["state_outliers"]:
        episodes_with_bad_state = set(item["episode_index"] for item in bad_data["state_outliers"] if item["episode_index"] >= 0)
        print(f"\n包含 state 异常值的 episode 数量: {len(episodes_with_bad_state)}")
        if episodes_with_bad_state:
            print(f"Episode 索引: {sorted(episodes_with_bad_state)[:20]}...")  # 只显示前20个
    
    if bad_data["action_outliers"]:
        episodes_with_bad_action = set(item["episode_index"] for item in bad_data["action_outliers"] if item["episode_index"] >= 0)
        print(f"包含 action 异常值的 episode 数量: {len(episodes_with_bad_action)}")
        if episodes_with_bad_action:
            print(f"Episode 索引: {sorted(episodes_with_bad_action)[:20]}...")  # 只显示前20个


if __name__ == "__main__":
    tyro.cli(main)

