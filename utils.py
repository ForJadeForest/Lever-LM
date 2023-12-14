from src.load_ds_utils import load_coco_ds, load_vqav2_ds


def load_ds(cfg, split=None):
    if cfg.task.task_name == 'caption':
        ds = load_coco_ds(
            name=cfg.dataset.name,
            train_coco_dataset_root=cfg.dataset.train_coco_dataset_root,
            train_coco_annotation_file=cfg.dataset.train_coco_annotation_file,
            val_coco_dataset_root=cfg.dataset.val_coco_dataset_root,
            val_coco_annotation_file=cfg.dataset.val_coco_annotation_file,
            split=split,
        )
    elif cfg.task.task_name == 'vqa':
        ds = load_vqav2_ds(
            version=cfg.dataset.version,
            train_path=cfg.dataset.train_path,
            val_path=cfg.dataset.val_path,
            train_coco_dataset_root=cfg.dataset.train_coco_dataset_root,
            val_coco_dataset_root=cfg.dataset.val_coco_dataset_root,
            split=split,
        )
    else:
        raise ValueError(f'{cfg.task.task_name=} error, should in ["caption", "vqa"]')
    return ds
