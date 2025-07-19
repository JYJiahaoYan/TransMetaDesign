from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from dataloader import CustomDataLoader
from optical_model import OpticalModel
import wandb
from loss_func import MSE_Loss
from utils import backup_code, create_folder_if_not_exists
import os
from config import FTConfig
import random
import numpy as np
import torch
from model_predict import evaluate_inverse
import pickle

# os.environ["HTTP_PROXY"] = "http://
# os.environ["HTTPS_PROXY"] = "http://
# os.environ["WANDB_MODE"]="offline"
# 设置全局随机种子
seed_value = 42  # 你可以根据需要选择一个固定的种子值
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def main(config):
    model_path = "OpticalGPT/" + config.pretrain_cfg.trainer_name + "/model"
    log_path = "OpticalGPT/" + config.pretrain_cfg.trainer_name + "/log"
    backup_path = "OpticalGPT/" + config.pretrain_cfg.trainer_name + "/backup"
    for path in [model_path, log_path, backup_path]:
        create_folder_if_not_exists(path)

    wandb.login(key="a2481ca949472cf0a93ab773c7b80f2c01001f0e")
    wandb.init(project='OpticalGPT - FT', name=config.pretrain_cfg.trainer_name, notes=config.pretrain_cfg.trainer_notes, save_code=True, dir=log_path)

    # 预训练数据模块
    pretrain_datamodule = CustomDataLoader(
        config.data_cfg.pretrain_data,
        max_num=config.data_cfg.max_data_num,
        batch_size=config.data_cfg.batch_size,
        num_workers=  config.data_cfg.num_workers,
        pin_memory=False,
    )
    pretrain_datamodule.setup()

    opticalModel = OpticalModel(MODEL_CONFIG,finetune_flag="pretrain")

    # 预训练回调

    callbacks = [
        ModelCheckpoint(
            dirpath = model_path,
            every_n_epochs=config.pretrain_cfg.checkpoint["every_n_epochs"],
            save_top_k=config.pretrain_cfg.checkpoint["save_top_k"],
            save_weights_only= config.pretrain_cfg.checkpoint["save_weights_only"],
            mode=config.pretrain_cfg.checkpoint["mode"],
            monitor=config.pretrain_cfg.checkpoint["monitor"],
            filename=config.pretrain_cfg.checkpoint["filename"],
        ),
        EarlyStopping(
            patience=config.pretrain_cfg.early_stopping["patience"],
            mode=config.pretrain_cfg.early_stopping["mode"],
            monitor=config.pretrain_cfg.early_stopping["monitor"],
            min_delta=config.pretrain_cfg.early_stopping["min_delta"],
        )
    ]

    logger = WandbLogger(project="OpticalGPT - FT", name=config.pretrain_cfg.trainer_name, notes=config.pretrain_cfg.trainer_notes, save_code=True, version=config.pretrain_cfg.trainer_name + "version")

    trainer = Trainer(
        accelerator= config.pretrain_cfg.accelerator,
        max_epochs=config.pretrain_cfg.max_epochs,

        min_steps=config.pretrain_cfg.min_steps,
        num_sanity_val_steps=config.pretrain_cfg.num_sanity_val_steps,
        callbacks=callbacks,
        logger=logger,
    )

    # 进行预训练
    print("开始预训练...")
    trainer.fit(opticalModel, datamodule=pretrain_datamodule)

    # 保存预训练模型
    print("保存预训练模型...")
    trainer.save_checkpoint(os.path.join(model_path, "pretrained_model.ckpt"))

    # 微调数据模块
    fine_tune_datamodule = CustomDataLoader(
        config.data_cfg.finetune_data,
        max_num=config.data_cfg.max_data_num,
        batch_size=config.data_cfg.batch_size,
        num_workers=  config.data_cfg.num_workers,
        pin_memory=False,
    )
    fine_tune_datamodule.setup()

    # 重新初始化模型以进行微调
    # opticalModel = OpticalModel(MODEL_CONFIG,finetune_flag="finetune")
    # opticalModel.load_from_checkpoint(os.path.join(model_path, "pretrained_model.ckpt"),config=MODEL_CONFIG,finetune_flag="finetune")
    # opticalModel.load_from_checkpoint(r"OpticalGPT/debug01/model/epoch=31-val/cer=0.14.ckpt", config=MODEL_CONFIG,
    #                                   finetune_flag="pretrain")
    opticalModel.finetune_flag = "finetune"
    config.trainer_cfg = config.finetune_cfg
    print(config.trainer_cfg)
    opticalModel.loss_fn = MSE_Loss(ignore_index=config.tokenizer.pad_index, tokenlizer=config.tokenizer)

    # 更新回调（可以自定义微调相关的设置）

    fine_tune_callbacks = [
        ModelCheckpoint(
            dirpath = model_path,
            every_n_epochs=config.finetune_cfg.checkpoint["every_n_epochs"],
            save_top_k=config.finetune_cfg.checkpoint["save_top_k"],
            save_weights_only= config.finetune_cfg.checkpoint["save_weights_only"],
            mode=config.finetune_cfg.checkpoint["mode"],
            monitor=config.finetune_cfg.checkpoint["monitor"],
            filename=config.finetune_cfg.checkpoint["filename"],
        ),
        EarlyStopping(
            patience=config.finetune_cfg.early_stopping["patience"],
            mode=config.finetune_cfg.early_stopping["mode"],
            monitor=config.finetune_cfg.early_stopping["monitor"],
            min_delta=config.finetune_cfg.early_stopping["min_delta"],
        )
    ]

    # 微调训练
    print("开始微调...")
    trainer = Trainer(
        accelerator= config.finetune_cfg.accelerator,
        max_epochs=config.finetune_cfg.max_epochs,
        min_steps=config.finetune_cfg.min_steps,
        num_sanity_val_steps=config.finetune_cfg.num_sanity_val_steps,
        callbacks=fine_tune_callbacks,
        logger=logger,
    )

    trainer.fit(opticalModel, datamodule=fine_tune_datamodule)
    # trainer.test(opticalModel, datamodule=fine_tune_datamodule)

    # 备份代码
    backup_code(os.getcwd(), backup_path, config.pretrain_cfg.trainer_name, ['config.py', 'model.py', 'dataloader.py', 'dataset.py', 'evaluate.ipynb', 'loss_func.py', 'optical_model.py', 'positional_encoding.py', 'tokenlizer.py', 'train.py', 'utils.py', 'condition.py'], config.pretrain_cfg.trainer_notes)
    predict_data = evaluate_inverse(opticalModel.model,config.tokenizer,fine_tune_datamodule.test_dataset,config)
    pickle.dump(predict_data,open(os.path.join(model_path,"predict.pkl"),"wb"))
if __name__ == "__main__":
    MODEL_CONFIG = FTConfig()
    main(MODEL_CONFIG)

