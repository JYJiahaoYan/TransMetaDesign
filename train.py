from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from dataloader import CustomDataLoader
from optical_model import OpticalModel
import wandb
from utils import backup_code,create_folder_if_not_exists
import os
from config import Config
import random
import numpy as np
import torch
import pickle
from model_predict import evaluate_inverse
# os.environ["HTTP_PROXY"] = "http://
# os.environ["HTTPS_PROXY"] = "http://
# os.environ["WANDB_MODE"]="offline"
# 设置全局随机种子
seed_value = 42  # 你可以根据需要选择一个固定的种子值
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def main(config):
    model_path = "OpticalGPT/"+config.trainer_cfg.trainer_name+"/model"
    log_path = "OpticalGPT/"+config.trainer_cfg.trainer_name+"/log"
    backup_path = "OpticalGPT/"+config.trainer_cfg.trainer_name+"/backup"
    for path in [model_path,log_path,backup_path]:
        create_folder_if_not_exists(path)

    wandb.login(key="a2481ca949472cf0a93ab773c7b80f2c01001f0e")
    wandb.init(project='OpticalGPT - test', name=config.trainer_cfg.trainer_name, notes=config.trainer_cfg.trainer_notes, save_code=True,dir=log_path)


    datamodule = CustomDataLoader(
        config.data_cfg.data_path,
        max_num=config.data_cfg.max_data_num,
        batch_size=config.data_cfg.batch_size,
        num_workers=  config.data_cfg.num_workers,
        pin_memory=False,
    )
    datamodule.setup()

    opticalModel = OpticalModel(config)

    callbacks= []
    callbacks.append(
        ModelCheckpoint(
            dirpath = model_path,
            every_n_epochs=config.trainer_cfg.checkpoint["every_n_epochs"],
            save_top_k=config.trainer_cfg.checkpoint["save_top_k"],
            save_weights_only= config.trainer_cfg.checkpoint["save_weights_only"],
            mode=config.trainer_cfg.checkpoint["mode"],
            monitor=config.trainer_cfg.checkpoint["monitor"],
            filename=config.trainer_cfg.checkpoint["filename"],
        )
    )

    callbacks.append(
        EarlyStopping(
            patience=config.trainer_cfg.early_stopping["patience"],
            mode=config.trainer_cfg.early_stopping["mode"],
            monitor=config.trainer_cfg.early_stopping["monitor"],
            min_delta=config.trainer_cfg.early_stopping["min_delta"],
        )
    )
    logger = WandbLogger(project="OpticalGPT",name=config.trainer_cfg.trainer_name,notes =config.trainer_cfg.trainer_notes,save_code=True,version=config.trainer_cfg.trainer_name+"version")


    trainer = Trainer(
        accelerator= config.trainer_cfg.accelerator,
        max_epochs=config.trainer_cfg.max_epochs,
        min_steps=config.trainer_cfg.min_steps,
        num_sanity_val_steps=config.trainer_cfg.num_sanity_val_steps,
        callbacks=callbacks,
        logger=logger,
        
    )

    trainer.tune(opticalModel, datamodule=datamodule)
    trainer.fit(opticalModel, datamodule=datamodule)
    trainer.test(opticalModel, datamodule=datamodule)
    backup_code(os.getcwd(),backup_path,config.trainer_cfg.trainer_name,['config.py','model.py','dataloader.py','dataset.py','evaluate.ipynb','loss_func.py','optical_model.py','positional_encoding.py','tokenlizer.py','train.py','utils.py','condition.py'],config.trainer_cfg.trainer_notes)
    opticalModel.freeze()
    predict_data = evaluate_inverse(opticalModel.model,config.tokenizer,datamodule.test_dataset,config)
    pickle.dump(predict_data,open(os.path.join(model_path,"predict.pkl"),"wb"))
if __name__ == "__main__":
    MODEL_CONFIG = Config()
    main(MODEL_CONFIG)
