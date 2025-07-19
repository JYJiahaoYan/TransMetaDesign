import torch
from pytorch_lightning import LightningModule
from model import Optical_GPT
from loss_func import CharacterErrorRate,Critic_Loss,Token_Loss,CE_Loss,MSE_Loss


class OpticalModel(LightningModule):
    def __init__(
        self,config,finetune_flag = None,

    ):
        super().__init__()
        self.config = config
        self.finetune_flag = finetune_flag
        if self.finetune_flag == "pretrain":
            config.trainer_cfg = config.pretrain_cfg
        elif self.finetune_flag == "finetune":
            config.trainer_cfg = config.finetune_cfg
        if config.trainer_cfg.milestones is None:
            self.milestones = [5]
        self.save_hyperparameters(config.to_dict())
        self.lr = config.trainer_cfg.lr
        self.weight_decay = config.trainer_cfg.weight_decay
        self.milestones = config.trainer_cfg.milestones
        self.max_eval_samples = config.trainer_cfg.max_eval_samples
        self.gamma = config.trainer_cfg.gamma
        self.tokenizer = config.tokenizer
        self.model = Optical_GPT(config)
        if self.finetune_flag == "pretrain":
            self.loss_fn = CE_Loss(ignore_index=self.tokenizer.pad_index, tokenlizer=self.tokenizer)
        elif self.finetune_flag == "finetune":
            self.loss_fn = MSE_Loss(ignore_index=self.tokenizer.pad_index, tokenlizer=self.tokenizer)
        else:
            if config.trainer_cfg.loss_func == "ce_loss":
                self.loss_fn = CE_Loss(ignore_index=self.tokenizer.pad_index,tokenlizer=self.tokenizer)
            if config.trainer_cfg.loss_func == "token_loss":
                self.loss_fn = Token_Loss(ignore_index=self.tokenizer.pad_index,tokenlizer=self.tokenizer)
            if config.trainer_cfg.loss_func == "critic_loss":
                self.loss_fn = Critic_Loss(ignore_index=self.tokenizer.pad_index,tokenlizer=self.tokenizer)
        # self.loss_fn = Critic_Loss(ignore_index=self.tokenizer.pad_index,tokenlizer=self.tokenizer) # 使用交叉熵损失函数
        self.val_cer = CharacterErrorRate(self.tokenizer.ignore_indices) # 计算验证集错误率
        self.test_cer = CharacterErrorRate(self.tokenizer.ignore_indices)# 计算测试集错误率

    def forward(self, w, d):
        return self.model(w, d)

    def training_step(self, batch, batch_idx):
        ipt_w,ipt_d,opts = batch
        logits = self.model(ipt_w,ipt_d, opts[:, :-1]) # 结合teacher forcing进行前向传播，计算每个位置的概率
        ce_loss, mse_loss, type_loss = self.loss_fn(logits, opts,ipt_d)
        self.log("train/ce_loss",ce_loss )
        self.log("train/mse_loss", mse_loss)
        self.log("train/type_loss", type_loss)
        inverse_loss = ce_loss + mse_loss + type_loss
        self.log("train/loss", inverse_loss)
        return inverse_loss


    def validation_step(self, batch, batch_idx):
        if batch_idx >= self.max_eval_samples:
            return None  # Skip remaining batches
        ipt_w,ipt_d,opts = batch
        logits = self.model(ipt_w,ipt_d, opts[:, :-1]) # 结合teacher forcing进行前向传播，计算每个位置的概率
        # ce_loss, mse_loss = self.loss_fn(logits, opts[:, 1:]) # 计算当前输出的交叉熵损失函数值 # 计算当前输出的交叉熵损失函数值
        ce_loss, mse_loss, type_loss = self.loss_fn(logits, opts,ipt_d)
        self.log("val/ce_loss", ce_loss )
        self.log("val/mse_loss", mse_loss)
        self.log("val/type_loss", type_loss)
        inverse_loss = ce_loss + mse_loss + type_loss

        self.log("val/loss", inverse_loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = self.model.predict(ipt_w,ipt_d) # 用当前epoch下训练出的模型，预测验证集的输出
        val_cer = self.val_cer(preds, opts)# 将验证集的预测输出与真实答案计算字符误差率
        self.log("val/cer", val_cer)

    def test_step(self, batch, batch_idx):

        ipt_w,ipt_d,opts = batch
        preds = self.model.predict(ipt_w,ipt_d) # 用当前epoch下训练出的模型，预测验证集的输出
        test_cer = self.test_cer(preds, opts) # 将验证集的预测输出与真实答案计算字符误差率
        self.log("test/cer", test_cer)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay) # 使用AdamW优化器优化参数
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma) # 动态学习率调整
        return [optimizer], [scheduler]
