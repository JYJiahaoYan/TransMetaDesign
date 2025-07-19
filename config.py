import json
from tokenlizer import Tokenizer

class BaseConfig:
    def __init__(self, config_path):
        self.tokenizer = Tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.sos_index = self.tokenizer.sos_index
        self.eos_index = self.tokenizer.eos_index
        self.pad_index = self.tokenizer.pad_index
        if config_path:
            self.config = self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def to_dict(self):
        config_dict = self.__dict__.copy()
        config_dict.pop("tokenizer")  # Remove tokenizer from the dictionary
        merged_dict = config_dict.copy()  # Create a new dictionary to hold merged results
        for key, value in config_dict.items():
            if isinstance(value, BaseConfig):
                merged_dict.update(value.to_dict())  # Merge with nested Config's attributes
        return merged_dict

    def __str__(self):
        return str(self.__dict__)

class DataConfig(BaseConfig):
    def __init__(self, config_path='config/data_config.json'):
        super().__init__(config_path)
        self.data_path = self.config.get('data_path')
        self.pretrain_data = self.config.get('pretrain_data')
        self.finetune_data = self.config.get('finetune_data')
        self.scaler = self.config.get('scaler')
        self.max_data_num = self.config.get('max_data_num')
        self.batch_size = self.config.get('batch_size')
        self.num_workers = self.config.get('num_workers')

class ViTConfig(BaseConfig):
    def __init__(self, config_path='config/model_config.json'):
        super().__init__(config_path)
        vit_config = self.config.get('ViT', {})
        self.num_encoder_layers = vit_config.get('num_encoder_layers')
        self.encoder_dropout = vit_config.get('encoder_dropout')
        self.emb_dropout = vit_config.get('emb_dropout')
        self.image_size = vit_config.get('image_size')
        self.patch_size = vit_config.get('patch_size')
        self.channels = vit_config.get('channels')
        self.num_decoder_layers = vit_config.get('num_decoder_layers')
        self.decoder_dropout = vit_config.get('decoder_dropout')
        self.d_model = vit_config.get('d_model')
        self.dim_feedforward = vit_config.get('dim_feedforward')
        self.heads = vit_config.get('heads')

class TransformerConfig(BaseConfig):
    def __init__(self, config_path='config/model_config.json'):
        super().__init__(config_path)
        transformer_config = self.config.get('Transformer', {})
        self.emb_dropout = transformer_config.get('emb_dropout')
        self.num_decoder_layers = transformer_config.get('num_decoder_layers')
        self.decoder_dropout = transformer_config.get('decoder_dropout')
        self.d_model = transformer_config.get('d_model')
        self.dim_feedforward = transformer_config.get('dim_feedforward')
        self.heads = transformer_config.get('heads')

class CNNConfig(BaseConfig):
    def __init__(self, config_path='config/model_config.json'):
        super().__init__(config_path)
        cnn_config = self.config.get('CNN', {})
        self.emb_dropout = cnn_config.get('emb_dropout')
        self.num_decoder_layers = cnn_config.get('num_decoder_layers')
        self.decoder_dropout = cnn_config.get('decoder_dropout')
        self.d_model = cnn_config.get('d_model')
        self.dim_feedforward = cnn_config.get('dim_feedforward')
        self.heads = cnn_config.get('heads')

class TrainerConfig(BaseConfig):
    def __init__(self, config_path='config/trainer_config.json'):
        super().__init__(config_path)
        self.trainer_name = self.config.get('trainer_name')
        self.trainer_notes = self.config.get('trainer_notes')
        self.max_epochs = self.config.get('max_epochs')
        self.min_steps = self.config.get('min_steps')
        self.max_output_len = self.config.get('max_output_len')
        self.lr = self.config.get('lr')
        self.weight_decay = self.config.get('weight_decay')
        self.milestones = self.config.get('milestones')
        self.gamma = self.config.get('gamma')
        self.max_eval_samples = self.config.get('max_eval_samples')
        self.loss_func = self.config.get('loss_func')
        self.encoder_model = self.config.get('encoder_model')
        self.checkpoint = self.config.get('checkpoint')
        self.early_stopping = self.config.get('early_stopping')
        self.accelerator = self.config.get('accelerator')
        self.num_sanity_val_steps = self.config.get('num_sanity_val_steps')
        self.beam_size = self.config.get('beam_size')
        self.temperature = self.config.get('temperature')
        


class Config(BaseConfig):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.data_cfg = DataConfig()
        self.trainer_cfg = TrainerConfig()
        if self.trainer_cfg.encoder_model == "ViT":
            self.model_cfg = ViTConfig()
        elif self.trainer_cfg.encoder_model == "CNN":
            self.model_cfg = CNNConfig()
        elif self.trainer_cfg.encoder_model == "Transformer":
            self.model_cfg = TransformerConfig()
        else:
            assert ValueError("Invalid encoder model, only ViT, CNN, and Transformer are supported.")
class FTConfig(BaseConfig):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.data_cfg = DataConfig()
        self.pretrain_cfg = TrainerConfig(config_path='config/pretrain_config.json')
        self.finetune_cfg = TrainerConfig(config_path='config/finetune_config.json')
        self.trainer_cfg = None
        if self.pretrain_cfg.encoder_model == "ViT":
            self.model_cfg = ViTConfig()
        elif self.pretrain_cfg.encoder_model == "CNN":
            self.model_cfg = CNNConfig()
        elif self.pretrain_cfg.encoder_model == "Transformer":
            self.model_cfg = TransformerConfig()
        else:
            assert ValueError("Invalid encoder model, only ViT, CNN, and Transformer are supported.")

if __name__ == "__main__":
    config = Config()
    print(config.to_dict())
    print(config)
