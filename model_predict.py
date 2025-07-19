import pickle
import torch
from tqdm import tqdm
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def preprocess_inverse(type,wave,structure,tokenizer):
    batch_size = len(type)
    structure_encoded = [tokenizer.encode(o) for o in structure]
    max_structure_length = max(len(o) for o in structure_encoded)
    max_wave_length = 500
    structure_dtype = torch.long
    type_encoded = [tokenizer.encode(i) for i in type]
        
    max_type_length = max(len(i) for i in type_encoded)

        
    type_batch_indices = torch.zeros((batch_size, max_type_length), dtype=torch.long)
    wave_batch_indices = torch.zeros((batch_size, max_wave_length), dtype=torch.float)
    structure_batch_indices = torch.zeros((batch_size, max_structure_length), dtype=structure_dtype)

    for i in range(batch_size):
        input_word_indices = type_encoded[i]
        wave_indices = wave[i]
        structure_indices = structure_encoded[i]
        type_batch_indices[i, :len(input_word_indices)] = torch.tensor(input_word_indices, dtype=torch.long)
        wave_batch_indices[i, :len(wave_indices)] = torch.tensor(wave_indices, dtype=torch.float)
        structure_batch_indices[i, :len(structure_indices)] = torch.tensor(structure_indices, dtype=structure_dtype)
    
    return type_batch_indices, wave_batch_indices, structure_batch_indices


def evaluate_inverse(model,tokenlizer,data,config):
    device = DEVICE
    model.to(device)
    model.eval()

    type = data.type
    wave = data.wave
    structure = data.structure
    type_batch_indices, wave_batch_indices, structure_batch_indices, = preprocess_inverse(type,wave,structure,tokenlizer)
    preds = []
    for i in tqdm(range(0, len(type_batch_indices), 500)):
        preds_part = model.predict(type_batch_indices[i:i+500].to(device),wave_batch_indices[i:i+500].to(device))
        preds = preds + preds_part.tolist()
    print(preds)
    pred_data = []
    for i in preds:
        decoded = tokenlizer.decode(i)
        decoded_str = "".join([str(i) for i in decoded])
        pred_data.append(decoded_str)
    
    eval_data = {
        "config":str(config),
        "type":type,
        "wave":wave,
        "structure":structure,
        "pred":pred_data
    }
    return eval_data

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import pickle
    import torch
    from optical_model import OpticalModel
    from tokenlizer import Tokenizer
    from config import Config
    MODEL_CONFIG = Config()
    model_path = r"D:\codes\transformer_for_metasurface\OpticalGPT\VIT07\model\epoch=8-val\cer=0.17.ckpt"
    device = DEVICE
    optical_model = OpticalModel.load_from_checkpoint(model_path,config = MODEL_CONFIG).to(device)
    optical_model.freeze()
    tokenizer = Tokenizer()
    evalue_data = evaluate_inverse(optical_model.model,tokenizer,MODEL_CONFIG)
    pickle.dump(evalue_data,open(r"evalue/eval_data_VIT_07_epoch_8.pkl","wb"))