import torch
import os

path = '/path/to/your/msg_model'
save_path = '/path/to/your/save_path'

for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    state = torch.load(file_path)['model']
    state_to_save = {}

    model = {}
    for k, p in state.items():
        k = k.replace('model.', '')
        if 'module' in k:
            new_k = k.replace('module.', '')
            model[new_k] = p
        else:
            model[k] = p
    state_to_save['model'] = model
    torch.save(state_to_save, os.path.join(save_path, filename))