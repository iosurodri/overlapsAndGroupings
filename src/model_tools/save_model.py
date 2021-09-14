import os
import torch
import json

PATH_ROOT = os.path.join('..', '..', 'reports')


def save_model(model, model_name, info_file=None):
    # Save model:
    model_path = os.path.join(PATH_ROOT, 'models')
    try:
        os.mkdir(model_path)
    except:
        pass
    torch.save(model.state_dict(), os.path.join(model_path, model_name))
    if info_file is not None:
        model_info = json.dumps(info_file)
        f = open(os.path.join(model_path, model_name + '_info.json'), 'w')
        f.write(model_info)
        f.close()
