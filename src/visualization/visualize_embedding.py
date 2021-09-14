import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Data loading and saving:
from src.data.load_data import load_dataset

# Model interaction:
from src.model_tools.load_model import load_model

RUNS_MODELS = os.path.join('..', '..', 'reports', 'runs')


def visualize_embeddings(model, dataloader, name_embedding, add_imgs=False, device='cuda'):
    writer = SummaryWriter(log_dir=name_embedding)

    if add_imgs:
        all_images = None
    all_features = None
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            features, labels = data[0], data[1].tolist()
            if add_imgs:
                if all_images is None:
                    all_images = features
                else:
                    all_images = torch.cat([all_images, features], dim=0)
            features = features.to(device)
            # Iterate over all the layers of the model until reaching the classifier:
            model_children = model.named_children()
            name_module, module = next(model_children)
            while 'classifier' not in name_module:
                features = module(features)
                name_module, module = next(model_children)
            # ToDo: Behaviour specific to DenseNet model
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            features = module(features)
            if all_features is None:
                all_features = features.cpu()
            else:
                all_features = torch.cat([all_features, features.cpu()], dim=0)
            all_labels.extend(labels)
        if add_imgs:
            writer.add_embedding(all_features, metadata=all_labels, label_img=all_images)
        else:
            writer.add_embedding(all_features, metadata=all_labels)
    writer.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_folder = 'dense_COVID3'
    model_file_name = 'dense_COVID3_pool_sugeno_general_weight_clases_checkpoint'
    model_file_name = os.path.join(base_folder, model_file_name)
    model_type = 'dense'
    info_file_name = 'dense_COVID3_pool_sugeno_general_weight_clases_info.json'
    info_file_name = os.path.join(base_folder, info_file_name)
    dataset = 'COVID3'

    val_loader = load_dataset(dataset, batch_size=32, type='val', num_workers=0, pin_memory=True, colour=True)
    model = load_model(model_file_name, model_type=model_type, info_file_name=info_file_name).to(device)
    visualize_embeddings(model, val_loader, os.path.join(RUNS_MODELS, model_file_name + '_embedding'), device=device,
                         add_imgs=True)
