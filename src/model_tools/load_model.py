import os
import json
import torch
from src.models.dense_net import BigDenseNet
from src.models.resnet import load_resnet
from src.functions.aggregation_functions import choose_aggregation

PATH_MODELS = os.path.join('..', '..', 'reports', 'models')
default_params_file = os.path.join('..', '..', 'config', 'default_parameters.json')


def return_if_available(dictionary, original_val, key):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return original_val


def load_model(file_name, model_type, info_file_name=None, info_data=None):
    # WARNING: Changes in the network definition may (and most certainly will) break the compatibility between previous
    # saved models and this function, since the state_dict structure will possibly vary:

    # WARNING: load_model assumes that the model to be loaded was trained using a cuda device.

    model = None

    if info_data is None:
        if info_file_name is not None:
            info_file_path = os.path.join(PATH_MODELS, info_file_name)
            info_file = open(info_file_path, mode='r')
            info_data = json.load(info_file)
            info_file.close()
        else:
            raise Exception('Neither info_data dict nor info_file_name provided.')
    if model_type == 'dense':
        num_layers = info_data['num_layers']
        num_classes = info_data['num_classes']
        # bn_size = info_data['bn_size']
        growth_rate = info_data['growth_rate']
        memory_efficient = info_data['memory_efficient']
        pool_aggregations = info_data['pool_aggregations']
        pool_learning_method = info_data['pool_learning_method']
        classifier_layers = info_data['classifier_layers']
        # ToDo: DEBUG
        in_channels = info_data['in_channels']  # 3
        poolType = None
        if 'pool_type' in info_data.keys():
            poolType = info_data['pool_type']
        pool_functions = []
        for pool_aggr in pool_aggregations:
            pool_functions.append(choose_aggregation(pool_aggr))
        model = BigDenseNet(growth_rate=growth_rate, num_layers=num_layers, bn_size=4,
                            num_classes=num_classes, memory_efficient=memory_efficient,
                            pool_learning_method=pool_learning_method, pool_functions=pool_functions,
                            in_channels=in_channels, classifier_layers=classifier_layers,
                            poolType=poolType)  # DEBUG: classifier_layers=2
    elif model_type == 'resnet':
        num_layers = info_data['num_layers']
        num_classes = info_data['num_classes']
        classifier_layers = info_data['classifier_layers']
        in_channels = info_data['in_channels']
        model = load_resnet(num_layers=num_layers, num_classes=num_classes, in_channels=in_channels,
                            classifier_layers=classifier_layers)


    # Load the state_dict of the model into the newly created model (load the learnt parameters):
    file_path = os.path.join(PATH_MODELS, file_name)
    if torch.cuda.is_available():
        state_dict = torch.load(file_path)
    else:
        # Convert the data saved in cuda format to cpu:
        state_dict = torch.load(file_path, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(state_dict)
    return model


#
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(PATH_MODELS,'prueba1'),info_file_name=os.path.join(PATH_MODELS,'dense_COVIDGR1_pool_mean_mean_sugeno_general_gated_121_pretrainedCV1_0_info.json'),model_type='dense').to(device)
    model = load_model(os.path.join(PATH_MODELS,'prueba2'),info_file_name=os.path.join(PATH_MODELS,'dense_COVIDGR1_pool_mean_mean_sugeno_general_gated_121_pretrainedCV1_0_info.json'),model_type='dense').to(device)

