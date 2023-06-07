import torch
import sklearn.metrics as metrics


def evaluate(model, criterion, test_loader, verbose=False):
    # 0. Prepare auxiliary functionality:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 1. Evaluation loop:
    with torch.no_grad():
        model.eval()
        number_samples = 0
        number_correct = 0
        running_loss_test = 0.0
        for i_test, data_test in enumerate(test_loader, 0):
            inputs_test, labels_test = data_test[0].to(device), data_test[1].long().to(device)
            outputs_test = model(inputs_test)
            loss = criterion(outputs_test, labels_test)
            running_loss_test += loss.item()
            # Accuracy:
            _, outputs_class = torch.max(outputs_test, dim=1)
            number_correct += torch.sum(outputs_class == labels_test).cpu().numpy()
            number_samples += len(labels_test)
        acc_test = number_correct / number_samples
        if verbose:
            print('Test - Accuracy: %.3f' % acc_test)
            print('Test - CrossEntropy: %.3f' % (running_loss_test / (i_test + 1)))
    return acc_test, running_loss_test / (i_test + 1)


import numpy as np
def get_probability_matrix(file_name, model, device, test_loader, num_classes=10):
    model.eval()
    probabilities = np.zeros([0, num_classes], dtype=np.float32)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            current_probs = torch.softmax(output, dim=1).cpu().numpy()
            probabilities = np.append(probabilities, current_probs, axis=0)
    np.savetxt(file_name + '_probabilities.csv', probabilities, delimiter=',')


def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]


def accuracy_top5_score(model, device, test_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs_test = model(data)
            # _, predicted = torch.max(outputs_test.data, 1)
            predicted_top5 = torch.argsort(outputs_test, dim=1, descending=True)[:, :5]
            total += target.size(0)
            correct += torch.sum((predicted_top5 == target.view(-1, 1))).item()  # DEBUG
    return correct / total

def get_prediction_metrics(model, device, test_loader, verbose=False):
    actuals, predictions = test_label_predictions(model, device, test_loader)
    confusion_matrix = metrics.confusion_matrix(actuals, predictions)
    if verbose:
        print(confusion_matrix)
    precision = metrics.precision_score(actuals, predictions, average=None)
    recall = metrics.recall_score(actuals, predictions, average=None)
    f1_score = metrics.f1_score(actuals, predictions, average=None)
    accuracy = metrics.accuracy_score(actuals, predictions)
    top5_accuracy = accuracy_top5_score(model, device, test_loader)
    if verbose:
        print('Accuracy: {}'.format(accuracy))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1-score: {}'.format(f1_score))
        print('Accuracy top5: {}'.format(top5_accuracy))
    prediction_metrics = {
        'accuracy': accuracy,
        'accuracy_top5': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,        
    }
    return prediction_metrics
