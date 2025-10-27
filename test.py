import torch
from torch.utils.data.dataloader import DataLoader
from data import PrepASV15Dataset, PrepASV19Dataset
import models
import moded2d
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_auc_score
import numpy as np


def asv_cal_accuracies(protocol, path_data, net, device, data_type='time_frame', dataset=19):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        softmax_acc = 0
        num_files = 0
        probs = torch.empty(0, 3).to(device)

        if dataset == 15:
            test_set = PrepASV15Dataset(protocol, path_data, data_type=data_type)
        else:
            test_set = PrepASV19Dataset(protocol, path_data, data_type=data_type)

        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

        for test_batch in test_loader:
            # load batch and infer
            test_sample, test_label, sub_class = test_batch

            # # sub_class level test, comment if unwanted
            # # train & dev 0~6; eval 7~19
            # # selected_index = torch.nonzero(torch.logical_xor(sub_class == 10, sub_class == 0))[:, 0]
            # selected_index = torch.nonzero(sub_class.ne(10))[:, 0]
            # if len(selected_index) == 0:
            #     continue
            # test_sample = test_sample[selected_index, :, :]
            # test_label = test_label[selected_index]

            num_files += len(test_label)
            test_sample = test_sample.to(device)
            test_label = test_label.to(device)
            infer = net(test_sample)

            # obtain output probabilities
            t1 = F.softmax(infer, dim=1)
            t2 = test_label.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)

            # calculate example level accuracy
            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(test_label).sum().item()
            softmax_acc += batch_acc

        softmax_acc = softmax_acc / num_files

    return softmax_acc, probs.to('cpu')


def cal_roc_eer(probs, show_plot=True):
    """
    probs: tensor, number of samples * 3, containing softmax probabilities
    row wise: [genuine prob, fake prob, label]
    TP: True Fake
    FP: False Fake
    """
    all_labels = probs[:, 2]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
    one_index = torch.nonzero(all_labels).squeeze(-1)
    zero_probs = probs[zero_index, 0]
    one_probs = probs[one_index, 0]

    threshold_index = torch.linspace(-0.1, 1.01, 10000)
    tpr = torch.zeros(len(threshold_index),)
    fpr = torch.zeros(len(threshold_index),)
    cnt = 0
    for i in threshold_index:
        tpr[cnt] = one_probs.le(i).sum().item()/len(one_probs)
        fpr[cnt] = zero_probs.le(i).sum().item()/len(zero_probs)
        cnt += 1

    sum_rate = tpr + fpr
    distance_to_one = torch.abs(sum_rate - 1)
    eer_index = distance_to_one.argmin(dim=0).item()
    out_eer = 0.5*(fpr[eer_index] + 1 - tpr[eer_index]).numpy()

    if show_plot:
        print('EER: {:.4f}%.'.format(out_eer * 100))
        plt.figure(1)
        plt.plot(torch.linspace(-0.2, 1.2, 1000), torch.histc(zero_probs, bins=1000, min=-0.2, max=1.2) / len(zero_probs))
        plt.plot(torch.linspace(-0.2, 1.2, 1000), torch.histc(one_probs, bins=1000, min=-0.2, max=1.2) / len(one_probs))
        #plt.title('Gauti rezultatai su WCE')
        #plt.title('Gauti rezultatai su sumaišymo reguliavimu')
        #plt.title('Gauti rezultatai su etikečių išlyginimu')
        plt.title('Gauta histograma be reguliavimo priemonių')
        plt.xlabel("Tikro tikimybė")
        plt.ylabel('Vienai klasei tenkantis santykis')
        plt.legend(['Tikras', 'Suklastotas'])
        plt.grid()
        

        plt.figure(3)
        plt.scatter(fpr, tpr)
        plt.xlabel('Klaidingai teigiamų (netikrų) atvejų skaičius')
        plt.ylabel('Tikrai teigiamų (suklastotų) atvejų skaičius')
        #plt.title('Gauti rezultatai su WCE')
        #plt.title('Gauti rezultatai su sumaišymo reguliavimu')
        #plt.title('Gauti rezultatai su etikečių išlyginimu')
        plt.title('Gauta ROC kreivė be reguliavimo priemonių')
        plt.grid()
        plt.show()
       

    return out_eer


def tdcf(probabilities, labels, p_tar, c_miss, c_fa):
    # Assuming probabilities are for the Genuine class
    genuine_probs = probabilities[:, 0]

    # Sort probabilities and corresponding labels in descending order
    sorted_probs, sorted_labels = zip(*sorted(zip(genuine_probs, labels), reverse=True))

    # Initialize variables for t-DCF computation
    c_det = 0  # Detection cost
    c_def = 0  # Decision error cost
    p_miss = 0  # Miss rate
    p_fa = 0    # False alarm rate

    # Iterate over different operating points
    for i, threshold in enumerate(sorted_probs):
        decisions = (genuine_probs >= threshold).int()  # Binary decisions

        # Update miss and false alarm rates
        p_miss = (decisions * labels).sum().item() / labels.sum().item()
        p_fa = ((1 - decisions) * (1 - labels)).sum().item() / (1 - labels).sum().item()

        # Check if the target miss rate is achieved
        if p_miss <= p_tar:
            break

    # Calculate t-DCF components
    c_det = min(p_miss, p_fa)
    c_def = abs(p_miss - p_fa)

    # Calculate t-DCF
    tdcf_value = c_det * c_miss * labels.sum().item() + c_def * c_fa * (1 - labels).sum().item()

    return tdcf_value, threshold  # Return the threshold along with th



def asv_cal_precision_recall_f1(protocol, path_data, net, device, data_type='time_frame', dataset=19):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        predictions = torch.tensor([], dtype=torch.int).to(device)
        labels = torch.tensor([], dtype=torch.int).to(device)

        if dataset == 15:
            test_set = PrepASV15Dataset(protocol, path_data, data_type=data_type)
        else:
            test_set = PrepASV19Dataset(protocol, path_data, data_type=data_type)

        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

        for test_batch in test_loader:
            # load batch and infer
            test_sample, test_label, sub_class = test_batch

            # infer
            test_sample = test_sample.to(device)
            test_label = test_label.to(device)
            infer = net(test_sample)

            # calculate example level predictions
            infer = infer.argmax(dim=1)
            predictions = torch.cat((predictions, infer), dim=0)
            labels = torch.cat((labels, test_label), dim=0)

    precision = precision_score(labels.cpu(), predictions.cpu())
    recall = recall_score(labels.cpu(), predictions.cpu())
    f1 = f1_score(labels.cpu(), predictions.cpu())

    conf_matrix = confusion_matrix(labels.cpu(), predictions.cpu())
    print("Confusion Matrix:")
    print(conf_matrix)

    auc_score = roc_auc_score(labels.cpu(), predictions.cpu())

    return precision, recall, f1, auc_score, conf_matrix




if __name__ == '__main__':

    test_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
##    protocol_file_path = 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
##    protocol_file_path = 'D:/end-to-end-synthetic-speech-detection-main/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
##    protocol_file_path = 'D:/DS_10283_853/CM_protocol/cm_evaluation.ndx.txt'
##    data_path = 'F:/ASVSpoof2019/LA/data/dev_6/'
##    data_path = 'D:/end-to-end-synthetic-speech-detection-main/LA/LA/data/eval_6/'
##    data_path = 'D:/DS_10283_853/data/eval_6/'

    
    protocol_file_path = 'D:/DATASET TACOTRON2/Atsisiuti konvertuoti/eval1.txt'
    #data_path = 'D:/DATASET TACOTRON2/Atsisiuti konvertuoti/data/eval_6.4_cqt/'
    data_path = 'D:/DATASET TACOTRON2/Atsisiuti konvertuoti/data/eval_6/'
    
   # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
   # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
   # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

    # protocol_file_path = 'F:/ASVspoof2015/CM_protocol/cm_develop.ndx.txt'
    # # cm_train.trn
    # # cm_develop.ndx
    # # cm_evaluation.ndx
    # data_path = 'F:/ASVspoof2015/data/dev_6/'

    #Net = moded2d.DilatedNet()
    Net = models.DilatedNet()
    #Net = models.SSDNet2D()
    #Net = models.SSDNet1D()
    #Net = models.RawNet2()
    #Net = models.CombinedModel()
    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

##    check_point = torch.load('./trained_models/***.pth')
    check_point = torch.load('D:/end-to-end-synthetic-speech-detection-main/end-to-end-synthetic-speech-detection-main/trained_models/MANO_Inc_TSSDNET_BE/time_frame_9_ASVspoof2019_LA_Loss_0.0_dEER_1.18%_eEER_11.25%.pth')
   
    Net.load_state_dict(check_point['model_state_dict'])

    accuracy, probabilities = asv_cal_accuracies(protocol_file_path, data_path, Net, test_device, data_type='time_frame', dataset=19)
    print(accuracy * 100)
    

    eer = cal_roc_eer(probabilities)



     # Calculate precision, recall, and F1 score
    threshold = 0.2
    predictions_binary = (probabilities[:, 0] > threshold).int()
    test_labels = probabilities[:, 2].int()

    precision = precision_score(test_labels, predictions_binary)
    recall = recall_score(test_labels, predictions_binary)
    f1 = f1_score(test_labels, predictions_binary)

    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    conf_matrix = confusion_matrix(test_labels, predictions_binary)
    print("Confusion Matrix:")
    print(conf_matrix)

    precision, recall, thresholds = precision_recall_curve(test_labels, probabilities[:, 0])

    # Plot Precision-Recall curve
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    auc = roc_auc_score(test_labels, predictions_binary)
    print('AUC: %.3f' % auc)

    precision, recall, f1, auc_score, conf_matrix = asv_cal_precision_recall_f1(protocol_file_path, data_path, Net, test_device, data_type='time_frame', dataset=19)
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC Score: {auc_score}')


     # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title('Gauti rezultatai su WCE')
    #plt.title('Gauti rezultatai su sumaišymo reguliavimu')
    #plt.title('Gauti rezultatai su etikečių išlyginimu')
    plt.title('Gauta sutrikimo matrica be reguliavimo priemonių')
    plt.colorbar()
##    tick_marks = [i for i in range(conf_matrix.shape[0])]
##    tick_marks = [1, 0]
##    plt.xticks(tick_marks, tick_marks)
##    plt.yticks(tick_marks, tick_marks)

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['1', '0'])
    plt.yticks(tick_marks, ['1', '0'])
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.xlabel('Tikra klasė')
    plt.ylabel('Spėjimo klasė')
    plt.tight_layout()
    plt.show()
 
##    thresholds = torch.linspace(0, 1, 1000)
##    tdcf_values = []
##
##    for threshold in thresholds:
##        predictions_binary = (probabilities[:, 0] > threshold).int()
##        tdcf_value, _ = tdcf(probabilities, test_labels, p_tar=0.01, c_miss=1, c_fa=1)
##        tdcf_values.append(tdcf_value)
##
##
##    optimal_tdcf_value, optimal_threshold = min((v, t) for v, t in zip(tdcf_values, thresholds))
##
##    print(f'Optimal t-DCF Value: {optimal_tdcf_value}, Optimal Threshold: {optimal_threshold}')






    


    print('End of Program.')
