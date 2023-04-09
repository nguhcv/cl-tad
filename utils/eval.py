import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def avg_euclidean(value_list,E=None,U=None, w=None, ):
    import torch.nn.functional as F

    # print(E.shape, U.shape)
    for idx in range (U.shape[0]):
        sample = U[idx]
        sample = torch.unsqueeze(sample,dim=0)
        # print(sample.shape)
        start_idx = idx*w
        end_idx = start_idx+w
        features = E[start_idx:end_idx,:]
        features = F.normalize(features,dim=1)
        sample = F.normalize(sample)
        sim = torch.zeros((len(features),1))
        for i in range (len(sim)):
            sim[i] = torch.sum(torch.square(features[i] - sample[0]))
            sim[i] = torch.sqrt(sim[i])

        sim = torch.flatten(sim).numpy()
        for v in sim:
            value_list.append(v)

        # value_list.append(torch.mean(sim))
        # if measure_method == 'max':
        #     value_list.append(torch.max(sim))
        #
        # elif measure_method == 'average':
        #     value_list.append(torch.mean(sim))

    # if measure_method == 'max':
    #     return max(value_list)
    # if measure_method =='average':
    #     return sum(value_list)/len(value_list)

    # return sum(value_list) / len(value_list)
    return value_list




def euclidean(measure_method, E=None,U=None, w=None, ):
    import torch.nn.functional as F

    # print(E.shape, U.shape)
    value_list =[]
    for idx in range (U.shape[0]):
        sample = U[idx]
        sample = torch.unsqueeze(sample,dim=0)
        # print(sample.shape)
        start_idx = idx*w
        end_idx = start_idx+w
        features = E[start_idx:end_idx,:]
        features = F.normalize(features,dim=1)
        sample = F.normalize(sample)
        sim = torch.zeros((len(features),1))
        for i in range (len(sim)):
            sim[i] = torch.sum(torch.square(features[i] - sample[0]))
            sim[i] = torch.sqrt(sim[i])

        value_list.append(torch.max(sim))
        # value_list.append(torch.mean(sim))
        # if measure_method == 'max':
        #     value_list.append(torch.max(sim))
        #
        # elif measure_method == 'average':
        #     value_list.append(torch.mean(sim))

    # if measure_method == 'max':
    #     return max(value_list)
    # if measure_method =='average':
    #     return sum(value_list)/len(value_list)

    # return sum(value_list) / len(value_list)
    return max (value_list)



def euclidean_testing(th, E=None, U=None, w=None):
    import torch.nn.functional as F
    anomaly=[]
    U = F.normalize(U,dim=1)
    E = F.normalize(E,dim=1)
    for idx in range(U.shape[0]):
        sample = U[idx]
        sample = torch.unsqueeze(sample, dim=0)
        start_idx = idx * w
        end_idx = start_idx + w
        features = E[start_idx:end_idx, :]

        sim = torch.zeros((len(features),))
        for i in range(len(sim)):
            sim[i] = torch.sum(torch.square(features[i] - sample[0]))
            sim[i] = torch.sqrt(sim[i])
            if sim[i] >= th:
                sim[i]=1.
            else: sim[i] =0.
        anomaly.append(sim)
    return anomaly


def F1_PA(groundtrue, predicted,delay):
    start_anomalies=[]
    end_anomalies=[]
    predicted = np.array(predicted)

    #find anomaly range
    for i in range (len(groundtrue)-1):
        if groundtrue[i+1] ==1. and groundtrue[i]==0:
            start_anomalies.append(i+1)
        if groundtrue[i] ==1. and groundtrue[i+1] ==0.:
            end_anomalies.append(i+1)

    print(start_anomalies)
    print(end_anomalies)

    # predicted = np.asarray(predicted)
    # print('len predict')
    # print(len(predicted))
    #
    # print('len ground-true')
    # print(len(groundtrue))

    # breakpoint()

    if delay ==None:

        # adjust predicted values
        for k in range (len(start_anomalies)):
            for j in range(start_anomalies[k], end_anomalies[k] + 1):
                if predicted[j] == 1.:
                    print('here')
                    np.put(predicted, np.arange(start_anomalies[k], end_anomalies[k] + 1, 1), 1.)
                    break
    else:
        for k in range (len(start_anomalies)):
            for j in range(start_anomalies[k], start_anomalies[k]+delay):
                if predicted[j] == 1.:
                    np.put(predicted, np.arange(start_anomalies[k], end_anomalies[k] + 1, 1), 1.)
                    break



    # plt.plot(groundtrue)
    # plt.title('ground true labels')
    # plt.ylabel('labels')
    # plt.xlabel('time point')
    # plt.show()
    #
    # plt.plot(predicted)
    # plt.title('predicted labels')
    # plt.ylabel('labels')
    # plt.xlabel('time point')
    # plt.show()

    # score = F1_score(predicted,groundtrue)
    # score = f1_score(groundtrue, predicted)
    score, precision, recall = F1_score(predicted, groundtrue)
    return score,precision, recall




def pca (output=None,data_features=None, w=None, s=None, dimension=2,ds_name=None):
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 7
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    p = PCA(n_components=dimension)

    output = output.cpu().detach().numpy()
    data_features = data_features.cpu().detach().numpy()
    output = StandardScaler().fit_transform(output)
    data_features = StandardScaler().fit_transform(data_features)
    pca_output = p.fit_transform(output)
    pca_data_features = p.fit_transform(data_features)
    pca_output = torch.tensor(pca_output)
    pca_data_features = torch.tensor(pca_data_features)


    #generate target for plotting
    target = torch.arange(0,s,1)
    target = target.repeat_interleave(w)
    target2 = torch.arange(0,s,1)

    new_tensor = torch.column_stack((pca_output, target))
    new_tensor2 = torch.column_stack((pca_data_features,target2))

    new_tensor = new_tensor.numpy()
    new_tensor2 = new_tensor2.numpy()


    for i in range (0, len(new_tensor), w):
        x = new_tensor[i:i+w, 0]
        y = new_tensor[i:i+w,1]
        plt.scatter(x,y, label='E of U'+str(int(i/w)))

    for j in range (0, len(new_tensor2)):
        x = new_tensor2[j, 0]
        y = new_tensor2[j,1]
        plt.scatter(x,y, label='U_'+str(j))

    plt.legend()
    plt.title('Power Demand Dataset')
    plt.show()


def F1_score(reconstruced_labels ,anomaly_labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    print(len(reconstruced_labels), len(anomaly_labels))

    if len(anomaly_labels)- len(reconstruced_labels) ==1:
        anomaly_labels = np.delete(anomaly_labels, -1)


    # calculate TP,FP,TN,FN
    for index, value in enumerate(anomaly_labels):
        if value == 1.:
            if reconstruced_labels[index] == 1:
                TP += 1
            elif reconstruced_labels[index] == 0.:
                FP += 1
        elif value == 0.:
            if reconstruced_labels[index] == 0:
                TN += 1
            elif reconstruced_labels[index] == 1.:
                FN += 1

    if (TP+FP)==0 or (TP+FN)==0:
        return 0,0,0
    else:
        Precision = TP /(TP +FP)
        Recall = TP /(TP +FN)

        if (Precision +Recall)==0:
            return 0, 0, 0
        else:
            F_score = ( 2* Precision *Recall) / (Precision +Recall)
            return F_score, Precision,Recall



def mahalanobis(E=None,U=None, w=None):
    max_distance = 0
    import numpy as np
    import torch.nn.functional as F

    print(E.shape, U.shape)

    for idx in range (U.shape[0]):
        sample = U[idx]
        sample = torch.unsqueeze(sample,dim=0)
        start_idx = idx*w
        end_idx = start_idx+w
        features = E[start_idx:end_idx,:]
        features = torch.cat((sample,features),dim=0)

        features = F.normalize(features,dim=1)
        # sample = F.normalize(sample,dim=1)
        dist = np.zeros((len(features), 1))

        features = features.cpu().numpy()

        mean = np.mean(features, axis=0)
        differences = np.zeros_like(features)

        differences[:] = features[:] - sample

        features = np.transpose(features)
        covM = np.cov(features, bias=False)
        invCovM = np.linalg.inv(covM)



        for i in range (len(dist)):
            v = np.dot(differences[i], invCovM)
            k = np.dot(v, np.transpose(differences[i]))
            k = np.sqrt(k)
            dist[i] = k

        max_distance = max (max_distance, np.amax(dist))
    return max_distance


def mahalanobis_testing(E=None, U=None, w=None, th=1.):
    max_distance = 0
    anomaly =[]
    import numpy as np
    import torch.nn.functional as F

    print(E.shape, U.shape)

    for idx in range(U.shape[0]):
        sample = U[idx]
        sample = torch.unsqueeze(sample, dim=0)
        start_idx = idx * w
        end_idx = start_idx + w
        features = E[start_idx:end_idx, :]
        features = torch.cat((sample, features), dim=0)

        features = F.normalize(features, dim=1)
        # sample = F.normalize(sample,dim=1)
        dist = np.zeros((len(features)-1, 1))

        features = features.cpu().numpy()

        mean = np.mean(features, axis=0)

        differences = np.zeros_like(features)

        differences[:] = features[:] - sample
        differences = differences[1:, :]

        features = np.transpose(features)
        covM = np.cov(features, bias=False)
        invCovM = np.linalg.inv(covM)



        for i in range(len(dist)):
            v = np.dot(differences[i], invCovM)
            k = np.dot(v, np.transpose(differences[i]))
            k = np.sqrt(k)
            dist[i] = k

            if dist[i]>= th:
                dist[i] = 1.
            else:
                dist[i] =0.


        anomaly.append(dist)
    return anomaly




