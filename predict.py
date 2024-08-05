from tensorflow.keras.models import load_model

CNNmodel = load_model('model/CNN7.26_.h5')
CNNmodel.load_weights('model/CNN7.26_.ckpt')
RESmodel = load_model('model/RESNET7.26_.h5')
RESmodel.load_weights('model/RESNET7.26_.ckpt')


import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.core import _read_segy
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

import pandas as pd
import numpy as np


#def normalization_one(data, axis,colinde):
def normalization_one(data, colinde):
    standardized_data=np.copy(data)
    for i in colinde:
        tem=data[:,i]
        minval = np.min(tem)
        maxval = np.max(tem)
        epsilon = 1e-8
        standardized_data[:,i] = (tem - minval) / (maxval - minval + epsilon)
    # 计算均值和标准差
#     minval = np.min(data, axis=axis, keepdims=True)
#     maxval = np.max(data, axis=axis, keepdims=True)
#     #std = np.std(data, axis=axis, keepdims=True)
#     # 防止除以0
#     #std[std == 0] = 1
#     # 标准化
#     epsilon = 1e-8
#     standardized_data = (data - minval) / (maxval - minval + epsilon)

    return standardized_data  


def normalization_(data, axis):
    # 计算均值和标准差
    minval = np.min(data, axis=axis, keepdims=True)
    maxval = np.max(data, axis=axis, keepdims=True)
    # 防止除以0
    #std[std == 0] = 1
    # 标准化
    epsilon = 1e-8
    standardized_data = (data - minval) / (maxval-minval+epsilon)
    return standardized_data  

# 读取Excel文件中的所有工作表

DATA1=np.load('DATA_lianghe_mulclas.npy',allow_pickle=True)

print(np.max(DATA1))
#Data_all=np.load('lianghe/Data_all_3curve.npy')
#Data_all=np.load('../../2.28/data_all/Data_all_3curve.npy')
Label_all=DATA1[:,-1]
Data_all=DATA1[:,:-1]
#Label_all[Label_all==3]=2

# plt.plot(Data_all[:,2])
# plt.plot(Data_all[:,3])
# plt.plot(Data_all[:,4])
# plt.show()

def cutdata(data1,label1):
    data0=[]
    label0=[]
    for i in range(20,len(data1)-20):
        tem=data1[i-20:i+20,:]
        data0.append(tem)
        label0.append(label1[i])
    return np.array(data0),np.array(label0)

fnameunique=np.unique(Data_all[:,0])
print(fnameunique)


Thred=0.5
def plotwell(fname,Predi,label1):
    N=Predi.shape[0]
    Predi = np.argmax(Predi,axis=1)
    Pred2D=np.zeros((N,20))
    index1=np.where(Predi==0)
    Pred2D[index1,:]=0
    index1=np.where(Predi==1)
    Pred2D[index1,:]=1
    index1=np.where(Predi==2)
    Pred2D[index1,:]=2
    index1=np.where(Predi==3)
    Pred2D[index1,:]=3
    
    Label2D=np.zeros((N,20))
    index1=np.where(label1==0)
    Label2D[index1,:]=0
    index1=np.where(label1==1)
    Label2D[index1,:]=1
    index1=np.where(label1==2)
    Label2D[index1,:]=2
    index1=np.where(label1==3)
    Label2D[index1,:]=3
    
    
    # index1=np.where(Predi==2)
    # Pred2D[index1,:]=2

    fig = plt.figure(figsize=(14, 2))
    # specify color bar parameters
    cmap = 'Spectral'
    #COL = MplColorHelper(cmap, 0., 2.)
    #plt.pcolor(-Depth+alti,np.linspace(0,20,20),Pred2D.transpose(),cmap = 'PuBu_r')
    plt.pcolor(Depth,np.linspace(0,20,20),Pred2D.transpose(),cmap = 'rainbow')
    plt.colorbar(extend='max')
    plt.subplots_adjust(top=0.95,bottom=0.13,left=0.11,right=0.997)
    plt.savefig('out/'+str(fname)+'predi.jpg',dpi=(500.0))
    
    fig = plt.figure(figsize=(14, 2))
    # specify color bar parameters
    cmap = 'Spectral'
    #COL = MplColorHelper(cmap, 0., 2.)
    #plt.pcolor(-Depth+alti,np.linspace(0,20,20),Pred2D.transpose(),cmap = 'PuBu_r')
    plt.pcolor(Depth,np.linspace(0,20,20),Label2D.transpose(),cmap = 'rainbow')
    plt.colorbar(extend='max')
    plt.subplots_adjust(top=0.95,bottom=0.13,left=0.11,right=0.997)
    plt.savefig('out/'+str(fname)+'label.jpg',dpi=(500.0))
    


def savedata(Depth,Predi,fname):
    f=open('out/'+str(fname)+'.txt','w')
    f.write('Depth prediction\n')
    for i in range(len(Depth)):
        f.write('%f %f\n' %(Depth[i],Predi[i]))
    f.close()



def calc_acc(label,predi):
    index0=np.where(label==0)
    
    predi = np.argmax(predi,axis=1)
    index1=np.where(predi==0)
    Accurancy=np.sum(label[index0]==predi[index0])/len(predi[index1])
    Recall=np.sum(label[index0]==predi[index0])/len(label[index0])
    F1score=2*Accurancy*Recall/(Accurancy+Recall)
    print('Class 0 number: ',len(label[index0]),'  Accurancy: ',Accurancy)
    print('Class 0 number: ',len(label[index0]),'  Recall: ',Recall)
    print('Class 0 number: ',len(label[index0]),'  F1score: ',F1score)
    index0=np.where(label==1)
    index1=np.where(predi==1)
    Accurancy=np.sum(label[index0]==predi[index0])/len(predi[index1])
    Recall=np.sum(label[index0]==predi[index0])/len(label[index0])
    F1score=2*Accurancy*Recall/(Accurancy+Recall)
    print('Class 1 number: ',len(label[index0]),'  Accurancy: ',Accurancy)
    print('Class 1 number: ',len(label[index0]),'  Recall: ',Recall)
    print('Class 1 number: ',len(label[index0]),'  F1score: ',F1score)
    index0=np.where(label==2)
    index1=np.where(predi==2)
    Accurancy=np.sum(label[index0]==predi[index0])/len(predi[index1])
    Recall=np.sum(label[index0]==predi[index0])/len(label[index0])
    F1score=2*Accurancy*Recall/(Accurancy+Recall)
    print('Class 2 number: ',len(label[index0]),'  Accurancy: ',Accurancy)
    print('Class 2 number: ',len(label[index0]),'  Recall: ',Recall)
    print('Class 2 number: ',len(label[index0]),'  F1score: ',F1score)
    index0=np.where(label==3)
    index1=np.where(predi==3)
    Accurancy=np.sum(label[index0]==predi[index0])/len(predi[index1])
    Recall=np.sum(label[index0]==predi[index0])/len(label[index0])
    F1score=2*Accurancy*Recall/(Accurancy+Recall)
    print('Class 3 number: ',len(label[index0]),'  Accurancy: ',Accurancy)
    print('Class 3 number: ',len(label[index0]),'  Recall: ',Recall)
    print('Class 3 number: ',len(label[index0]),'  F1score: ',F1score)
    
    
    Accurancy=np.sum(label==predi)/len(predi)
    Recall=np.sum(label==predi)/len(label)
    F1score=2*Accurancy*Recall/(Accurancy+Recall)
    print('Class all number: ',len(label),'  Accurancy: ',Accurancy)
    print('Class all number: ',len(label),'  Recall: ',Recall)
    print('Class all number: ',len(label),'  F1score: ',F1score)
    




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


if __name__ == "__main__":
    try:
        start_time = time.time()
        parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
        parser.add_argument('-m', '--inputmodel', required=True, help='Input DL model file to execute')
        #parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

        args = parser.parse_args()

        fname = args.inputgeo
        #fnamePara = args.inputpara
        
    except:
        fname='RESNET'

    print('Input DL model file:',fname)
    
    


    DATA0=[]
    LABEL0=[]
    Predi0=[]
    for i in range(len(fnameunique)):
        
        if fnameunique[i] in [101,102,1701,3301]:
            print(fnameunique[i])
            indexpick=np.where(Data_all[:,0]==fnameunique[i])[0]
            data0=Data_all[indexpick,:]
            label0=Label_all[indexpick]
            Depth=data0[20:-20,1]
            #fig,ax = plt.subplots(figsize=(6, 4))
            # plt.plot(data0[:,2])
            # plt.plot(data0[:,3])
            # plt.plot(data0[:,4])
            # plt.show()
            data1_tem=normalization_one(data0,colinde=[2,3,4])
            data1_tem,label1=cutdata(data1_tem,label0)
            
            data1,label1=cutdata(data0,label0)
            data1=normalization_(data1,axis=1)
            data1=np.concatenate((data1,data1_tem[:,:,2:]), axis=2)
            
            if(fname=='CNN'):
                predi=CNNmodel.predict(data1[:,:,2:])
            elif(fname=='RESNET'):
                predi=RESmodel.predict(data1[:,:,2:])
            calc_acc(label1,predi)
            
            DATA0.append(data1)
            LABEL0.append(label1)
            Predi0.append(predi)
            
            #Depth=np.arange(0,data1.shape[0],1)
            print(predi.shape,Depth.shape)
            plotwell(fnameunique[i],predi,label1)
            #savedata(Depth,predi,fnameunique[i])

    DATA0=np.concatenate(DATA0, axis=0)
    LABEL0=np.concatenate(LABEL0, axis=0)
    Predi0=np.concatenate(Predi0, axis=0)


    print('all data calculation')
    calc_acc(LABEL0,Predi0)


