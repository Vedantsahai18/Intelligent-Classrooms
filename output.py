import keras, os, pickle, ast
# import implicit
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from IPython.display import SVG
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.layers import Dense,Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.callbacks import ReduceLROnPlateau, History
from keras.regularizers import l1,l2
import seaborn as sns
sns.set()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = 0.000005), metrics=['categorical_accuracy'])

#useful variables
emo_order = [0,2,1,3,4,5]
pose_order = [1,0,2,3,4]

def _emoData():
    #emotion dataset
    df_emo = pd.read_csv('Expression.csv')
    df_emo = df_emo.drop(columns = ['_x', '_y', '_width', '_height', 'fearful', 'neutral', 'sad','surprised'])
    df_emo = df_emo[df_emo.columns[emo_order]]
    
    #this value calculates the maximum number of people in the frame to provide a value for each of them
    num_emo = df_emo['numPerson'].max()
    value = int(num_emo)
    
    if num_emo > 1:
        df_emo = df_emo.groupby(['personId', 'timestamp']).mean()
        df_emo = df_emo.reset_index()
        df_emo = df_emo.drop(columns = ['numPerson'])

    else:
        df_emo = df_emo.drop(columns = ['numPerson', 'personId'])
        df_emo = df_emo.groupby(['timestamp']).mean()
        
    return df_emo, value

def _poseData():
    #human pose dataset
    df_pose = pd.read_csv('Pose.csv')
    df_pose = df_pose.drop(columns = ['eyeCoordX', 'eyeCoordY'])
    df_pose = df_pose[df_pose.columns[pose_order]]
    
    num_pose = df_pose['numPersons'].max()
    
    if num_pose > 1:
        df_pose = df_pose.groupby(['personId', 'timestamp']).max()
        df_pose = df_pose.reset_index()
        df_pose = df_pose.drop(columns = ['numPersons'])
    
    else:
        df_pose = df_pose.drop(columns = ['numPersons', 'personId'])
        df_pose = df_pose.groupby(['timestamp']).max()
    
    return df_pose

def _gazeData():
    #head gaze dataset
    df_gaze = pd.read_csv('HeadGaze.csv')
    df_gaze = df_gaze.drop(columns = ['xCord', 'yCord', 'count'])
    #note, we dont need to reorder like the rest
    
    num_gaze = df_gaze['numPerson'].max()
    
    if num_gaze > 1:
        df_gaze = df_gaze.groupby(['personId', 'timestamp']).mean()
        df_gaze = df_gaze.reset_index()
        df_gaze = df_gaze.drop(columns = ['numPerson'])
        
    else:
        df_gaze = df_gaze.drop(columns = ['numPerson', 'personId'])
        df_gaze = df_gaze.groupby(['timestamp']).mean()
        
    return df_gaze


def _data():    
    df_emo, value = _emoData()
    df_pose = _poseData()
    df_gaze = _gazeData()
    
    # merge the different dataframes
    df_test = df_emo.merge(df_pose, on = 'timestamp', how = 'left')
    df_test = df_test.merge(df_gaze, on = 'timestamp', how = 'left')
    
    if value > 1:
        df_test = df_test.drop(columns = ['personId_y', 'personId'])
    
    #fill up the nan values
    df_test = df_test.fillna(method = 'bfill')
    
    #fill up any remaining Nan first
    df_test = df_test.fillna(method = 'ffill')
    df_test = df_test.fillna(0)
    
    if value > 1:
        df_test = df_test.groupby(['personId_x', 'timestamp']).mean()
    
    #currently indexed with the timestamp, change back to normal index and drop timestamp column
    df_test = df_test.reset_index()
    df_test = df_test.drop(columns = ['timestamp'])
    
    
    return df_test, value


data = {}
i = 0

def hello(model):
    global i
    # read csv from the bottom
    df_test, value = _data()
    
    
    # here we're looping through the 5 seconds to read each row one time 
    if value > 1:
        for k in range(value):
            #convert straight to numpy to feed in values
            try:
                array = df_test.iloc[(df_test.personId_x.values == k).argmax() + i].to_numpy()
            except:
                continue

            #remove personId_x column
            array = np.delete(array, 0)
            array = array.reshape(6,1).T
            try:
                B = loaded_model.predict_classes(array, verbose = 1) + 1
            except:
                B = 0

            data[k] = int(B)
    
    else:
        # change to numpy format
        A = df_test.iloc[i:i + 1,:6].to_numpy()
        #make prediction
        print()
        print(A)
        print("Happy: ", A[0][0])
        print("Looking away: ", A[0][-1])
        print("Raising Hand: ", A[0][-3])
        print("Sleeping : ", A[0][-2])

    #make prediction
    if value > 1:
        print(data)
        
    else:
        try:
            predictions = loaded_model.predict_classes(A, verbose = 1) + 1
        except:
            predictions = 0
        
        print(int(predictions))
        
    print("read csv, reading from second {}".format(i))
    i = i + 1
    
    #reset count every 5 seconds
    if i == 5:
        i = 0



# actual real-time predictions
import time
i = 0
count = 0
count_max = 100


nexttime = time.time()
while True:
    hello(loaded_model)          
    
    print("count number {}".format(count))
    
    #i want to loop it every second to output an engagement value every second, nexttime will be 1
    nexttime += 1
    sleeptime = nexttime - time.time()
    if sleeptime > 0:
        time.sleep(sleeptime)
    count += 1
    
    if count == count_max:
        print("CSV Reading Stopped")
        break