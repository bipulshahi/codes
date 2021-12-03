import json
import pickle
import numpy as np


_data_columns = None
_model = None

def load_artifacts():
    print('Lading Artifacts....')

    global _data_columns
    global _model
    
    f = open('./columns.json','r')
    _data_columns = json.load(f)['features']
    f.close()

    f = open('./car_price.pickle','rb')
    _model = pickle.load(f)
    f.close()
    

def predict_price(age,kms,mlg,engine,pwr,sts,own,trans,fuel,city,car):
    input = np.zeros(len(_data_columns))

    input[0] = age
    input[1] = kms
    input[2] = mlg
    input[3] = engine
    input[4] = pwr
    input[5] = sts

    input[_data_columns.index(own)] = 1
    input[_data_columns.index(trans)] = 1
    input[_data_columns.index(fuel)] = 1
    input[_data_columns.index(city)] = 1
    input[_data_columns.index(car)] = 1

    return _model.predict([input])[0][0]


def show_feature_names():
    print(_data_columns)
    #print(_data_columns.index('Pune'))



load_artifacts()
#show_feature_names()
print(predict_price(10,60000,15,1356,95,5,'Second','Automatic','Diesel','Chennai','nissan terrano'))
