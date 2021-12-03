from flask import Flask,request
import util

app = Flask(__name__)


@app.route('/get_feature_names')
def get_feature_names():
    response = {'Features' : util.show_feature_names()}
    return response

@app.route('/predict_car_price',methods=['POST'])
def predict_car_price():

    age = request.form['Age']
    kms = request.form['Kms']
    mlg = request.form['Mileage']
    engine = request.form['Engine']
    pwr = request.form['Power']
    sts = request.form['Seats']
    own = request.form['Owner']
    trans = request.form['Transmission']
    fuel = request.form['Fuel']
    city = request.form['City']
    car = request.form['Car']

    response = {'Price' : util.predict_price(age,kms,mlg,engine,pwr,
                                             sts,own,trans,fuel,city,car)}
    return response
    


app.run()
