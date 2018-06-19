from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask import jsonify
import urllib
import urllib2, base64
import json
import datetime

import csv 
import tensorflow as tf 

app = Flask(__name__)
api = Api(app)



request_app_dummy = {
    "deliveries": [
        {
            "packages": [
                {
                    "address": {
                        "zip": "3011TP",
                        "number": 68,
                        "streetname": "Wijnstraat"
                    },
                    "meta": {
                        "weight": 15,
                        "size": 10,
                        "floor_num": 10,
                        "elevator_present": True,
                        "weather_conditions": 3
                    }
                }
            ],
            "_id": "ws685afad59c9068b80ecdd75db5",
            "isAtHome": True
        },{
            "packages": [
                {
                    "address": {
                        "zip": "3032RN",
                        "number": 108,
                        "streetname": "Vrouw-Jannestraat"
                    },
                    "meta": {
                        "weight": 15,
                        "size": 10,
                        "floor_num": 10,
                        "elevator_present": True,
                        "weather_conditions": 3
                    }
                }
            ],
            "_id": "vjs1085afad59c9068b80ecdd75db5",
            "isAtHome": True
        },
        {
            "packages": [
                {
                    "address": {
                        "zip": "3021KE",
                        "number": 3,
                        "streetname": "Schermlaan"
                    },
                    "meta": {
                        "weight": 15,
                        "size": 10,
                        "floor_num": 10,
                        "elevator_present": True,
                        "weather_conditions": 3
                    }
                }
            ],
            "_id": "sl35afad59c9068b80ecdd75db5",
            "isAtHome": True
        },
        {
            "packages": [
                {
                    "address": {
                        "zip": "3011AB",
                        "number": 1,
                        "streetname": "Bulgersteyn"
                    },
                    "meta": {
                        "weight": 15,
                        "size": 10,
                        "floor_num": 10,
                        "elevator_present": True,
                        "weather_conditions": 3
                    }
                }
            ],
            "_id": "bs15afad59c9068b80ecdd75db5",
            "isAtHome": True
        },
        {
            "packages": [
                {
                    "address": {
                        "zip": "3011VW",
                        "number": 5,
                        "streetname": "Jufferkade"
                    },
                    "meta": {
                        "weight": 15,
                        "size": 10,
                        "floor_num": 10,
                        "elevator_present": True,
                        "weather_conditions": 3
                    }
                }
            ],
            "_id": "jk55afad59c9068b80ecdd75db5",
            "isAtHome": True
        },
        {
            "packages": [
                {
                    "address": {
                        "zip": "3014PX",
                        "number": 248,
                        "streetname": "Gouvernestraat"
                    },
                    "meta": {
                        "weight": 15,
                        "size": 10,
                        "floor_num": 10,
                        "elevator_present": True,
                        "weather_conditions": 3
                    }
                }
            ],
            "_id": "gs2485afad59c9068b80ecdd75db5",
            "isAtHome": True
        }
    ]
}



def convert_to_coords (streetname, house_nr, zip_letters, zip_numbers):
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + streetname + '+' + str(house_nr) + ',+' + zip_letters + '+' + zip_numbers + '&key=AIzaSyBf7NgNQh4WKI9fbkNlvIKNYQLWWwy47h8'
    response = json.load(urllib2.urlopen(url))
    return response["results"][0]["geometry"]["location"]

def calc_travel_time(coords_origin, coords_destination):
    url = "https://maps.googleapis.com/maps/api/directions/json?origin=" + coords_origin + "&destination=" + coords_destination + "&key=AIzaSyD_8EG1LJWQ8RmBpcZUb_2gF3fdyR7C9U8"
    response = json.load(urllib2.urlopen(url))
    return response["routes"][0]["legs"][0]["duration"]["value"]

max_floors = 44

def calc_building_difficulty(floor_number, elevator_present, weight_box):
    building_difficulty = 0

    if floor_number > 3:
        elevator_present = True

    if elevator_present == True:
        building_difficulty = (float(max_floors) / float(100)) * float(floor_number)
    else:
        mp = 0

        if weight_box > 20:
            mp = 2
        else:
            mp = 1.5

        building_difficulty = ((float(max_floors) / float(100)) * float(floor_number)) * (mp)

    return building_difficulty

def calc_time_meta(weight, size, floor_number, elevator_present, weather):
    time = 0

    training_set = []
    training_set_y = []

    with open("./data/training-data.csv","rb") as file:
        reader = csv.reader(file)
        for row in reader:
            training_set.append([row[1],row[2],row[3],row[4]])
            training_set_y.append(row[5])

    # Excluding the first column from the list (which is nothing but name column)
    training_set = training_set[1:]
    training_set_y = training_set_y[1:]

    testing_set = []

    # Preparing Test data set
    # Extracting each parameter into different list.
    # with open("./data/test-data.csv","rb") as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         testing_set.append([row[1],row[2],row[3],row[4]])

    building_difficulty = calc_building_difficulty(floor_number, elevator_present, weight)

    testing_set.append([weight, size, building_difficulty, weather])

    # Excluding the first column from the list (which is nothing but name column)
    # testing_set = testing_set[1:]

    # Placeholder you can assign values in future its kind of a variable
    training_values = tf.placeholder("float",[None,len(training_set[0])])
    test_values     = tf.placeholder("float",[len(training_set[0])])

    # This is the distance formula to calculate the distance between the test values and the training values
    distance = tf.reduce_sum(tf.abs(tf.add(training_values,tf.negative(test_values))),reduction_indices=1) 	

    # Returns the index with the smallest value across dimensions of a tensor
    prediction = tf.arg_min(distance,0)


    # Initializing  the session
    init = tf.initialize_all_variables()

    # Starting the calculation process
    # For every test sample, the above "distance" formula will get called and the distance formula will return the 
    # distances from the traning set values to the test sample and then the "prediction" will return the smallest 
    # distance index.
    with tf.Session() as sess:
        sess.run(init)	

        # Looping through the test set to compare against the training set
        for i in range (len(testing_set)):
            # Tensor flow method to get the prediction nearer to the test parameters from the training set.
            index_in_trainingset = sess.run(prediction,feed_dict={training_values:training_set,test_values:testing_set[i]})	

            print "==========="
            print "TEST RECORD | Weight box (kg): " + str(testing_set[i][0]) + " | Size box (cm3): " + str(testing_set[i][1]) + " | Accessibility score: " + str(testing_set[i][2]) + " | Weather conditions: " + str(testing_set[i][3]) + " | PREDICTION: " + str(training_set_y[index_in_trainingset]) + " SECONDS"
            print "CLOSEST RECORD | Weight box (kg): " + str(training_set[index_in_trainingset][0]) + " | Size box (cm3): " + str(training_set[index_in_trainingset][1]) + " | Accessibility score: " + str(training_set[index_in_trainingset][2]) + " | Weather conditions: " + str(training_set[index_in_trainingset][3]) + " | Total time (s): " + str(training_set_y[index_in_trainingset])
            # print "CLOSEST RECORD | Employee: " + str(training_set[index_in_trainingset][0]) + " | Pacakge size (cm3): " + str(training_set[index_in_trainingset][1]) + " | Distace to door (m): " + str(training_set[index_in_trainingset][2]) + " | Total time (s): " + str(training_set_y[index_in_trainingset])
            print "==========="

            time = training_set_y[index_in_trainingset]

    return time



# foo = calc_time_meta(10, 10, 5)

# print foo


@app.route("/", methods=['GET'])
def hello():
    return "(y)"

@app.route('/fedex_ai', methods=['POST'])
def fedex_ai():
    if request.method == 'POST':
        request_app = request.get_json()

        routexl_locations = []
        package_id_dictionary = {}

        # Add departure address (depot)
        depot = {}
        depot["address"] = "Wijnhaven 99, 3011WN, Rotterdam, The Netherlands"
        depot["lat"] = "51.9174221"
        depot["lng"] = "4.4828617"
        routexl_locations.append(depot)

####### Create RouteXL request. #######
        for delivery in request_app["deliveries"]:
            if delivery.get("isAtHome", True):
                routexl_location = {}

                address = delivery["packages"][0]["address"]
                meta = delivery["packages"][0]["meta"]
                routexl_location["address"] = address["street"] + " " + str(address["number"]) + ", " + address["zip"] + ", Rotterdam, The Netherlands"
                
                coords = convert_to_coords(address["street"], address["number"], address["zip"][4:], address["zip"][:4])
                routexl_location["lat"] = str(coords["lat"])
                routexl_location["lng"] = str(coords["lng"])

                routexl_locations.append(routexl_location)

                # Add location to dictionary with its lat/lng as a unique key.
                package_id_key = str(coords["lat"]) + str(coords["lng"])
                package_id_dictionary[package_id_key] = {}
                package_id_dictionary[package_id_key]["_id"] = delivery["_id"]
                package_id_dictionary[package_id_key]["meta_time"] = calc_time_meta(meta["weight"], meta["size"], meta["floor_num"], meta["elevator_present"], meta["weather_conditions"])
                # package_id_dictionary[package_id_key]["meta_time"] = 5#calc_time_meta(10, 10, 5)
                # package_id_dictionary[package_id_key]["meta_time"] = 10

        print json.dumps(routexl_locations)

####### Send RouteXL request. #######
        # data = urllib.urlencode({"locations": '[{"address":"The Hague, The Netherlands","lat":"52.05429","lng":"4.248618"},{"address":"The Hague, The Netherlands","lat":"52.076892","lng":"4.26975"},{"address":"Uden, The Netherlands","lat":"51.669946","lng":"5.61852"},{"address":"Sint-Oedenrode, The Netherlands","lat":"51.589548","lng":"5.432482"}]'})
        data = urllib.urlencode({"locations": json.dumps(routexl_locations)})
        req = urllib2.Request('https://api.routexl.nl/v2/tour/', data)
        base64string = base64.b64encode('%s:%s' % ("Burn123", ""))
        req.add_header("Authorization", "Basic %s" % base64string) 
        response_route_xl = json.loads(urllib2.urlopen(req).read())
        print "--------------"
        print response_route_xl

####### Format RouteXL response in custom object. #######
        response_final = {}
        response_final["deliveries"] = []
        coords_depot= "51.9174221,4.4828617"#
        previous_coords = 0
        timestamp_start_day = request_app["timestamp_start_day"] / 1000#1528441200 # June 7th 2018, 09:00
        timestamp_last_delivery = 0;

        # Iterate trough sorted deliveries, calculate ETA.
        for i in range(1, response_route_xl["count"]):
            delivery = {}
            delivery["order"] = i
            delivery["address"] = response_route_xl["route"][str(i)]["name"]
            delivery["coordinates"] = {}
            lat = response_route_xl["route"][str(i)]["lat"]
            lng = response_route_xl["route"][str(i)]["lng"]
            delivery["coordinates"]["lat"] = lat
            delivery["coordinates"]["lng"] = lng
            delivery["_id"] = package_id_dictionary.get(str(lat) + str(lng), "none")["_id"]

            coords_destination = response_route_xl["route"][str(i)]["lat"] + "," + response_route_xl["route"][str(i)]["lng"]
            
            if i == 1:
                time = calc_travel_time(coords_depot, coords_destination)
                delivery["eta_timestamp"] = timestamp_start_day + time + int(package_id_dictionary.get(str(lat) + str(lng), "none")["meta_time"])
                delivery["time"] = datetime.datetime.fromtimestamp(delivery["eta_timestamp"]).strftime('%H:%M:%S %d-%m-%Y')
                timestamp_last_delivery = delivery["eta_timestamp"]
                previous_coords = coords_destination
            else:
                time = calc_travel_time(previous_coords, coords_destination)
                delivery["eta_timestamp"] = timestamp_last_delivery + time + int(package_id_dictionary.get(str(lat) + str(lng), "none")["meta_time"])
                delivery["time"] = datetime.datetime.fromtimestamp(delivery["eta_timestamp"]).strftime('%H:%M:%S %d-%m-%Y')
                timestamp_last_delivery = delivery["eta_timestamp"]
                previous_coords = coords_destination

            response_final["deliveries"].append(delivery)

####### RETURN JSON RESPONSE #######
        return json.dumps(response_final)

if __name__ == '__main__':
     app.run()