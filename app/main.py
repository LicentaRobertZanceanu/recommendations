from flask import Flask, request, jsonify, make_response
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from flask_cors import CORS
import json

from check_authentication import CheckAuth
from prepare_data import prepare_data
from nlp_recommendations import get_user_recommendations

app = Flask(__name__)
app.wsgi_app = CheckAuth(app.wsgi_app)

CORS(app)


def save_to_csv_from_url(url, csv_name):
    csvResponse = requests.get(url)
    print(csvResponse)
    csvResponseContent = csvResponse.content.decode('utf-8')
    with open(csv_name, 'wb') as file:
        for line in csvResponseContent:
            if line != '/n':
                file.write(line.encode('utf-8'))
    print('updated '+csv_name)
    return


def load_and_prepare_data():
    print('a intrat load')
    save_to_csv_from_url('https://smartmusic-licenta-bucket.s3.amazonaws.com/songs.csv',
                         'songs.csv')
    save_to_csv_from_url('https://smartmusic-licenta-bucket.s3.amazonaws.com/liked.csv',
                         'liked.csv')
    save_to_csv_from_url('https://smartmusic-licenta-bucket.s3.amazonaws.com/listened.csv',
                         'listened.csv')
    prepare_data()
    print('cron finished')

    return


@app.route("/recommendations", methods=["OPTIONS"])
def api_create_order():
    return build_cors_prelight_response()


def build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    result = get_user_recommendations(request.environ['user']['id'])

    return jsonify(result)


app.run(debug=True)

scheduler = BackgroundScheduler()
scheduler.add_job(
    load_and_prepare_data,
    'cron',
    day_of_week="mon-sun",
    hour=2,
    minute=0)
scheduler.start()

atexit.register(lambda: scheduler.shutdown())
