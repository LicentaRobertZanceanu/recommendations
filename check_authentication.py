from werkzeug.wrappers import Request, Response, ResponseStream
import requests
from flask import jsonify
import json

AUTH_API = 'https://smartmusicapi-auth.herokuapp.com/auth'


class CheckAuth():

    def __init__(self, app):
        self.app = app
        self.userId = ''

    def __call__(self, environ, start_response):
        request = Request(environ)
        if request.method == 'OPTIONS':
            return self.app(environ, start_response)
        
        auth_token = request.headers['Authorization']
        request_headers = {"Authorization": auth_token}

        requestResponse = requests.get(
            AUTH_API + "/is-authenticated", headers=request_headers)
        if requestResponse.status_code == 401:
            res = Response(
                u'{"message": "Access denied!"}', mimetype='application/json', status=401)
            return res(environ, start_response)

        responseContent = json.loads(requestResponse.content)
        environ['user'] = {"id": responseContent['_id']}

        return self.app(environ, start_response)
