# Making a microservice using Flask

from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/new')
def hello_world_new():
    return 'Hello, new world!'

@app.route('/printone/<int:the_num>')
def print_one(the_num):
    return jsonify(the_num)
#f'{the_num}'

# @app.route('/vars')
# def add_two(x,y):
#     return x+y


@app.route('/vars', methods=['GET'])
def foo():
   n1=int(request.args.get('n1'))
   n2=int(request.args.get('n2'))
   return jsonify(n1+n2)
