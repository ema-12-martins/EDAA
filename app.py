from flask import Flask, render_template, jsonify
import os
import random
from kdtree import get_recommendations


app = Flask(__name__)

@app.route('/')
def clothing():
    images = os.listdir('static/images')
    return render_template('home.html', images=images)

@app.route('/recommendations/<selected_image>')
def recommendations(selected_image):
    recommended = get_recommendations(selected_image)
    return jsonify(recommended)

if __name__ == '__main__':
    app.run(debug=True)