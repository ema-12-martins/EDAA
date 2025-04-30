from flask import Flask, render_template, jsonify, request
import os
from kdtree import get_recommendations  

app = Flask(__name__)

IMAGES_PER_PAGE = 5

@app.route('/')
@app.route('/page/<int:page>')
def clothing(page=1):
    images = sorted(os.listdir('static/images'))
    total_images = len(images)
    total_pages = (total_images - 1) // IMAGES_PER_PAGE + 1
    start = (page - 1) * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE
    images_paginated = images[start:end]
    return render_template('home.html', images=images_paginated, page=page, total_pages=total_pages)

@app.route('/recommendations/<selected_image>')
def recommendations(selected_image):
    recommended = get_recommendations(selected_image)
    return jsonify(recommended)

if __name__ == '__main__':
    app.run(debug=True)
