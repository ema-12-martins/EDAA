from flask import Flask, render_template, jsonify
import os
import pandas as pd
from kdtree import get_recommendations  

app = Flask(__name__)
IMAGES_PER_PAGE = 5

# Load product data once at startup
PRODUCTS_CSV = 'styles_joined.csv'  # replace with your actual path
df = pd.read_csv(PRODUCTS_CSV)
df['id'] = df['id'].astype(str)
df.set_index('id', inplace=True)

@app.route('/')
@app.route('/page/<int:page>')
def clothing(page=1):
    all_images = sorted(os.listdir('static/images'))
    all_images_ids = [img for img in all_images if img.split('.')[0] in df.index]

    total_images = len(all_images_ids)
    total_pages = (total_images - 1) // IMAGES_PER_PAGE + 1
    page = max(1, min(page, total_pages))  # Clamp to valid page range

    start = (page - 1) * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE
    page_images = all_images_ids[start:end]

    images_info = []
    for img in page_images:
        img_id = img.split('.')[0]
        row = df.loc[img_id]
        price = row['price']
        discounted = row['discountedPrice']
        discount = None
        if pd.notnull(discounted) and discounted < price:
            discount = round(100 * (price - discounted) / price)
        images_info.append({
            'filename': img,
            'id': img_id,
            'price': price,
            'discounted': discounted,
            'discount_percent': discount
        })

    return render_template('home.html', images=images_info, page=page, total_pages=total_pages)

@app.route('/recommendations/<selected_image>')
def recommendations(selected_image):
    recommended_files = get_recommendations(selected_image)
    recommended_info = []

    for filename in recommended_files:
        img_id = filename.split('.')[0]
        if img_id in df.index:

            row = df.loc[img_id]
            price = row['price']
            discounted = row['discountedPrice']
            discount = None
            if pd.notnull(discounted) and discounted < price:
                discount = round(100 * (price - discounted) / price)
            recommended_info.append({
                'filename': filename,
                'id': img_id,
                'price': price,
                'discounted': discounted,
                'discount_percent': discount
            })
    return jsonify(recommended_info)

if __name__ == '__main__':
    app.run(debug=True)
