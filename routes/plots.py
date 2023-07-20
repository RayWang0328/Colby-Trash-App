from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from python.config import app
import csv

@app.route('/plotting', methods=['POST'])
def plots():

    with open('detections.csv', 'r') as read_file:
        first_line = read_file.readline()

    if not first_line:
        return jsonify({'message': 'Nothing in the file!'}), 400

    df = pd.read_csv('detections.csv')
    num_clust = 3
  
    scaler = StandardScaler()
    df[['longitude', 'latitude']] = scaler.fit_transform(df[['longitude', 'latitude']])

    kmeans = KMeans(n_clusters=num_clust, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['longitude', 'latitude']])

    item_types = df['type'].unique()
    palette = sns.color_palette('hls', len(item_types))
    color_dict = dict(zip(item_types, palette))

    images = []

    for i in range(num_clust):
       # plt.figure(figsize=(10, 5))
        sns.countplot(x='type', data=df[df['cluster']==i], palette=color_dict)
        plt.title(f'Frequency Plot for Cluster {i+1}')
        sio = BytesIO()
        plt.savefig(sio, format='png')
        plt.close()
        sio.seek(0)
        image = base64.b64encode(sio.read()).decode('utf-8')
        images.append(image)

    return jsonify({'images': images}), 200