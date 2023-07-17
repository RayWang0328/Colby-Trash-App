from flask import render_template, request, session, jsonify
from python.config import app
import csv
from pyproj import Transformer
from ..script import calculate_GSD
import statistics
import folium

@app.route('/mapping/', methods=['POST'])
def mapping():

    boxes = []
    lats = []
    longs = []
    predictions = []
    # Open both CSV files
    with open('detections.csv', 'r') as csv_file1:
        # Create a CSV reader for both files
        reader1 = csv.reader(csv_file1)

        # Convert the readers to lists of rows
        rows1 = list(reader1)[1:]

        # Iterate over all pairs of rows
        for row1 in rows1:

            long, lat = float(row1[4]), float(row1[5])
            alt = float(row1[6])
            gsd = calculate_GSD(alt)
            bbox_pixels = [float(row1[0]),float(row1[1]),float(row1[2]),float(row1[3])]
            predictions.append(row1[9])


            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

            # Calculate the top left corner of the bounding box
            tl_lon, tl_lat = transformer.transform(gsd * bbox_pixels[0], gsd * bbox_pixels[1])

            long_temp, lat_temp = transformer.transform(gsd * 2736, gsd * 1824)

            ref_lon = long - long_temp
            ref_lat = lat - lat_temp


            tl_lon += ref_lon
            tl_lat += ref_lat

            # Calculate the bottom right corner of the bounding box
            br_lon, br_lat = transformer.transform(gsd * bbox_pixels[2], gsd * bbox_pixels[3])
            br_lon += ref_lon
            br_lat += ref_lat


            box1 = [tl_lon,tl_lat,br_lon,br_lat]
            boxes.append(box1)





    average_lat = statistics.mean(longs)
    average_long = statistics.mean(lats)

    tile = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = False,
            control = True
        ).add_to(m)

    m = folium.Map(location=[average_lat, average_long], zoom_start=15, max_zoom = 40, tiles=tile)

    
    for i, box1 in enumerate(boxes):

        if predictions[i] == "plastic": 
            color = '#dee619'
        elif predictions[i] == "cage":
            color = '#D900EB'
        elif predictions[i] == "wood": 
            color = '#FC6F15'
        elif predictions[i] == "fishing gear":
            color = '#e75480' # assumed color for wheel
        elif predictions[i] == "nature":
            color = '#00FF00' 
        elif predictions[i] == "metal":
            color = '#46473E' 
        elif predictions[i] == "wheel":
            color = '#110C0A' 

        folium.Rectangle(
            bounds=[[box1[0], box1[1]], [box1[2], box1[3]]],
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
        
        
    # Color mapping for the legend
    color_dict = {"metal": "#46473E", "wheel": "#110C0A", "plastic": "#dee619", "cage": "#D900EB", "wood": "#FC6F15", "nature": "#00FF00", "fishing gear": "#e75480"}

    # Add custom legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 120px; height: 130px; border:2px solid grey; z-index:9999; font-size:14px; color: white; background-color: rgba(0, 0, 0, 0.5);">
    &nbsp;<b>Legend:</b><br>
    '''

    for key, value in color_dict.items():
        legend_html += '&nbsp;<i class="fa fa-square fa-1x" style="color:{}"></i> {}<br>'.format(value, key)
    

    legend_html += '</div>'

    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()
