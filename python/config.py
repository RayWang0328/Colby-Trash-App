from flask import Flask, request, make_response
import pandas as pd

# init app.
csv_file = pd.DataFrame()
map = ''
application  = Flask(__name__, template_folder='../templates')