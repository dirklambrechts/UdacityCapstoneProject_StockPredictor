"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

import Stock_price_prediction_web_application.views
