from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, template_folder='public', static_folder='public/static')
CORS(app)



