from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from api.p2 import p2_bp

app = Flask(__name__, template_folder='public', static_folder='public/static')
CORS(app)

app.register_blueprint(p2_bp, url_prefix="/api/p2")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/p2")
def p2():
    return render_template("p2.html")

@app.route("/p3")
def p3():
    return render_template("p3.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6767)

