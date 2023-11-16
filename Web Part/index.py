from flask import Flask
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def homepage():
    UPLOAD_FOLDER = "ML part/inputs/uploads/"
    if request.method == "POST":
        f = request.files["uploadedImage"]
        file_name = secure_filename(f.filename)
        f.save(f"ML part/inputs/uploads/{file_name}")

    return render_template("home.html")


if __name__ == "__main__":
    app.run(port=1234, debug=True)
