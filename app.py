from flask import Flask, render_template, request, send_file, flash
from werkzeug.utils import redirect

app = Flask(__name__)

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route('/concat', methods=['POST'])
def concat():
  import StringIO
  import numpy as np
  from PIL import Image
  from skimage.io import imsave

  if 'image' not in request.files:
    imagefile = Image.open('static/img/1.tif')
  else:
    imagefile = Image.open(request.files['image'])
    arr = np.asarray(imagefile)
    for i in range(3):
      arr = np.concatenate((arr, np.rot90(imagefile, i + 1)), axis=1)

    strIO = StringIO.StringIO()
    imsave(strIO, arr, plugin='pil', format_str='png')
    strIO.seek(0)
    return send_file(strIO, mimetype='image/png')

if __name__ == "__main__":
  app.secret_key = 'key'
  app.run(debug=True)