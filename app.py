import os
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
from analysisTool import *

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__,static_folder='./images',)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/evaluation')
def evaluate_squad():
    out_eval = fire('dataset.json','pred_data.json')
    with open('./all.json') as f:
        data = json.load(f)
    image_names = os.listdir('./images')
    print(image_names)
    return render_template('evaluation.html',datalen =len(data), data={'out_eval': out_eval},dataJson=json.dumps(data))
    
@app.route('/',methods=['POST','GET'])
def home():
    if request.method =='POST':
        file1 = request.files['dataset']
        file2 = request.files['pred_data']

        if file1:
            filename = secure_filename(file1.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'],"dataset.json"))
            # return hello()
        if file2:
            filename = secure_filename(file2.filename)
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'],"pred_data.json"))
            #   return hello()
        else:
            return "MISSING: Error in uploading files"

        return evaluate_squad()
        
    return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True)