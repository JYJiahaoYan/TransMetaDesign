from flask import Flask, request, jsonify
import torch
from inference import Inference
from config import FTConfig, Config
from model_evalue_plot import double_ellipse_draw,double_rectangle_draw,circle_draw,rec_draw,cross_draw,lack_rec_draw,ring_draw
from model_evalue import model_predict
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__)

# Initialize Inference object
model_path = r"D:\\codes\\transformer_for_metasurface\\xxxx.ckpt"
config = Config()
inference = Inference(model_path, config)
forward_model_path = r"data/forward_model.pth"

@app.route('/inverse', methods=['POST'])
def inverse_predict():
    data = request.get_json()
    type = data.get('type')
    wave = data.get('wave')
    
    if not type or not wave:
        return jsonify({'error': 'Invalid input'}), 400
    
    try:
        predictions = inference.predict(type, wave)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/forward', methods=['POST'])
def forward_predict():
    data = request.get_json()
    type = data.get('type')
    paras = data.get('paras')

    if not type or not paras:
        return jsonify({'error': 'Invalid input'}), 400
    
    def _forward_predict(pic):
        try:
            pic = torch.tensor(pic).unsqueeze(0)
            predictions = model_predict(pic, forward_model_path)
            return jsonify({'predictions': predictions.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    if type == 'double_ellipse':
        A = paras.get('A')
        B = paras.get('B')
        a = paras.get('a')
        Px = paras.get('Px')
        Py = paras.get('Py')
        phi = paras.get('phi')
        if A is None or B is None or a is None or Px is None or Py is None or phi is None:
            return jsonify({'error': 'Invalid params'}), 400
        paras = [float(A), float(B), float(a), float(Px), float(Py), float(phi)]
        pic = double_ellipse_draw(paras)
        return _forward_predict(pic)
    elif type == 'double_rec':
        W1 = paras.get('W1')
        L1 = paras.get('L1')
        W2 = paras.get('W2')
        L2 = paras.get('L2')
        Px = paras.get('Px')
        Py = paras.get('Py')
        phi = paras.get('phi')
        if W1 is None or L1 is None or W2 is None or L2 is None or Px is None or Py is None or phi is None:
            return jsonify({'error': 'Invalid params'}), 400
        paras = [float(W1), float(L1), float(W2), float(L2), float(Px), float(Py), float(phi)]
        pic = double_rectangle_draw(paras)
        return _forward_predict(pic)
    elif type == 'ellipse':
        A = paras.get('A')
        B = paras.get('B')
        Px = paras.get('Px')
        Py = paras.get('Py')
        phi = paras.get('phi')
        if A is None or B is None or Px is None or Py is None or phi is None:
            return jsonify({'error': 'Invalid params'}), 400
        paras = [float(A), float(B), float(Px), float(Py), float(phi)] 
        pic = circle_draw(paras)
        return _forward_predict(pic)
    elif type == 'rec':
        L = paras.get('L')
        W = paras.get('W')
        Px = paras.get('Px')
        Py = paras.get('Py')
        phi = paras.get('phi')
        if L is None or W is None or Px is None or Py is None or phi is None:
            return jsonify({'error': 'Invalid params'}), 400
        paras = [float(L), float(W), float(Px), float(Py), float(phi)]
        pic = rec_draw(paras)
        return _forward_predict(pic)
    elif type == 'cross':
        W1 = paras.get('W1')
        L1 = paras.get('L1')
        W2 = paras.get('W2')
        L2 = paras.get('L2')
        offset = paras.get('offset')
        Px = paras.get('Px')
        Py = paras.get('Py') 
        phi = paras.get('phi')
        if W1 is None or L1 is None or W2 is None or L2 is None or offset is None or Px is None or Py is None or phi is None:
            return jsonify({'error': 'Invalid params'}), 400
        paras = [float(W1), float(L1), float(W2), float(L2), float(offset), float(Px), float(Py), float(phi)]
        pic = cross_draw(paras)
        return _forward_predict(pic)
    elif type == 'lack_rec':
        L = paras.get('L')
        W = paras.get('W')
        alpha = paras.get('alpha')
        beta = paras.get('beta')
        gama = paras.get('gama')
        Px = paras.get('Px')
        Py = paras.get('Py')
        phi = paras.get('phi')
        if L is None or W is None or alpha is None or beta is None or gama is None or Px is None or Py is None or phi is None:
            return jsonify({'error': 'Invalid params'}), 400
        paras = [float(L), float(W), float(alpha), float(beta), float(gama), float(Px), float(Py), float(phi)]
        pic = lack_rec_draw(paras)
        return _forward_predict(pic)
    elif type == 'ring':
        R = paras.get('R')
        r = paras.get('r')
        theta = paras.get('theta')
        phi = paras.get('phi')
        Px = paras.get('Px')
        Py = paras.get('Py')
        if R is None or r is None or theta is None or phi is None or Px is None or Py is None:
            return jsonify({'error': 'Invalid params'}), 400
        paras = [float(R), float(r), float(theta), float(phi), float(Px), float(Py)]
        pic = ring_draw(paras)
        return _forward_predict(pic)
    else:
        return jsonify({'error': 'Invalid type'}), 400



if __name__ == '__main__':
    app.run(debug=True,port=4180)