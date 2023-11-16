import json
import numpy as np
import pandas as pd
from pyngrok import ngrok
from flask_cors import CORS, cross_origin
# from IPython.display import display, HTML
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, send_from_directory

data = pd.read_csv('dataset\trainn.csv', index_col = False, on_bad_lines = 'skip', sep = ',')

doenca = ['Dengue', 'Zika', 'Chikungunya', 'Yellow Fever', 'Malaria']
mascara = data['prognosis'].isin(doenca)

data = data[mascara]

data = data.drop(['microcephaly', 'pleural_effusion', 'rigor', 'bitter_tongue',
                  'jaundice', 'cocacola_urine', 'prostraction', 'paralysis', 'toenail_loss',
                  'lips_irritation', 'itchiness', 'ulcers', 'speech_problem', 'hypoglycemia', 'anemia', 'yellow_skin',
                  'ascites', 'bullseye_rash', 'coma', 'hypotension', 'lymph_swells', 'toe_inflammation', 'finger_inflammation',
                  'breathing_restriction', 'confusion', 'hyperpyrexia', 'tremor', 'irritability', 'stiff_neck', 'convulsion'], axis = 1)

data['muscle_pain'] = data[['muscle_pain', 'myalgia']].max(axis = 1 )
data = data.drop(['myalgia'], axis = 1)

rotulos = data.pop('prognosis')
dados = data

x = dados
y = rotulos

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.2, random_state = 42)

rforest = RandomForestClassifier(n_estimators = 501, random_state = 42)
rforest.fit(x_train, y_train)

def analise_sintomas(sintomas):
  previsao = rforest.predict(sintomas)

  return previsao[0]

from itertools import count
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEARDERS'] = 'Content-Type'


@app.route('/process_data', methods=['POST'])
def prever():

  sintomas = request.get_json()

  contar = sum(linha.count(1) for linha in sintomas)

  if contar >= 5:
    resultado = analise_sintomas(sintomas)
    resposta = {
        "dtc": resultado
    }
    return jsonify(resposta)
  else:
    resposta = {
        "dtc": "Inconclusivo"
    }
    return jsonify(resposta)

if __name__ == '__main__':
    ngrok.set_auth_token('2Xu7rJiXarrPW0kHcMsnYDEVx3q_6UdytjToSFYxrtotPV9Ur')

    public_url = ngrok.connect(5000)
    print(f' * Porta {public_url}')
    app.run()