from flask import Flask, render_template, request, jsonify
from recom import Recom
import numpy as np
import datrie as dtr

app = Flask(__name__)
R = Recom()


trie = dtr.Trie.load('./data/hints.trie')

def reform(query):
    return query.upper().strip()

def get_suggestion(query, source=trie):
    print('search for', query)
    alphabet = []
    with open('./data/alphabet.txt', 'r') as fr:
        for line in fr:
            curr = line[:-1]
            alphabet.append(curr)
    new_query = reform(query)
    if new_query == '':
        return None
    result = []
    result = source.keys(new_query)
    result = result[:5]
    return result

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/by_desrciption', methods=['GET'])
def by_desrciption():
    return render_template('by_description.html')

@app.route('/by_scores', methods=['GET'])
def by_score():
    return render_template('by_scores.html', films=R.get_random_films())

@app.route('/by_desrciption/_autocomplete', methods=['GET', 'POST'])
def autocomplete():
    if request.method == 'GET':
      print('gEt')
    if request.method == 'POST':
      print('post')
    print(request.args)
    res = get_suggestion(request.args.get('q', ''))
    return jsonify(matching_results=res)

@app.route('/predict_by_desc', methods=['POST'])
def predict_by_desc():
    pos_data = request.form['autocomplete']
    if pos_data == '':
        return render_template('No_string.html')
    result = R.get_recom(request=pos_data)
    if result is not None:
        return render_template('predict.html', pos_movies_entered=pos_data, recommendations=result)
    else:
        return render_template('No_string.html')


@app.route('/predict_by_scores', methods=['POST'])
def predict_by_scores():
    rec_vector = np.zeros((R.get_movies_shape(),))
    for key, value in request.form.items():
        if value != '':
            rec_vector[int(key)] = float(value)
    if len(rec_vector.nonzero()[0]) == 0:
        return render_template('No_string.html')
    else:
        try:
            result = R.get_recom_by_films(rec_vector)
            return render_template('predict_by_scores.html', result=result)
        except:
            return render_template('No_string.html')

if __name__ == '__main__':
    app.run()

