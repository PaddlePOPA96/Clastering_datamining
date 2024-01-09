from flask import Flask, render_template, request, url_for, redirect
from joblib import load
import pandas as pd

app = Flask(__name__)

# Mengeload data scaler dan data model yang sudah di generate waktu training tadi ya ges
model = load('kmeans_model.joblib')
scaler = load('scaler.joblib')

# setalah load data training kita mengeload dataset mentah kita untuk kita seleksai menggunakan scale dan model 
df = pd.read_csv('formresponden.csv', delimiter=';', on_bad_lines='skip', encoding='utf-8')

# Menambahkan kolom fomo kedalam dataset dari dataset model 
features = df[['jam_dalam_seminggu', 'X_Gadget', 'Y_Fomo']]
scaled_features = scaler.transform(features)
df['Fomo'] = model.predict(scaled_features)


# Inisialisasi router 
@app.route('/')
def index():
    return render_template('index.html')


# pdata prediction ini untuk memprediksi apakah kita fomo apa tidak 
@app.route('/predict', methods=['POST'])
def predict():
    jam_dalam_seminggu = request.form.get('jam_dalam_seminggu', type=float)
    X_Gadget = request.form.get('X_Gadget', type=float)
    Y_Fomo = request.form.get('Y_Fomo', type=float)
    features = scaler.transform([[jam_dalam_seminggu, X_Gadget, Y_Fomo]])
    cluster_label = model.predict(features)
    return redirect(url_for('result', cluster=int(cluster_label[0])))


@app.route('/result/<int:cluster>')
def result(cluster):
    title = 'FOMO Result' if cluster == 0 else 'Non-FOMO Result'
    return render_template('result.html', title=title, cluster=cluster)

@app.route('/iterations')
def iterations():
    iterations_data = model.n_iter_
    return render_template('iterations.html', iterations=iterations_data)

@app.route('/fomo_dataset')
def fomo_dataset():
    selected_columns = ['Nama', 'Jenis Kelamin', 'jam_dalam_seminggu', 'X_Gadget', 'Y_Fomo', 'Fomo']
    df_fomo = df[df['Fomo'] == 0][selected_columns]
    return render_template('fomo_dataset.html', dataset=df_fomo.to_html(classes='table table-striped', index=False))

@app.route('/non_fomo_dataset')
def non_fomo_dataset():
    selected_columns = ['Nama', 'Jenis Kelamin', 'jam_dalam_seminggu', 'X_Gadget', 'Y_Fomo', 'Fomo']
    df_non_fomo = df[df['Fomo'] == 1][selected_columns]
    return render_template('non_fomo_dataset.html', dataset=df_non_fomo.to_html(classes='table table-striped', index=False))


@app.route('/dataset')
def dataset():
    
    df_display = df[['Nama', 'Jenis Kelamin', 'Asal daerah', 'Usia', 'jam_dalam_seminggu', 'Sosial Media yang anda gunakan ', 'X_Gadget', 'Y_Fomo', 'Fomo']]
    return render_template('dataset.html', dataset=df_display.to_html(classes='table table-responsive table-striped', index=False, escape=False))


if __name__ == '__main__':
    app.run(debug=True)
