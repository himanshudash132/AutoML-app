from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
from ydata_profiling import ProfileReport
from pycaret.classification import setup as class_setup, compare_models as class_compare_models, pull as class_pull, save_model as class_save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model

app = Flask(__name__)

# Global DataFrame
df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global df
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df.to_csv('sourcedata.csv', index=None)  # Saving uploaded file
            return redirect(url_for('profiling'))
    return render_template('upload.html')

@app.route('/profiling')
def profiling():
    global df
    if df is None:
        return redirect(url_for('upload'))

    profile = ProfileReport(df, title="Automated EDA", explorative=True)
    profile.to_file("static/profiling_report.html")  # Save to 'static' directory
    return render_template('profiling.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    global df
    if df is None:
        return redirect(url_for('upload'))
    
    target_column = request.form.get('target') if request.method == 'POST' else None
    if target_column:
        class_setup(df, target=target_column)
        setup_df = class_pull()
        best_model = class_compare_models()
        compare_df = class_pull()
        class_save_model(best_model, 'models/best_class_model.pkl')
        return render_template('classification.html', setup_df=setup_df.to_html(), compare_df=compare_df.to_html())
    return render_template('classification.html', columns=df.columns)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    global df
    if df is None:
        return redirect(url_for('upload'))
    
    target_column = request.form.get('target') if request.method == 'POST' else None
    if target_column:
        reg_setup(df, target=target_column)
        setup_df = reg_pull()
        best_model = reg_compare_models()
        compare_df = reg_pull()
        reg_save_model(best_model, 'models/best_reg_model.pkl')
        return render_template('regression.html', setup_df=setup_df.to_html(), compare_df=compare_df.to_html())
    return render_template('regression.html', columns=df.columns)

@app.route('/download')
def download():
    file_path = 'models/best_model.pkl'
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')

