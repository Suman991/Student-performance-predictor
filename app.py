from flask import Flask, request ,render_template, redirect, url_for, flash
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application
app.secret_key='secret'

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        # Flash the prediction message
        flash(f'Your predicted Math Score: {results[0]}')
        return redirect(url_for('predict_datapoint'))
    

if __name__=="__main__":
    app.run(debug=True)   