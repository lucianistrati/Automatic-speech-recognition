import mlflow
import json


class SentimentAnalysis(mlflow.pyfunc.PythonModel):
    def __init__(self):
        from transformers import pipeline
        self.nlp = pipeline('sentiment-analysis')
    def do_nlp_fnx(self, row):
        s = self.nlp(row['text'])[0]
        return [s['label'], s['score']]
    def predict(self, context, model_input):
        print('model_input=' + str(model_input), flush=True)
        model_input[['label', 'score']] =    model_input.apply(self.do_nlp_fnx, axis=1, result_type='expand')
        return model_input


inp = json.dumps([{'name': 'text', 'type': 'string'}])
outp = json.dumps([{'name': 'text', 'type':'string'},
                   {'name': 'label', 'type':'string'},
                   {'name': 'score', 'type': 'double'}])
signature = ModelSignature.from_dict({'inputs': inp, 'outputs': outp})

with mlflow.start_run():
    mlflow.pyfunc.log_model('model',
                            loader_module=None,
                            data_path=None,
                            code_path=None,
                            conda_env=None,
                            python_model=SentimentAnalysis(),
                            artifacts=None,
                            registered_model_name=None,
                            signature=signature,
                            input_example=None,              await_registration_for=0)