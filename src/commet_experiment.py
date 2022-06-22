from comet_ml import Experiment

def save_experiment_commet(name, model, historys, history, metrics):

    # Se crea un experimento utilizando nuestra API_KEY
    COMET_API_KEY = '8KM5gTaM4zSkcTrQ4BtwLOmFo'
    exp = Experiment(api_key=COMET_API_KEY,  # api_key=os.environ.get("COMET_API_KEY"),
                    project_name="taa-proyecto2",
                    workspace="fede-p")
    exp.set_name(name)
    historys.append(history)
    
    # Se guarda el modelo
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = os.path.join('models',name+'.h5')
    tf.keras.models.save_model(model, model_path)
    exp.log_model('nlp-rnn', model_path)

    exp.log_metrics(metrics)
    exp.end()

if __name__ == '__main__':
pass
