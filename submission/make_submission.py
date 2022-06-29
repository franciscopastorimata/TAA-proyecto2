import pickle
import sys

sys.path.append("../src")
from read_records import read_tf_test_records

output_submission_path = f"{os.path.dirname(__file__)}/output"

def create_submission(file_name, image_name, level):
    df_submission = pd.DataFrame({'image': image_name,'level':level.astype(int)})
    os.makedirs(output_submission_path, exist_ok=True)
    output_path = f"{output_submission_path}/{file_name}.csv"
    df_test = pd.read_csv(f'{INPUT_RAW_DATA_PATH}/test.csv')

if __name__ == '__main__':
    test_dataset = read_tf_test_records()
    test_dataset = test_dataset.map(preprocess)

    model = pickle.load(open(f"{INPUT_MODELS_PATH}/{model_name}.sav", 'rb'))
    create_submission(submission_name, df_test['image'], model.predict(X_test))
    print('the submission file has been successfully created')