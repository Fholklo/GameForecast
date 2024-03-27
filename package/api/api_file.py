from fastapi import FastAPI
import pandas as pd


from package.main import preprocess_test, load_most_recent_model
from package.scripts.params import folder_path
from package.api.scrapper import get_steam_game_info

app = FastAPI()

app.state.model_rating_num =load_most_recent_model(folder_path,model_name="model_rating",model_type='model_num')
app.state.model_rating_text =load_most_recent_model(folder_path,model_name="model_rating",model_type= 'model_text')
app.state.model_rating_image =load_most_recent_model(folder_path,model_name="model_rating",model_type= 'model_image')
app.state.model_rating_metamodel =load_most_recent_model(folder_path,model_name="model_rating",model_type= 'model_meta')

app.state.model_player_num =load_most_recent_model(folder_path,model_name="model_player",model_type='model_num')
app.state.model_player_text =load_most_recent_model(folder_path,model_name="model_player",model_type= 'model_text')
app.state.model_player_image =load_most_recent_model(folder_path, model_name="model_player",model_type='model_image')
app.state.model_player_metamodel =load_most_recent_model(folder_path,model_name="model_player",model_type= 'model_meta')

@app.get("/")
def root():
    return {'greetings':"hello"}

@app.get("/App_ID")
def Predict_on_App_ID(App_ID: int):
    #scrap des infos sur l'appid
    infos = get_steam_game_info(app_id=App_ID)
    infos=pd.DataFrame([infos])
    nom=infos["Name"]
    X_pred = infos.drop(columns=["Name","Header_Image"])
    X_pred["App_ID"]=App_ID
    numeric_input,text_input, images_input = preprocess_test(X_pred,train=False,api=True)

    #predict du rating
    preds_new_numeric_rating = app.state.model_rating_num.predict(numeric_input)
    preds_new_text_rating = app.state.model_rating_text.predict(text_input)
    preds_new_image_rating = app.state.model_rating_image.predict(images_input)

    X_meta_test_rating = {"base_pred_input1":preds_new_numeric_rating,
                   "base_pred_input2":preds_new_text_rating,
                   "base_pred_input3":preds_new_image_rating}

    prediction_rating = app.state.model_rating_metamodel.predict(X_meta_test_rating)
    #predict du player peak
    preds_new_numeric_player = app.state.model_player_num.predict(numeric_input)
    preds_new_text_player = app.state.model_player_text.predict(text_input)
    preds_new_image_player = app.state.model_player_image.predict(images_input)

    X_meta_test_player = {"base_pred_input1":preds_new_numeric_player,
                   "base_pred_input2":preds_new_text_player,
                   "base_pred_input3":preds_new_image_player}

    prediction_player = app.state.model_player_metamodel.predict(X_meta_test_player)

    return {"name":nom[0],
            "Prediction_rating":round(float(prediction_rating)*100,1),
            "Prediction_player":round(float(prediction_player)*100,1),
            "text":round(float(preds_new_text_rating)*100,1),
            "text_p":round(float(preds_new_text_player)*100,1),
            "num":round(float(preds_new_numeric_rating)*100,1),
            "num_p":round(float(preds_new_numeric_player)*100,1),
            "image":round(float(preds_new_image_rating)*100,1),
            "image_p":round(float(preds_new_image_player)*100,1)}
