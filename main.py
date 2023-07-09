from enum import Enum
import os
import time
import subprocess
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
from roberta_run import RobertaInferer
from tfidf_run import TfidfInferer, LemmaTokenizer 



class TextRequest(BaseModel):
    text: str

app = FastAPI(title="Модели Roberta и Tfidf",
            version="0.0.1")

class App_Name(str, Enum):
    roberta = "roberta"
    tfidf = "tfidf"
    

path_roberta = os.getenv("PATH_ROBERTA")
path_tfidf = os.getenv("PATH_TFIDF")

roberta = RobertaInferer( path_to_model=path_roberta)
tfidf = TfidfInferer( path_to_model=path_tfidf)


with open("./s3_models.txt", "r") as file:
    read_file = file.read()
    models_titles = read_file.split("\n")
models_titles = [model.replace(".zip", "") for model in models_titles]

models: Dict[str, dict] = {
    "roberta": {"model": None, "get_data": None, "launch_function": roberta.infer,},
    "tfidf": {"model": None,  "get_data": None, "launch_function": tfidf.infer,},
}


@app.post("/infer/{app_name}")

def infer(app_name: App_Name, text: TextRequest):
    """
    Возвращает вероятность, что новость является нерелевантной
    """
    select_model = None
    if app_name in {"roberta", "tfidf"}:
        select_model = models[app_name]["launch_function"](text.text)
    else: 
        raise Exception("not supported")
    return select_model

@app.get("/models_names")
def get_models_names(app_name: App_Name):
    """
    Возвращает словарь с названиями моделей из файла s3_models.
    """
    if app_name is App_Name.roberta:
        return {"model_name": models_titles[0],}
    else:
        return {"model_name": models_titles[1],}

