from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import io
import pandas as pd
import pickle
import uvicorn

from utils import preprocess_item, process_df


app = FastAPI()


with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.get('/')
async def root():
    return {"Herzlich Willkommen bei ML-inferenz Start-Seite!"}


@app.post("/predict_item")
def predict_item(item: Item) -> str:
    processed_item = preprocess_item(item.model_dump())
    result = loaded_model.predict(processed_item)
    return f'Regressor with categoties and Ridge prediction: {result}'


@app.post("/predict_items")
async def predict_items(file: UploadFile, response_class=StreamingResponse):
    content = await file.read()
    df_csv = pd.read_csv(io.BytesIO(content))
    df, df_scaled = process_df(df_csv)
    pred = loaded_model.predict(df_scaled)
    df_2 = df_csv.copy()
    idx = df.index.to_numpy()
    df_2.loc[idx, 'selling_price'] = pred

    output = io.StringIO()
    df_2.to_csv(output, index=False)
    output.seek(0)

    response = StreamingResponse(
        iter([output.getvalue()]), media_type="text/csv"
    )
    response.headers[
        "Content-Disposition"
    ] = "attachment; filename=predictions.csv"
    return response


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)
