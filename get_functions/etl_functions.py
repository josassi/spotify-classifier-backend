import requests
import pandas as pd
from pandas import json_normalize
import json


def request_to_df(r: requests.models.Response, attributes_list: list) -> pd.DataFrame:
    json_data = json.loads(r.text)
    try:
        for attribute in attributes_list:
            json_data = json_data[attribute]
    except KeyError:
        print("Key Error")
        print(r, r.json())
        return pd.DataFrame()
    return json_normalize(json_data)
