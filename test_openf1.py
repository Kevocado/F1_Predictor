import asyncio
import pandas as pd
from f1_predictor.data import openf1
session = openf1.fetch_latest_session()
print(session['session_key'])
data = openf1._get("car_data", {"session_key": session['session_key'], "driver_number": 1})
print(len(data))
if data:
    print(data[0])
