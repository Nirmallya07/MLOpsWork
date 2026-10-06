"Behaviour tests for the trained model."

import pandas as pd


def make_order(distance_km=5.0, prep_time_min=20,
               traffic_level=2, rain=0):
    return pd.DataFrame([{
        "distance_km": distance_km,
        "prep_time_min": prep_time_min,
        "traffic_level": traffic_level,
        "rain": rain,
    }])


def test_rain_never_makes_delivery_faster(trained_model):
    dry = trained_model.predict(make_order(rain=0))[0]
    wet = trained_model.predict(make_order(rain=1))[0]
    assert wet >= dry


def test_further_is_never_faster(trained_model):
    near = trained_model.predict(make_order(distance_km=2))[0]
    far = trained_model.predict(make_order(distance_km=12))[0]
    assert far > near


def test_error_is_better_than_guessing(model_mae, baseline_mae):
    assert model_mae < baseline_mae


def test_heavier_traffic_is_never_faster(trained_model):
    light_traffic = [[5.0, 20, 1, 0]]
    heavy_traffic = [[5.0, 20, 3, 0]]

    light_prediction = trained_model.predict(light_traffic)[0]
    heavy_prediction = trained_model.predict(heavy_traffic)[0]

    assert heavy_prediction >= light_prediction
