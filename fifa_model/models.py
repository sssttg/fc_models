from joblib import load

model = load('models/poly_play_model.joblib')
converter =  load('models/final_converter.joblib')

campaign = [[81, 78, 87, 89, 50, 67]]
transform = converter.transform(campaign)
print(model.predict(transform))