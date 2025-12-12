from be_great import GReaT
import pandas as pd

# load data
df = pd.read_csv("final_synth_gen_df.csv")

# train model
print("Training model.")
model = GReaT(llm='distilgpt2',
               epochs=5, 
               fp16=True, 
               logging_steps=500, 
               experiment_dir='trainer_logs')
model.fit(df)
print("Finished training.")

# save model
model.save("GRT")  
print("Saved model.")
model = GReaT.load_from_dir("GRT")  # loads the model again
print("Loaded model.")

print("Saving synthetic data.")
n_samples=len(df)*2
samples = model.sample(n_samples,max_length=2000)
samples.to_csv("great_synthdata.csv", index=False)
print("Saved synthetic data.")