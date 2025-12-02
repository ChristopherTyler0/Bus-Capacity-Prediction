import pandas as pd
df = pd.read_csv("busdata.csv")
print(df.head)
columns = ['stopNumber', 'actualCapacity']
result = df[columns].groupby('stopNumber').mean()

print(result)

total = 0
correct = 0
marginOfError = 0.75
i = 0 
for index,row in df.iterrows():
    predictedCapacity = result.loc[i, 'actualCapacity']
    actualCapacity = row['actualCapacity']
    print(f"Stop Name: {row['stopName']} | Actual Capacity: {row['actualCapacity']}| Predicted Capacity: {predictedCapacity:.0f}")
    total += 1
    i+=1
    if i > 37:
        i = 0
    if (predictedCapacity >= actualCapacity * marginOfError) and (predictedCapacity <= actualCapacity / marginOfError):
        correct += 1
print(f"Final accuracy with margin of error {marginOfError}: {correct/total}")