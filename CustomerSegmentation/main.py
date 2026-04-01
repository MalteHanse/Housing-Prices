from src.data_loader import load_data
from src.preprocessing import preprcoessing

# load the dataset
data = load_data()

# sample only few, so testing becomes faster
data = data.sample(10000, random_state=50)

# process the data into a dataframe, that gives customer specified information (recency, frequency, value)
data = preprcoessing(data)


print(data)
