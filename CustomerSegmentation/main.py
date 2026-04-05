from src.data_loader import load_data
from src.preprocessing import preprocessing, build_preprocessor
from src.models import build_model
import src.evaluation as eval
import matplotlib.pyplot as plt

# silence warning
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")


MODEL_TYPE = "kmeans"

# load the dataset
data = load_data()

# sample only few, so testing becomes faster (remove for final model)
# data = data.sample(10000, random_state=50)

# process the data into a dataframe, that gives customer specified information (recency, frequency, value)
data = preprocessing(data)
# eval.visualize_data(data, show=False)

# build and fit model
model = build_model(build_preprocessor(data), model_type=MODEL_TYPE, clusters=3)  # amount of clusters found in elbow experiment
model.fit(data)

labels = model.predict(data)

# visualize result
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
eval.visualize_clusters(ax, data, labels, show=False)

# evaluate the results and see how the groups differ
eval.evaulate_groups(data, labels)

# the boxplots now use the normalized data and clip outliers
preprocessor = build_preprocessor(data)
eval.visualize_boxplots(data, labels, preprocessor=preprocessor, show=True)

# finally the customers can be split up in three different groups. The first group
# has normaldistributed behavior in all three categories, while the second has a low recency
# but has bought items more frequently and with more value. The last group
# has a high recency, but has bought for less money and less frequent.
# This might mean that the last group probably are churned customers,
# since they have not left much money and have not bought items for a longer
# time. The second group are most likely customers who come and order
# frequent. The first group are the rest, which is the average customer.

