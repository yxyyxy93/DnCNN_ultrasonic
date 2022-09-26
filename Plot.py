import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Using numpy we can use the function loadtxt to load your CSV file.
# We ignore the first line with the column names and use ',' as a delimiter.
data = np.loadtxt('my_file_2_fold.csv', delimiter=',', skiprows=1)

# You can access the columns directly, but let us just define them for clarity.
# This uses array slicing/indexing to cut the correct columns into variables.
train_loss = data[:, 0]
val_loss = data[:, 1]

# With matplotlib we define a new subplot with a certain size (10x10)
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(train_loss, label='train loss')
ax.plot(val_loss, label='val loss')

# Show the legend
plt.legend()

plt.show()