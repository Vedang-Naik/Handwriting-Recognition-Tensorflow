import pandas as pd
import numpy as np

data = pd.read_csv("emnist-letters-train.csv")
data = data.as_matrix()

import scipy.misc
space = [0] * 784
scipy.misc.imsave("Train Images/space.png",
					np.array(data[i][1:]).reshape(28, 28, order='F'))
for i in range(65000, len(data)):
	scipy.misc.imsave("Train Images/" + chr(data[i][0] + 64) + str(i) + ".png",
						np.array(data[i][1:]).reshape(28, 28, order='F'))
