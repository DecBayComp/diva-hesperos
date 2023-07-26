import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

###################################
###################################
###################################


def plot_confusion_matrix(confusion):


	fig, (ax1, ax2) = plt.subplots(2, 1)

	im = ax1.pcolor(confusion)
#ax1.label_outer()
	ax1.set_xticks([])
	ax1.set_yticks([])
	fig.colorbar(im, ax=ax1)
	fig.show()

	confusion_norm = confusion/np.linalg.norm(confusion, ord=2, axis=1, keepdims=True)

	im2 = ax2.pcolor(confusion_norm)
#ax2.label_outer()
	ax2.set_xticks([])
	ax2.set_yticks([])
	fig.colorbar(im2, ax=ax2)
	fig.show()

	return fig, ax1, ax2

###################################
###################################
###################################
