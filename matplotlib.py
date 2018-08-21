# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:04:43 2018

@author: Abhishek S
"""

import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(2,2,1)
ax.plot([1, 2, 3, 4], [10, 20, 25, 30],color='lightblue',linewidth=3)
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26],color='darkgreen',marker='^')
ax.set_xlim(0,4.5)
plt.show()

plt.plot([1, 2, 3, 4], [10, 20, 25, 30],color='lightblue',linewidth=3)
plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26],color='darkgreen',marker='^')
plt.xlim(0,4.5)
plt.show()

plt??
