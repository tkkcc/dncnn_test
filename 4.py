# import csv
# f=open('employee_file2.csv',mode='w')
# fieldnames = ['emp_name', 'dept', 'birth_month']
# # writer = csv.DictWriter(f)
# writer = csv.writer(f)
# writer.writerow(['Spam'] * 5 + ['Baked Beans'])
# writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
# # writer.writeheader()
# # writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
# # writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})
# # writer.writerow({ 'dept': 'IT', 'birth_month': 'March'})
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
# fig = plt.figure(figsize=(5, 30))
# gs = gridspec.GridSpec(30, 5)
# ax1 = plt.subplot(gs[:5, :])
# ax2 = plt.subplot(gs[5,0])
# ax2 = plt.subplot(gs[5,1])
# ax2 = plt.subplot(gs[5, 2])
# ax2.show()
# # ax3 = plt.subplot(gs[1:, -1])
# # ax4 = plt.subplot(gs[-1, 0])
# # ax5 = plt.subplot(gs[-5:,:])
# ax0 = plt.subplot2grid((5, 5), (0, 0), colspan=3)
# # plt.figure
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(30, 5, figsize=(5, 30))
fig.show()
ax0 = plt.subplot2grid((30, 5), (0, 0), colspan=5,rowspan=5)
# t = plt.subplot2grid(30, 5, cospan=)
plt.show()
