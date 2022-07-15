import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# define plot morphology function
def plot_morphology_swc(swc_full,child_idx=None,root_id=None,ax=None,figsize=(8,6),clr=['g','r','b','c']):
    seg_type = ['soma','axon','dend','apic']
    coor3d = list('xyz')
    rm = swc_full.loc[swc_full['type']!=1,'r'].mean()
    if child_idx is None:
        swc = swc_full
    else:
        swc = swc_full.loc[child_idx]
    ilab = []
    for i in range(4):
        try:
            ilab.append(list(swc['type']==i+1).index(True))
        except:
            ilab.append(-1)
    if root_id is None:
        root_id = swc_full.index[swc_full['pid']<0][0]
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
    else:
        fig = ax.figure
    for i, idx in enumerate(swc.index):
        label = str(ilab.index(i)+1)+': '+seg_type[ilab.index(i)] if i in ilab else None
        typeid = swc.loc[idx,'type']
        if typeid==1:
            ax.scatter(*swc.loc[idx,coor3d],c=clr[0],s=swc.loc[idx,'r']/rm,label=label)
        else:
            pid = swc.loc[idx,'pid']
            if pid is not root_id:
                line = np.vstack((swc.loc[idx,coor3d],swc_full.loc[pid,coor3d]))
                ax.plot3D(line[:,0],line[:,1],line[:,2],color=clr[typeid-1],
                          linewidth=.5*swc.loc[idx,'r']/rm,label=label)
    ax.legend()
    plt.show()
    return fig,ax
