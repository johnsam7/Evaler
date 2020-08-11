#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:55:39 2019

@author: ju357
"""
import matplotlib.pyplot as plt
#from mayavi import mlab
import numpy as np
import mne
def figure_preferences():
    font = {'weight' : 'normal',
            'size'   : 8}

    plt.rc('font', **font)

    dpi = 300
    figsize = (4,3)    
    return (dpi,figsize)
from .source_space_tools import join_source_spaces
#from .mne_simulations import get_raw_noise



def generate_colors(N):
    """
    Returns colors in RGB, tuple format, ranging from 0-1.0 that can be used by plotting/rendering packages.
    Input:
        N = desired number of colors (may not exceed 10)
    Output:
        colors_rgb = list of colors in tuple format
    """
    if N>10:
        raise Exception('Can only generate a maximum of 10 colors! Breaking.')
    else:
        colors_rgb = [(0,0,1.0),(1.0,0,0),(0,1.0,0),(0.7,0,0.7),(1.,0.64,0.),(0,1.0,1.0), \
                      (0.5,0.5,0.5),(0.5,0.,0.5)]
        return colors_rgb[0:N]


def generate_markers(N):
    """
    Returns a list of markers that can be used by plotting/rendering packages.
    Input:
        N = desired number of markers (may not exceed 8)
    Output:
        markers = list of markers 
    """
    if N>8:
        raise Exception('Can only generate a maximum of 8 markers! Breaking.')
    else:
        return ['o', '^', 'D', '*', 'x', 's', '>', '+']


def add_tick(axis,tick_value,color='k'):
    """
    Add a tick to axis at tick_value and color it with the color argument (in-place adjustment of current plt figure).
    """
    if axis not in ['x','y']:
        raise ValueError('Axis argument must be either x or y')
    if axis == 'y':
        all_ticks = np.sort(np.insert(plt.yticks()[0],0,tick_value))
        tick_ind = np.where(np.isin(all_ticks,tick_value))[0][0]
        all_ticks = np.round(all_ticks,2)
        plt.yticks(list(all_ticks))
        plt.yticks()[1][tick_ind].set_color(color)
    if axis == 'x':
        all_ticks = np.sort(np.insert(plt.xticks()[0],0,tick_value))
        tick_ind = np.where(np.isin(all_ticks,tick_value))[0][0]
        all_ticks = np.round(all_ticks,2)
        plt.xticks(list(all_ticks))
        plt.xticks()[1][tick_ind].set_color(color)
    return 


def change_tick(axis, tick_value, new_label, n_ticks_inv=2):
    """
    Change the label of tick at tick_value to new_label (in-place adjustment of current plt figure).
    n_ticks_inv is a parameter stating the number of ticks that are not visible - adjust until right.
    
    """
    if axis not in ['x','y']:
        raise ValueError('Axis argument must be either x or y')
    if axis == 'y':
        all_ticks = [ y[n_ticks_inv : len(plt.yticks()[0])-n_ticks_inv] for y in plt.yticks()]
        tick_values = all_ticks[0]
        tick_ind = np.where(np.isin(tick_values,tick_value))[0][0]
        all_ticks[1][tick_ind].set_text(new_label)
        plt.yticks(all_ticks[0], all_ticks[1])
    if axis == 'x':
        all_ticks = [ x[n_ticks_inv : len(plt.xticks()[0])-n_ticks_inv] for x in plt.xticks()]
        tick_values = all_ticks[0]
        tick_ind = np.where(np.isin(tick_values,tick_value))[0][0]
        all_ticks[1][tick_ind].set_text(new_label)
        plt.xticks(all_ticks[0], all_ticks[1])
    return 
        

def plot_event_topography(data, raw, events, tmin, tmax):
    """
    Plots event signal in sensor space in a topographic manner.
    Input:
        data = data to be plotted, in the form of an nparray of shape (sensors,time points)
        raw = raw file where info on sensor locations will be drawn from
        events = just any non-empty event file (these data are not used)
        tmin = start of event
        tmax = end of event
    """
    epochs = mne.Epochs(raw, events[0:1,:], tmin=tmin, tmax=tmax)
    evoked = epochs.average()
    evoked.data = data
    mne.viz.plot_evoked_topo(evoked,color='k')
    return


def plot_wave_forms(dipole_waveforms, t, dip_colors):
    """
    Plots wave forms of activated dipoles.
    Input:
        dipole_waveforms = wave forms to be plotted, in the form of an nparray of shape (dipoles,time points)
        t = time
        events = just any non-empty event file (these data are not used)
        tmin = start of event
        tmax = end of event
    """
    plt.figure()
    plt.xlabel('time (s)')
    plt.ylabel('current dipole amplitude')
    plt.title('simulated waveforms')
    for dip in range(dipole_waveforms.shape[0]):
        plt.plot(t, dipole_waveforms[dip,:], color=dip_colors[dip])
    plt.show()
    return


def plot_3d_dipoles(fwd, dipole_waveforms, t, verts):
    """
    Plots position of dipoles over cortex along with their respective wave forms (color coded)
    Input:
        fwd = forward model used for simulations.
        dipole_waveforms = wave forms to be plotted, in the form of an nparray of shape (dipoles,time points)
        t = time
        verts = list of vertices or groups of vertices found from reading labels
    """    
    dip_colors = generate_colors(len(verts))
    plot_wave_forms(dipole_waveforms, t, dip_colors)
    src = join_source_spaces(fwd['src'])
    verts_lh = src['rr']  # The vertices of the source space
    tris = src['tris']  # Groups of three vertices that form triangles
    mlab.figure(size=(600, 400), bgcolor=(1.0, 1.0, 1.0))
    mlab.triangular_mesh(verts_lh[:, 0], verts_lh[:, 1], verts_lh[:, 2], tris, color=(0.5, 0.5, 0.5))
    
    print('Rendering dipoles over brain surface, if you used big labels with many vertices, this will take a while...')
    for c in range(len(verts)):
        for dipoles in verts[c]:
            dip_pos = src['rr'][np.array(dipoles).flatten().astype(int)]  # The position of the dipoles
            normals = src['nn'][np.array(dipoles).flatten().astype(int)]
            mlab.quiver3d(dip_pos[:, 0], dip_pos[:, 1], dip_pos[:, 2],
                          normals[:, 0], normals[:, 1], normals[:, 2],
                          color=dip_colors[c], scale_factor=0.5E-2)


def plot_auc(auc_dic, SNRs, plot_limits=False, plot_SNR_0_inf=False, curve='roc'):
    import matplotlib.ticker as mticker
    
    dpi,figsize = figure_preferences()
    colors = generate_colors(len(auc_dic.keys()))
    h = plt.figure(dpi=dpi, figsize=figsize)
    markers = generate_markers(len(auc_dic.keys()))
    markersize = 5
    alpha = 1.
    
    for c, key in enumerate(auc_dic.keys()):
        auc = auc_dic[key]
        if key == 'mixed_norm':
            label = 'MxNE'
        else:
            label = key
        if plot_limits:
            plt.semilogx(SNRs, auc[0,:], '--', alpha = 0.4, linewidth = .8, marker=markers[c], color=colors[c])
            plt.semilogx(SNRs, auc[1,:], '-', alpha=alpha, label = label, marker=markers[c], color=colors[c])
            plt.semilogx(SNRs, auc[2,:], '--', alpha = 0.4, linewidth = .8, marker=markers[c], color=colors[c])
        else:
            if plot_SNR_0_inf:
                plt.semilogx(SNRs[1:len(SNRs)-1], auc[1,:][1:len(SNRs)-1], '-', 
                                  alpha=alpha, marker=markers[c], markersize=markersize, color=colors[c], label=label)
                plt.semilogx(SNRs[0:2], auc[1,:][0:2], '--', alpha=alpha, marker=markers[c], markersize=markersize, color=colors[c])
                plt.semilogx(SNRs[len(SNRs)-2:len(SNRs)], auc[1,:][len(SNRs)-2:len(SNRs)], '--', 
                                  alpha=alpha, marker=markers[c], markersize=markersize, color=colors[c])
            else:
                plt.semilogx(SNRs, auc[1,:], '-', alpha=alpha, label = label, marker=markers[c], markersize=markersize, color=colors[c])
    
#    plt.ylim(0.35, 1.05)
    if curve == 'roc':
        plt.plot(np.array([np.min(SNRs) , np.max(SNRs)]), np.array([1.0, 1.0]), 
                 '--', color='black', alpha=0.5, label='ideal')
        plt.plot(np.array([np.min(SNRs) , np.max(SNRs)]), np.array([0.5, 0.5]), 
                 '--', color='gray', alpha=0.5, label='random')
        plt.ylabel('AUC (ROC)')
    if curve == 'prc':
        plt.plot(np.array([np.min(SNRs) , np.max(SNRs)]), np.array([1.0, 1.0]), 
                 '--', color='black', alpha=0.5, label='ideal')
        plt.plot(np.array([np.min(SNRs) , np.max(SNRs)]), np.array([.001, .001]), 
                 '--', color='gray', alpha=0.5, label='random')
        plt.ylabel('AUC (PRC)')


    plt.xlim(np.min(SNRs),np.max(SNRs))
    plt.xlabel('SNR')
    plt.legend(loc='upper left')
    
    # Set non-scientific notation on x-axis
    ax=plt.gca()
    a = mticker.ScalarFormatter()
    ax.xaxis.set_major_formatter(a)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    return h


def plot_roc(roc, labels, curve='roc'):
    colors = generate_colors(len(labels))
    dpi,figsize = figure_preferences()
    h = plt.figure(dpi=dpi, figsize=figsize)
    markers = generate_markers(len(labels))
    markersize=5
    alpha = 1.0

    for c,roc in enumerate(roc):
        plt.plot(roc[0,:], roc[1,:], label=labels[c], marker=markers[c], alpha=alpha, markersize=markersize, color=colors[c])

    if curve == 'roc':
        plt.plot([0,0,1],[0,1,1],linestyle = '--', color='black', alpha=0.5, label='ideal')
        plt.plot([0,1],[0,1],linestyle = '--', color='gray', alpha=0.5, label='random')
        plt.xlabel('False positive rate (1-specificity)')
        plt.ylabel('True positive rate (sensitivity)')
        plt.legend(loc='lower right')

    if curve == 'prc':
        plt.plot([1,1,0],[.001,1,1],linestyle = '--', color='black', alpha=0.5, label='ideal')
        plt.plot([0,1],[.001,.001],linestyle = '--', color='gray', alpha=0.5, label='random')
        plt.xlabel('True positive rate (sensitivity)')
        plt.ylabel('Positive predictive value (precision)')
        plt.legend(loc='upper right')
        
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    return h


def plot_auc_fit(auc, SNRs, inv_method):
    from scipy.optimize import minimize
    import matplotlib.ticker as mticker

    def tanh_log(para, x):
        return para[0]*np.tanh(para[1]*np.log10(x)+para[2])+para[3]

    def tanh_fit(para,x,y):
        """
        Returns an objective function to fit (x,y) data to a tanh function.
        Input:
            x : np.array of shape (n,), x data to be fitted
            y : np.array of shape (n,), y data to be fitted
            para : np.array of shape (4,), the a,b,c,d parameters in a*tanh(b*x+c)+d
        Output:
            min_fun : a least squares cost function that can be used by an optimizer to
                        find a,b,c,d
        """
        return np.linalg.norm(y-(tanh_log(para, x))) 
    
    dpi, figsize = figure_preferences()
    h = plt.figure(dpi=dpi, figsize=(5,4) )
    para = minimize(tanh_fit, np.array([0.25, 1, 0.1, 1]), (SNRs, auc[inv_method][1,:]))
    x = np.logspace(np.log10(np.min(SNRs)), np.log10(np.max(SNRs)), 100)
    auc_analytical = tanh_log(para['x'], x)
    plt.semilogx(x, auc_analytical, linestyle = '--', color='gray', alpha=0.5)    
    plt.title(inv_method)
    
    # Normalized Derivate
    derivative = np.abs(para['x'][0]*para['x'][1]*(1-np.tanh(para['x'][1]*np.log10(x)+para['x'][2])**2))
    derivative_normalized = derivative/np.max(derivative)
    plt.semilogx(x, derivative_normalized, linestyle = '--', color='k', alpha=0.5)
    
    # Curve
#    plt.semilogx(SNRs, auc[inv_method][0,:], '--', color=generate_colors(1)[0], alpha=0.5, linewidth=.8)
    plt.semilogx(SNRs, auc[inv_method][1,:], '-', color=generate_colors(1)[0])
#    plt.semilogx(SNRs, auc[inv_method][2,:], '--', color=generate_colors(1)[0], alpha=0.5, linewidth=.8)
    plt.xlim(np.min(SNRs),np.max(SNRs))
    plt.xlabel('SNR')
    plt.ylabel('AUC')
    
    # Set non-scientific notation on x-axis
    ax=plt.gca()
    a = mticker.ScalarFormatter()
    ax.xaxis.set_major_formatter(a)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    # Set 0.5 line (random guessing) and 1.0 (perfect estimates) and grid
    plt.plot(np.array([np.min(SNRs), np.max(SNRs)]), np.array([0.5, 0.5]),
             color='r', alpha=0.6, linewidth=1.0, linestyle='--')
    plt.plot(np.array([np.min(SNRs), np.max(SNRs)]), np.array([1.0, 1.0]),
             color='r', alpha=0.6, linewidth=1.0, linestyle='--')    
    plt.grid()
    
    return x, auc_analytical, np.array(auc[inv_method][1,:]), h


def cumulative_plot(res_metric, x_label, cutoff = 1.,
                    plot_cutoff_line=False, labels=None, log_scale=True):
    # Plots cumulative histogram plots of the data in dictionary res_metric

    dpi,figsize=figure_preferences()
    percs = []            
    colors = generate_colors(len(labels))
    markers = generate_markers(len(labels))
    markersize = 5

    for key in list(res_metric.keys()):
        metrics = res_metric[key]
        metrics = metrics[~np.isnan(metrics)]
        hist_fig = plt.figure()
        if log_scale:
            # Replace 0's with 10**-3 and change tick later in figure (to avoid log(0) problem)
            metrics[np.where(metrics == 0.)[0]] = 10**-3
            metrics[np.argmin(metrics)] = 10**-3
            percs.append(plt.hist(metrics, cumulative=True,
                                  bins=len(metrics), density=True))
        else:
            percs.append(plt.hist(metrics, cumulative=True, bins=len(metrics),
                                  density=True))
        plt.close(hist_fig)
        
    h = plt.figure(dpi=dpi,figsize=figsize)
    for c,perc in enumerate(percs):
        average_bins = np.mean(np.array([perc[1][0:-1],perc[1][1:len(perc[1])]]).T,axis=1)
        if not labels == None:
            if log_scale:
                plt.semilogx(np.insert(average_bins,0,average_bins[0]),
                             np.insert(perc[0],0,0)*100, color=colors[c], marker=markers[c], markersize=markersize, label=labels[c])
            else:
                plt.plot(np.insert(average_bins,0,average_bins[0]),
                         np.insert(perc[0],0,0)*100,color=colors[c], marker=markers[c], markersize=markersize, label=labels[c])
        else:
            if log_scale:
                plt.semilogx(np.insert(average_bins,0,average_bins[0]),
                             np.insert(perc[0],0,0)*100, marker=markers[c], markersize=markersize, color=colors[c])
            else:
                plt.plot(np.insert(average_bins,0,average_bins[0]),
                         np.insert(perc[0],0,0)*100, marker=markers[c], markersize=markersize, color=colors[c])

        plt.ylim((0.,100.))
        
        plt.xlabel(x_label)
        plt.ylabel('Percent of brain regions (%)')

        if plot_cutoff_line:
            unit_ind = np.argmin(np.abs(average_bins-np.log10(cutoff)))
            cutoff_1 = perc[0][unit_ind]*100
            plt.plot(plt.xlim(),(cutoff_1,)*2,color=colors[c],linestyle='--')
            add_tick(axis='y',tick_value=cutoff_1,color=colors[c])
    
    if plot_cutoff_line:
        plt.plot([cutoff,cutoff],[0,100],color='gray',linestyle='--')

    ax = h.axes[0]
    label_ind = np.argmin(np.abs(plt.xticks()[0]-10**-3))
    labels = ax.get_xticks().tolist()
    labels[label_ind] = '0'
    ax.set_xticklabels(labels)
    

    plt.xlim(plt.xlim())
    if x_label == 'Point spread (normalized)':
        plt.xlim((9*10**-4, plt.xlim()[1]))
    plt.legend(loc='lower right')
    plt.tight_layout()
    return h


def plot_topographic_parcellation(scalar, settings, labels, clims = [2,50,98], vrange = None,
                                  hemi='both', title='brain plot', transparent=True):
    if not scalar.shape[0] == len(labels):
        raise ValueError('Number of labels must match length of resolution matrix. Aborting.')
    fwd = mne.read_forward_solution(settings.fname_fwd())
    waveform = scalar.reshape(len(labels),1)
    stc = mne.simulation.simulate_stc(fwd['src'],labels,waveform,tmin=0,tstep=1)
    if vrange == None:
        if np.sort(scalar)[int(len(scalar) * clims[0] / 100)] == 0.:
            clim = {'kind' : 'value', 'lims' : (np.max(scalar)*0.05, np.max(scalar)*0.5, np.max(scalar)*0.95)}
        else:
            clim = {'kind' : 'percent', 'lims' : clims}        
    else:
        clim = {'kind' : 'value', 'lims' : (vrange[0], vrange[1], vrange[2])}
    brain = stc.plot(hemi=hemi, subjects_dir=settings.subjects_dir(), clim=clim, \
                     colormap = 'hot_r', transparent=transparent, background='white', foreground='black',
                     title=title, alpha=1.0, views='med')
    return brain, stc

def plot_resolution_matrix(R, labels, title, SNR, vrange = None, show_colorbar=False, show_labels=False, figsize=(5,5)):
    import matplotlib
    
    # Set figure preferences
    font = {'weight' : 'normal',
            'size'   : 4}

    plt.rc('font', **font)
    plt.figure(figsize=figsize, dpi=300)
    plt.title(title, size=10)

    # Rearrange labels in hemispheres and lobes
    group_dir = [{}, {}]
    labels_lh = [label for label in labels if label.hemi=='lh']
    labels_rh = [label for label in labels if label.hemi=='rh']
    labels_hemi = [labels_lh, labels_rh]

    # Fix ticks after label groups
    for c, hemi in enumerate(['lh', 'rh']):
        tick_labels = labels_hemi[c]
        group_name = ''
        border_number = 0
        for tick_label in tick_labels:
            name = tick_label.name
            group = ''
            for letter in name:
                if letter == '_':
                    break
                group = group + letter
            if not group == group_name:
                if not group_name == '':
                    group_dir[c].update({group_name : border_number})
                group_name = group
                border_number = 1
            else:
                border_number = border_number + 1
        group_dir[c].update({group_name : border_number})

    frontal_groups = [1, 2, 5, 12, 14, 16, 18, 19, 20, 24, 26, 27, 28]
    parietal_groups = [7, 10, 22, 23, 25, 29, 31]
    temporal_groups = [0, 4, 6, 8, 9, 15, 17, 30, 32, 33]
    occipital_groups = [3, 11, 13, 21]
    all_groups = [1, 2, 5, 12, 14, 16, 18, 19, 20, 24, 26, 27, 28, 7, 10, 22, 23, 25, 29, 31, \
                  0, 4, 6, 8, 9, 15, 17, 30, 32, 33, 3, 11, 13, 21]
    lengths = [len(frontal_groups), len(parietal_groups), len(temporal_groups), len(occipital_groups)]

    # Find indices for alphabetic order of labels
    inds = []
    count = 0
    for key in list(group_dir[0].keys()):        
        inds.append(np.arange(count, count + group_dir[0][key]))
        count = count + group_dir[0][key]
    for key in list(group_dir[1].keys()):        
        inds.append(np.arange(count, count + group_dir[1][key]))
        count = count + group_dir[1][key]
        
    # Flip around indices to lobular order
    ind_flip = np.array([])
    all_groups_2 = [x + len(all_groups) for x in all_groups]
    all_groups_whole = all_groups + all_groups_2
    lob_order = [inds[ind] for ind in all_groups_whole]
    for array in lob_order:
        ind_flip = np.concatenate((ind_flip, array)).astype(int)

    R = R[ind_flip, :][:, ind_flip]
    
    # Scale to 1, plot R with vmax=99:th percentile
    R = np.abs(R/np.max(np.abs(R)))
    if vrange == None:
        vmax = np.sort(R.flatten())[int(0.99*R.flatten().shape[0])]
        if vmax == 0.0:
            vmax = 0.04
        vrange = [0, vmax]
            
    plot = plt.imshow(R, vmin=vrange[0], vmax=vrange[1], cmap='hot_r')

    # Rearrange labels
    labels_rear = []
    for hemi in group_dir:
        dir_arr = {}
        keys = list(hemi.keys())
        keys = [keys[ind] for ind in all_groups]
        for key in keys:
            dir_arr.update({key: hemi[key]})
        labels_rear.append(dir_arr)

    tick_ind = 0
    minor_ticks = []
    minor_ticks.append(0)
    for c, hemi in enumerate(['lh', 'rh']): 
        for group in list(labels_rear[c].keys()):
            minor_ticks.append(int(labels_rear[c][group])+tick_ind)
            tick_ind = tick_ind + labels_rear[c][group]

    tick_ind = 0
    major_ticks = []
    major_ticks.append(0)
    ind_l = 0
    for c, hemi in enumerate(['lh', 'rh']):
        keys = list(labels_rear[c].keys())
        pointer = 0
        for length in lengths:
            jump = np.sum(labels_rear[c][key] for key in keys[pointer : pointer + length])
            pointer = pointer + length
            ind_l = ind_l + jump
            major_ticks.append(ind_l)

#        for group in [frontal_groups, parietal_groups, temporal_groups, occipital_groups]:
#            for ind in group:
#                major_ticks.append(int(labels_rear[c][group])+tick_ind)
#                tick_ind = tick_ind + labels_rear[c][group]
    
    # Change size of minor ticks
    locs, tick_labels = plt.xticks()
    plt.xticks(minor_ticks, np.concatenate((np.arange(len(all_groups)),np.arange(len(all_groups)))), size=4, rotation='horizontal')
    locs, tick_labels = plt.yticks()
    plt.yticks(minor_ticks, np.concatenate((np.arange(len(all_groups)),np.arange(len(all_groups)))), size=4, rotation='horizontal')
    
    ax = plt.gca()
    axes = [ax.get_xaxis(), ax.get_yaxis()]
    for axis in axes:
        axis.set_minor_locator(matplotlib.ticker.FixedLocator(minor_ticks))
        axis.set_minor_formatter(matplotlib.ticker.FixedFormatter(np.concatenate((range(len(all_groups)),range(len(all_groups))))))
        axis.set_major_locator(matplotlib.ticker.FixedLocator(major_ticks))    
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter(['frontal', 'parietal',
                                                                  'temporal', 'occipital',
                                                                  'frontal', 'parietal', 'temporal', 'occipital', ' ']))
        axis.set_tick_params(which='major', pad=10)

    plt.yticks(rotation='vertical')
    plt.xticks(size=6)
    plt.yticks(size=6)

    # Set grid lines between source groups and make lines between hemispheres
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2, linewidth = .2)
    plt.grid(b=True, which='major', color='k', linestyle='--', alpha=0.5, linewidth = .4)
    plt.plot(np.array([0,len(labels)-1]), np.array([len(labels_lh), len(labels_lh)]), color='b', linewidth = 0.5)
    plt.plot(np.array([len(labels_lh), len(labels_lh)]), np.array([0,len(labels)-1]), color='b', linewidth = 0.5)
    
#    for d in ["left", "top", "bottom", "right"]:
#        plt.gca().spines[d].set_visible(False)
    plt.ylabel('SNR = '+str(SNR), fontsize=10)
    plt.tight_layout()

    # Create Figure for the colormap
    if show_colorbar:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=6)
    
    # Create new Figure for the label ticks
    if show_labels:
        plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
        names = list(group_dir[0].keys())
        names = [names[x] for x in all_groups]
        tick_names = [str(i) + '. ' + key for i, key in enumerate(names)]
        figure_preferences()
        plt.figure(figsize=(5,5), dpi=300)
        plt.plot(np.arange(len(group_dir[0].keys())), np.arange(len(group_dir[0].keys())))
        locs, tick_labels = plt.xticks()
        plt.xticks(np.arange(len(group_dir[0].keys())), tick_names, rotation='horizontal')
        locs, tick_labels = plt.yticks()
        plt.yticks(np.arange(len(list(group_dir[0].keys()))), tick_names[::-1], rotation='horizontal')
        plt.show()
        plt.tight_layout()
        plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
        plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    plt.tight_layout()
    
    return plot


def circle_plot_resolution(center_labels, R, labels, threshold=0.1, graph_threshold=0.25, color_mode='white'):
    import mne.viz

    title = ['Point spread', 'Cross talk']
    for k in range(2):
        # Do point spread first, then transpose resolution matrix to get cross talk
        if k == 1:
            R = R.T

        # Find what labels have over threshold psf/ct with each center_label (standardized with respect to own label estimate)
        relevant_labels = np.array([])
        n_graphs = 0
        for center_label in center_labels:
            ind_label = labels.index(center_label)
            psf = np.abs(R[:,ind_label])/np.abs(R[ind_label,ind_label])
            inds = np.where(psf>threshold)[0]
            relevant_labels = np.concatenate((relevant_labels, inds)).astype(int)
        relevant_labels = list(set(relevant_labels))
        
        # Put relevant labels in the right order, sorted as sum of psf/ct from center_labels[0]
        relevant_labels = [relevant_labels[i] for i in np.argsort(R[relevant_labels,labels.index(center_labels[0])])]#[::-1]
        inds_center_labels = round(len(relevant_labels)/len(center_labels))*np.linspace(0,len(center_labels)-1,len(center_labels)).astype(int)

        # Spread center_labels azimuthally evenly over the circle plot
        for c, center_label in enumerate(center_labels):
            relevant_labels.remove(labels.index(center_label))        
        for c, center_label in enumerate(center_labels):
            relevant_labels.insert(inds_center_labels[c],labels.index(center_label))
    
        # Get connection strengths
        cov = np.array([])
        indices = (np.array([]), np.array([]))
        for c, center_label in enumerate(center_labels):
            ind_label = labels.index(center_label)
            psf = np.abs(R[relevant_labels,ind_label])/np.abs(R[ind_label,ind_label])
            cov = np.concatenate((cov, psf))
            n_graphs = n_graphs + np.where(psf > graph_threshold)[0].shape[0]
            indices = (np.concatenate((indices[0].astype(int), (np.ones(len(relevant_labels))*inds_center_labels[c]).astype(int))),
                       np.concatenate((indices[1].astype(int), np.linspace(0,len(relevant_labels)-1,len(relevant_labels)).astype(int))))
        
        # Set group boundaries to separate center_labels from rest
        label_names = [labels[relevant_label].name for relevant_label in relevant_labels]
        node_order = label_names
        group_boundaries=[]
        for center_label_ind in inds_center_labels:
            group_boundaries.append(center_label_ind)
            group_boundaries.append(center_label_ind+1)
        node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90, start_between = True,
                                      group_boundaries=group_boundaries)
        node_angles = node_angles - (node_angles[0]-450)
        
        # Set color mode
        if color_mode == 'white':
            facecolor='white'
            textcolor='black'
            node_edgecolor='white'
            colormap='hot_r'
        if color_mode == 'black':
            facecolor='black'
            textcolor='white'
            node_edgecolor='black'
            colormap='hot'
            
        # Remove the nodes that go to the same label it started from (will always be 1 bc of normalization anyway)
        same_nodes = []
        for c, ind in enumerate(indices[0]):
            if ind == indices[1][c]:
                same_nodes.append(c)
        cov = cov[[x for x in range(len(cov)) if x not in same_nodes]]
        indices = (indices[0][[x for x in range(len(indices[0])) if x not in same_nodes]],
                   indices[1][[x for x in range(len(indices[1])) if x not in same_nodes]])
        
                
        # Make the plot
        mne.viz.plot_connectivity_circle(cov, label_names, indices=indices, n_lines=n_graphs, \
                                 node_angles=node_angles, title=title[k], vmin=graph_threshold, vmax=1.0, \
                                 facecolor=facecolor, textcolor=textcolor, node_edgecolor=node_edgecolor, \
                                 colormap=colormap, node_linewidth=1.0, linewidth=2.0,  fontsize_names=10)
    if len(center_labels) > 1:
        # Make interconnectivity plot for center_labels
        center_label_inds = []
        for center_label in center_labels:
            center_label_inds.append(labels.index(center_label))  
        R_center = R[center_label_inds, :][:, center_label_inds]
        center_label_names = [labels[np.array(center_label_ind)].name for center_label_ind in center_label_inds]
        central_connections = (np.repeat(np.arange(len(center_labels)),repeats=len(center_labels)),
                               np.dot(np.ones((R_center.shape[0], 1)),np.linspace(0, R_center.shape[1]-1, 
                               R_center.shape[1]).reshape(1,R_center.shape[1])).flatten().astype(int))
        R_ave = (R_center+R_center.T)*0.5
        
        # Shows both ct and psf but does not show which one is which (directionaltiy cannot be displayed bc of limitations in Python)
        mne.viz.plot_connectivity_circle(R_center.flatten(), center_label_names, indices=central_connections, vmin=0.0, vmax=1.0, title='intra ct/psf',
                         facecolor=facecolor, textcolor=textcolor, node_edgecolor=node_edgecolor, colormap=colormap, node_linewidth=1.0, linewidth=2.0, fontsize_names=10)
        # Shows average, giving a rough idea of how much different labels bleed into eachother
        mne.viz.plot_connectivity_circle(R_ave, center_label_names, vmin=0.0, vmax=1.0, title='intra average ct, psf', fontsize_names=10,
                         facecolor=facecolor, textcolor=textcolor, node_edgecolor=node_edgecolor, colormap=colormap, node_linewidth=1.0, linewidth=2.0)
        
    return


def bar_plot(x, y, plot_labels, bars='std', x_label='', y_label='', y_lims=None, x_lims=None, show=True):
    """
    Plots single sided bars representing standard deviation of data distribution.
    y - 3d numpy array of shape (2, data points, distribution at each data point)
    bars - 'std' or 'stde' tells whether bars represent standard deviation or standard error
    """
    (dpi,figsize) = figure_preferences()
    
    bar_plot_fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.tight_layout()
    colors = generate_colors(2)
 
    if not y_lims == None:
        plt.ylim(y_lims[0],y_lims[1])
    if not x_lims == None:
        plt.xlim(x_lims[0],x_lims[1])
        
    for k in range(y.shape[0]):
        my_mean = np.nanmedian(y[k,:,:],axis=1)
        if bars == 'stde':
            my_std = np.nanstd(y[k,:,:],axis=1)/np.sqrt(y[k,:,:].shape[1])
        else:
            my_std = np.nanstd(y[k,:,:],axis=1)
        if k == 1:
            yerr = np.array([np.zeros(len(my_std)),my_std])
        if k == 0:
            yerr = np.array([my_std,np.zeros(len(my_std))])
        plt.errorbar(x, my_mean, yerr=yerr, linestyle = '-', color = colors[k], \
                     label=plot_labels[k],linewidth=1.2, alpha=0.7) 
        plt.grid(linestyle='--',alpha=0.3)

    plt.legend(loc = 'upper right',fontsize=9)
    plt.ylabel('Conservation Factor $C$')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
        
#    plt.savefig('figures/canc_patch.svg',bbox_inches="tight",pad_inches=0.1)
    if show:
        plt.show()

    return bar_plot_fig


def viz_noise(raw, eve=None, n_t=None, eve_dur=1.0, n_epoch=1):
    """
    Function to visualize noise data generated by get_noise function based on raw and eve.
    """
    from .mne_simulation import get_raw_noise
    font = {'weight' : 'normal',
            'size'   : 10}
    plt.rc('font', **font)
    modalities = [['mag', False], ['grad', False], [False, True]]
    labels = ['magnetometers', 'gradiometers', 'EEG']
    if not eve == None:
        noise_raw = get_raw_noise(raw, eve, event_duration=eve_dur, n_t=n_t)
    else:
        noise_raw = raw
    noise_cov = mne.compute_raw_covariance(noise_raw)
#    noise_power = np.diag(noise_cov.data)
    fig_eigenspectrum = plt.figure()
    plt.figure(fig_eigenspectrum.number)
    plt.semilogy(-np.sort(-np.abs(np.linalg.eig(noise_cov['data'])[0])), '*')
    plt.title('Eigenspectrum')
    if not raw.preload:
        raw.load_data()
    
    for c, modality in enumerate(modalities):
        plt.figure()
        plt.title(labels[c])
        meg = modalities[c][0]
        eeg = modalities[c][1]
        raw_ch = raw.copy()
        raw_ch = raw_ch.pick_types(meg=meg, eeg=eeg)
        noise_cov = mne.compute_raw_covariance(raw_ch)
        ch_power = np.sqrt(np.diag(noise_cov.data))
#        chs = np.array([c for c, ch in enumerate(raw.info['ch_names']) if not ch in raw.info['bads'] and ch in raw_ch.info['ch_names']])
#        chs = mne.pick_types(raw.info,meg=meg,eeg=eeg)
#        ch_power = noise_power[chs]
        mne.viz.plot_topomap(ch_power, raw_ch.info)

        plt.figure()
        plt.title('power of ' + labels[c] + ' channels')
        ind_sort = np.argsort(ch_power)
        ch_power = ch_power[ind_sort]
        plt.plot(ch_power, '*')
        locs, tick_labels = plt.xticks()
        plt.xticks(range(len(raw_ch.info['ch_names'])), [raw_ch.info['ch_names'][x] for x in ind_sort], rotation='vertical')

    return
