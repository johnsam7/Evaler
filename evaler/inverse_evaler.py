"""
Function wrappers for evaluating and visualizing M/EEG source estimates using 
Evaler repo.
Author: John GW Samuelsson. 
"""
import evaler
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle

def setup_subject(subjects_dir, subject, data_path, n_epochs):
    """Setups some important subject data.

    Parameters
    ----------
    subjects_dir : string
    subject : string
    data_path : string
    n_epochs : int
        Number of epochs that the average will be based on.

    Returns
    -------
    subj_data : dictionary
        Dictionary with subject data.

    """
    settings, labels, labels_unwanted, fwd, epochs_to_use = \
        evaler.setup(subjects_dir='/autofs/cluster/fusion/data/resting_state/subjects_mri/', 
                     subject=subject,
                     data_path=data_path + subject + '/',
                     fname_raw=subject + '_raw.fif', fname_fwd=subject + '_fwd.fif',
                     fname_eve=subject+'_raw-eve.fif', fname_trans='trans.fif',
                     fname_epochs=subject+'_epo.fif', n_epochs=n_epochs, meg_and_eeg=True)
    subj_data = {'settings' : settings, 'labels' : labels, 'labels_unwanted' : labels_unwanted, 
                 'fwd' : fwd, 'epochs_to_use' : epochs_to_use, 'subject' : subject}
    return subj_data

def get_analytical_R(subject, data_path, inv_method, inv_function):
    """Gets analytical and empirical resolution matrix (only for linear inverse methods).
    
    Parameters
    ----------
    subject : string
    data_path : string
        Path to subject data folder.
    inv_method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Which inverse method to test.
    inv_function : function handle
        User-defined function that returns the estimate. See documentation in
        for_manuscript.py.

    Returns
    -------
    R_anal : array, shape (n_labels, n_labels)
        Analytical resolution matrix.
    R_emp : array, shape (n_labels, n_labels)
        Empirical resolution matrix.
        
    """
    labels, labels_unwanted = pickle.load(open(data_path+subject+'/labels', 'rb'))
    inp = np.ones((1,101)), data_path+subject+'/'+subject+'-fwd.fif', labels, inv_method, labels_unwanted, np.inf, \
        labels, True, inv_function, data_path+subject+'/'+subject+'-cov.fif', data_path+subject+'/'+subject+'-ave.fif'
    R_emp, R_anal, R_points = evaler.get_R(inp)
    return R_anal, R_emp, R_points

def get_empirical_R(data_path, subjects, inv_methods, SNRs, waveform, inv_function=None, n_jobs=1):
    """Gets empirical resolution matrix.
    
    Parameters
    ----------
    data_path : string
        Path to subject data folder.
    subject : string
    inv_methods : list
        List containing strings "MNE" | "dSPM" | "sLORETA" | "eLORETA" |  "mixed_norm"
        Which inverse methods to test.
    SNRs : list
        List of floats specifying which SNRs to test.
    waveform : array, shape (n_active_sources, n_time)
        Waveform of activation(s).
    inv_function : function handle
        User-defined function that returns the estimate. See documentation in
        for_manuscript.py.
    n_jobs : int
        Number of threads to run in parallell.

    Returns
    -------
    R_emp : dictionary 
        Dictionary containing; 
            - r_master, empirical resolution matrix; array of shape (n_labels, 
                n_labels).
            - r_master_point_patch, empirical resolution matrix without grouping
                of receiving sources; array of shape (n_vertices, n_labels).
            - roc_stats, dictionary containing ROC stats.
        
    """
    R_emp = {}
    for subject in subjects:
        labels, labels_unwanted = pickle.load(open(data_path+subject+'/labels', 'rb'))
        fname_fwd = data_path+subject+'/'+subject+'-fwd.fif'
        r_m, r_mpp = evaler.get_r_master(SNRs, waveform, fname_fwd, labels, inv_methods, labels_unwanted,
                                         data_path+subject+'/'+subject+'-cov.fif', data_path+subject+'/'+subject+'-ave.fif',
                                         labels, inv_function, n_jobs)
        roc_stats = evaler.get_roc_statistics(r_m, inv_methods)
        prc_stats = evaler.get_prc_statistics(r_m, inv_methods)
        R_emp.update({subject : {'r_master' : r_m, 
                                 'r_master_point_patch' : r_mpp, 
                                 'roc_stats' : roc_stats,
                                 'prc_stats' : prc_stats}})
    return R_emp

def get_average_R(R_emp):
    """Gets average of empirical resolution matrix across subjects.
    
    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.

    Returns
    -------
    r_master : dictionary 
        Dictionary containing average of resolution matrices across subjects.
        
    """
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    SNRs = R_emp[list(R_emp.keys())[0]]['r_master']['SNRs']
    r_master = {'SNRs' : SNRs}
    for method in inv_methods:
        r_master_subject = np.array([]).reshape(1000,1000,len(SNRs),0)
        for subject in list(R_emp.keys()):
            r_master_subject = np.concatenate((r_master_subject, R_emp[subject]['r_master'][method].reshape(1000,1000,len(SNRs),1)/
                                               np.sum(R_emp[subject]['r_master'][method])), axis=3)
        r_master.update({method : np.mean(r_master_subject, axis=3)})
    return r_master

def get_resolution_metrics(R_emp, data_path):
    """Calculates resolution metrics.
    
    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.
    data_path : string
        Path to subject data folder.

    Returns
    -------
    res_metrics : dictionary 
        Dictionary with keys 'PE' and 'SD', corresponding to peak localization
        error and spatial dispersion, respectively.
        
    """
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    n_subj = len(list(R_emp.keys()))
    n_labels = R_emp[list(R_emp.keys())[0]]['r_master'][inv_methods[0]].shape[0]
    SNRs = R_emp[list(R_emp.keys())[0]]['r_master']['SNRs']
    res_metrics = {}
    
    for a, method in enumerate(inv_methods):
        PE_method = np.zeros((n_subj, n_labels, len(SNRs)))
        SD_method = np.zeros((n_subj, n_labels, len(SNRs)))
        
        for d, subject in enumerate(list(R_emp.keys())):
            fwd = mne.read_forward_solution( data_path+subject+'/'+subject+'-fwd.fif')
            labels, labels_unwanted = pickle.load(open(data_path+subject+'/labels', 'rb'))
            src = evaler.correct_fwd(fwd, labels_unwanted)['src']
            src_space_sphere = mne.read_source_spaces(data_path+subject+'/'+subject+'sphere-src.fif')
            
            for c, SNR in enumerate(SNRs):
                R_vl = R_emp[subject]['r_master_point_patch'][method][:,:,c]
                PE_method[d, :, c] = evaler.get_peak_dipole_error(R_vl, src, src_space_sphere, labels)
                SD_method[d, :, c] = evaler.get_spatial_dispersion(R_vl, src, labels)
        res_metrics.update({method : {'PE' : PE_method, 'SD' : SD_method}})
    return res_metrics

def get_classifier_curve_stats(R_emp, curve='roc_stats'):
    """Calculates ROC or PRC resolution metrics.
    
    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.
    curve : 'roc_stats' | 'prc_stats'
        String indicating whether to calculate receiver-operator or precision-recall curves.

    Returns
    -------
    roc_stats : dictionary 
        Dictionary containing min, mean and max on ROC curve across subjects.
    roc_stats_all_subjects : dictionary
        Dictionary containing ROC stats across subjects.
        
    """
    if not curve in ['roc_stats', 'prc_stats']:
        raise ValueError('Curve argument has to be either roc_stats or prc_stats')
    roc_stats = {}
    roc_stats_all_subjects = {}
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    subjects = list(R_emp.keys())
    for metrics in list(R_emp[subjects[0]][curve].keys())[0:2]:
        methods_dir = {}
        methods_dir_subjects = {}
        for method in inv_methods:
            r_master_subject = np.array([]).reshape(np.array(R_emp[subjects[0]][curve][metrics][method]).shape+(0,))
            for subject in list(R_emp.keys()):
                r_master_subject = np.concatenate((r_master_subject, np.array(R_emp[subject][curve][metrics][method]). \
                                                   reshape(np.array(R_emp[subject][curve][metrics][method]).shape+(1,))), \
                                                   axis=len(r_master_subject.shape)-1)
            scals = np.array([np.nanmin(r_master_subject, axis=len(r_master_subject.shape)-1), 
                              np.nanmean(r_master_subject, axis=len(r_master_subject.shape)-1), 
                              np.nanmax(r_master_subject, axis=len(r_master_subject.shape)-1)])
            methods_dir.update({method : scals})
            methods_dir_subjects.update({method : r_master_subject})
        roc_stats.update({metrics : methods_dir})
        roc_stats_all_subjects.update({metrics : methods_dir_subjects})
    return roc_stats, roc_stats_all_subjects



###############################################################
# Plot example results

def plot_label_activation(R_emp, inv_method, subj_data, label_ind, SNR_ind):
    """ Plots example psf and coge for label_ind over cortex at SNR_ind.

    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.
    inv_method : "MNE" | "dSPM" | "sLORETA" | "eLORETA" |  "mixed_norm"
        Inverse method to plot.
    subj_data : instance of subject data
    label_ind : int
        Index of label in subj_data['labels'] whose activation to plot.
    SNR_ind : int
        Index of SNR in SNRs to plot.

    Returns
    -------
    brain : mlab plot 
        Plot of psf resulting from activation with activated and label closest 
        to center of gravity marked.
        
    """
    
    brain = evaler.plot_topographic_parcellation(scalar=R_emp[subj_data['subject']]['r_master'][inv_method][:,label_ind,SNR_ind],
                                  settings=subj_data['settings'], labels=subj_data['labels'], clims = [90,95,100], transparent=True, hemi='both')[0]
    brain.add_label(label=subj_data['labels'][label_ind],borders=10,color='green',alpha=0.6) 
    cog_closest_source = evaler.get_center_of_gravity_error(R=R_emp[subj_data['settings']]['r_master'][inv_method][:,:,SNR_ind],
                                                 src=subj_data['fwd']['src'], labels=subj_data['labels'])[2][label_ind]
    brain.add_label(label=subj_data['labels'][cog_closest_source],borders=10,color='blue',alpha=0.6)
    return brain

###############################################################
# Plot average of subjects results

# Plot median source metrics over cortex for SNR_ind
def plot_res_metrics_topo(R_emp, src, SNRs, res_metrics, labels, SNR_ind, data_path, 
                          subject, data_dir='', save_stc=False, plot=False):
    """ Plots example psf and coge for label_ind over cortex at SNR_ind.

    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.
    src : list
        List of source space objects.
    SNRs : list
        List of SNRs.
    res_metrics: dictionary
        Resolution metrics as returned by get_resolution_metrics.
    labels : list
        List of labels.
    SNR_ind : int
        Index in SNRs to plot.
    data_path : string
        Path to subject data. 
    subject : string
        Name of subject.
    data_dir = string
        Path to directory to save stc if save_stc is True.

    Returns
    -------
    brain : mlab plot 
        Plot of resolution metrics.
        
    """
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    for inv_method in inv_methods:
        for c, metric in enumerate(['SD', 'PE']):
            scals = np.median(res_metrics[inv_method][metric][:,:,SNR_ind], axis=0).reshape(len(labels),1)
            stc = mne.simulation.simulate_stc(src, labels, scals, tmin=0, tstep=1)
            if plot:
                brain = stc.plot(subjects_dir=data_path, subject=subject, smoothing_steps=3, 
                                 transparent=True, title=inv_method+' '+metric+' '+str(SNRs[SNR_ind]),
                                 clim = {'kind' : 'value', 'lims' : [(0., 5.0, 7.0), (0., 0.75, 4.)][c]})
            if save_stc:
                stc.save(data_dir + metric + '_' + inv_method + '_' + str(SNRs[SNR_ind]))
    if plot:
        return brain
    return 


# Plot cumulative histograms
def plot_res_metrics_hist(res_metrics, inv_methods, SNR_ind):
    """ Plots cumulative histograms of resolution metrics for SNR determined by
    SNR_ind.

    Parameters
    ----------
    res_metrics: dictionary
        Resolution metrics as returned by get_resolution_metrics.
    inv_methods : list
        List of inverse methods to plot.
    SNR_ind : int
        Index in SNRs to plot (SNR=SNRs[SNR_ind]).

    Returns
    -------
    fig : Figure  
        Histogram plot.
        
    """
    for c, metric in enumerate(['SD', 'PE']):
        data_to_plot = {}
        for inv_method in inv_methods:
            sorted_medians = np.sort(np.median(res_metrics[inv_method][metric][:, :, SNR_ind], axis=0))
            data_to_plot.update({inv_method : np.insert(sorted_medians[::50], 20, sorted_medians[999])})
        fig = evaler.cumulative_plot(data_to_plot, ['Spatial dispersion SD (cm)', 'Localization error PE (cm)'][c], 
                                     cutoff = 1., plot_cutoff_line=False, labels=inv_methods, log_scale=False)
        plt.tight_layout()
        plt.grid(linestyle='--', alpha=0.2)
    return fig
    
 
def plot_medians(R_emp, res_metrics, figure_labels, metric_labels, SNRs_to_plot):
    """ Plots the median of resolution metrics for each SNR across subjects p/m
    one standard error.

    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.
    res_metrics: dictionary
        Resolution metrics as returned by get_resolution_metrics.
    figure_labels : list
        List containing the legend labels for the figures.
    metric_labels : list
        List containing the metrics, will be plotted as y-labels.
    SNRs_to_plot : list
        List of floats containing SNRs to plot. Will be the same as SNRs except
        for when SNRs=0 and inf.

    Returns
    -------
    fig : Figure  
        Medians plot.
        
    """
    import matplotlib.ticker as mticker
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    subjects = list(R_emp.keys())
    colors = evaler.generate_colors(len(inv_methods))
    markers = evaler.generate_markers(len(inv_methods))
    plot_SNR_0_inf = True
    markersize = 5

    for b, metric in enumerate(['PE', 'SD']):
        dpi,figsize = evaler.figure_preferences()
        fig = plt.figure(dpi=dpi,figsize=figsize)
        for c, inv_method in enumerate(inv_methods):
            median_sources = np.median(res_metrics[inv_method][metric], axis=1)
            median_subjects = np.median(median_sources, axis=0)
            std_error_subj = np.std(median_sources, axis=0)/len(subjects)
            plt.semilogx(SNRs_to_plot, median_subjects + 2*std_error_subj, '-',
                         color=colors[c], alpha=0.1, linewidth=.5)
            plt.semilogx(SNRs_to_plot, median_subjects - 2*std_error_subj, '-',
                         color=colors[c], alpha=0.1, linewidth=.5)
            plt.fill_between(SNRs_to_plot, median_subjects - 2*std_error_subj,
                             median_subjects + 2*std_error_subj, color=colors[c], 
                             alpha=0.05)
            if plot_SNR_0_inf:
                plt.semilogx(SNRs_to_plot[1:len(SNRs_to_plot)-1], median_subjects[1:len(SNRs_to_plot)-1],
                                          '-', marker=markers[c], markersize=markersize, color=colors[c], label=figure_labels[c], linewidth=1.)
                plt.semilogx(SNRs_to_plot[0:2], median_subjects[0:2], '--', marker=markers[c], markersize=markersize, color=colors[c], linewidth=1.)
                plt.semilogx(SNRs_to_plot[len(SNRs_to_plot)-2:len(SNRs_to_plot)], 
                                          median_subjects[len(SNRs_to_plot)-2:len(SNRs_to_plot)],
                                          '--', marker=markers[c], markersize=markersize, color=colors[c], linewidth=1.)
            else:
                plt.semilogx(SNRs_to_plot, median_subjects, '-', marker=markers[c],
                              markersize=markersize, color=colors[c], label=figure_labels[c])
        plt.legend()
        plt.ylabel(metric_labels[b])
        plt.xlabel('SNR')
#        plt.ylim(0,9)
        plt.grid(linestyle='--', alpha=0.2)
        ax=plt.gca()
        a = mticker.ScalarFormatter()
        ax.xaxis.set_major_formatter(a)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
#        # Pause and copy-paste here... (pyplot LOL)
#        if 10**-5 in SNRs_to_plot:
#            evaler.change_tick('x', 10**-5, '0')
#        if 10**3 in SNRs_to_plot:
#            evaler.change_tick('x', 10**3, r'$\infty$', n_ticks_inv=0)
        plt.xlim(1.001*np.min(SNRs_to_plot), 0.999*np.max(SNRs_to_plot))
        plt.tight_layout()
    return fig

def plot_roc_auc(R_emp, roc_stats, SNR_ind, figure_labels, SNRs_to_plot, 
                 plot_limits=False, plot_SNR_0_inf=True, curve='roc'):
    """ Plots the ROC curves for one SNR and AUC as a function of SNR.

    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.
    roc_stats : dictionary 
        Dictionary containing ROC stats across subjects (roc_stats_all_subjects
        as returned from get_classifier_curve_stats).
    SNR_ind : int
        Index of SNR at which ROC will be plotted.
    figure_labels: list
        List of strings of legend labels, usually methods.
    SNRs_to_plot : list
        List of floats containing SNRs to plot. Will be the same as SNRs except
        for when SNRs=0 and inf.
    plot_limits : boolean
        If True, will plot with dashed line style between first and second data
        points as well as between second-to-last and last data points to 
        indicate that there is a discontinuity on the x-axis between these points.
    plot_SNR_0_inf : boolean
        If True, will replace the first and last ticks by 0 and inf, respectively.
    curve : 'roc' | 'prc'
        String indicating whether you want to plot roc or prc curves.

    Returns
    -------
    h : Figure  
        ROC plot.
    h2 : Figure  
        AUC plot.
        
    """
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    roc_to_plot = [np.mean(roc_stats[curve][inv_method][SNR_ind, :, :, :], axis=2) for inv_method in inv_methods]
    step = int(roc_to_plot[0].shape[1]/20)
    roc_to_plot = [np.insert(arr[:,::step], 20, arr[:,arr.shape[1]-1],axis=1) for arr in roc_to_plot]
    h = evaler.plot_roc(roc_to_plot, labels=figure_labels, curve=curve)
    plt.grid(linestyle='--', alpha=0.2)
    auc = {}
    for method in list(roc_stats['acu'].keys()):
        auc.update({method : np.array([np.min(roc_stats['acu'][method], axis=1), 
                                      np.mean(roc_stats['acu'][method], axis=1), 
                                      np.max(roc_stats['acu'][method], axis=1)])})
    h2 = evaler.plot_auc(auc, SNRs_to_plot, plot_limits=plot_limits, plot_SNR_0_inf=plot_SNR_0_inf, curve=curve)
    plt.grid(linestyle='--', alpha=0.2)
    plt.xlim(1.001*np.min(SNRs_to_plot), 0.999*np.max(SNRs_to_plot))
    plt.tight_layout()
    return h, h2

def plot_auc_sigmoid_fit(R_emp, roc_stats, SNRs_to_plot):
    """ Plots AUC and a fitted sigmoidal curve.

    Parameters
    ----------
    R_emp : dictionary
        Dictionary containing empirical resolution matrix data.
    roc_stats : dictionary 
        Dictionary containing ROC stats across subjects (roc_stats as returned 
        from get_classifier_curve_stats).
    SNRs_to_plot : list
        List of floats containing SNRs to plot. Will be the same as SNRs except
        for when SNRs=0 and inf.

    Returns
    -------
    h : Figure  
        AUC plot.
    r2 : float  
         Pearson product-moment correlation coefficient.
        
    """
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    SNRs = R_emp[list(R_emp.keys())[0]]['r_master']['SNRs']
    for inv_method in inv_methods:
        x, auc_analytical, auc_numerical, h = evaler.plot_auc_fit(roc_stats['acu'],
                                                                  SNRs_to_plot, inv_method)
        x_inds = np.array([np.argmin(np.abs(x-SNR)) for SNR in SNRs])
        r2 = np.corrcoef(auc_analytical[x_inds], auc_numerical)**2
        print('r2 = '+str(r2))
    return h, r2
