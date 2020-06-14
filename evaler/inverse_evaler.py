"""
Created on Wed Sep 11 17:10:00 2019

@author: ju357
"""
import evaler
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle

def setup_subject(subjects_dir, subject, data_path, n_epochs):
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
    """
    Returns analytical resolution matrix (only for linear inverse methods)
    """
    labels, labels_unwanted = pickle.load(open(data_path+subject+'/labels', 'rb'))
    inp = np.ones((1,101)), data_path+subject+'/'+subject+'-fwd.fif', labels, inv_method, labels_unwanted, np.inf, \
        labels, True, inv_function, data_path+subject+'/'+subject+'-cov.fif', data_path+subject+'/'+subject+'-ave.fif'
    R_emp, R_anal, R_points = evaler.get_R(inp)
    return R_anal, R_emp

def get_empirical_R(data_path, subjects, inv_methods, SNRs, waveform, inv_function=None, n_jobs=1):
    """
    Returns empirical resolution matrices.
    """
    R_emp = {}
    for subject in subjects:
        labels, labels_unwanted = pickle.load(open(data_path+subject+'/labels', 'rb'))
        fname_fwd = data_path+subject+'/'+subject+'-fwd.fif'
        r_m, r_mpp = evaler.get_r_master(SNRs, waveform, fname_fwd, labels, inv_methods, labels_unwanted,
                                         data_path+subject+'/'+subject+'-cov.fif', data_path+subject+'/'+subject+'-ave.fif',
                                         labels, inv_function, n_jobs)
        roc_stats = evaler.get_roc_statistics(r_m, inv_methods)
        R_emp.update({subject : {'r_master' : r_m, 
                                 'r_master_point_patch' : r_mpp, 
                                 'roc_stats' : roc_stats}})
    return R_emp

def get_average_R(R_emp):
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

def get_roc(R_emp):
    roc_stats = {}
    roc_stats_all_subjects = {}
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    subjects = list(R_emp.keys())
    for metrics in list(R_emp[subjects[0]]['roc_stats'].keys())[0:2]:
        methods_dir = {}
        methods_dir_subjects = {}
        for method in inv_methods:
            r_master_subject = np.array([]).reshape(np.array(R_emp[subjects[0]]['roc_stats'][metrics][method]).shape+(0,))
            for subject in list(R_emp.keys()):
                r_master_subject = np.concatenate((r_master_subject, np.array(R_emp[subject]['roc_stats'][metrics][method]). \
                                                   reshape(np.array(R_emp[subject]['roc_stats'][metrics][method]).shape+(1,))), \
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
    """
    Example psf and coge for label_ind over cortex at SNR_ind
    """
    
    brain = evaler.plot_topographic_parcellation(scalar=R_emp[subj_data['subject']]['r_master'][inv_method][:,label_ind,SNR_ind],
                                  settings=subj_data['settings'], labels=subj_data['labels'], clims = [90,95,100], transparent=True, hemi='both')[0]
    brain.add_label(label=subj_data['labels'][label_ind],borders=10,color='green',alpha=0.6) 
    cog_closest_source = evaler.get_center_of_gravity_error(R=R_emp[subj_data['settings']]['r_master'][inv_method][:,:,SNR_ind],
                                                 src=subj_data['fwd']['src'], labels=subj_data['labels'])[2][label_ind]
    brain.add_label(label=subj_data['labels'][cog_closest_source],borders=10,color='blue',alpha=0.6)
    return brain

def plot_dipolar_activation(R, inv_method, subj_data, SNRs, label_ind, SNR_ind):
    stc = mne.SourceEstimate(data=R[subj_data['subject']]['r_master_point_patch'][inv_method][:,label_ind,SNR_ind], 
                             vertices = [subj_data['fwd']['src'][0]['vertno'], 
                                         subj_data['fwd']['src'][1]['vertno']], 
                             tmin=0, tstep=1, subject=subj_data['subject'])
    brain = stc.plot(subjects_dir = subj_data['settings'].subjects_dir(), smoothing_steps=4, transparent=True, 
                     clim = {'kind' : 'percent', 'lims' : [2,50,98]}, 
                     title='method : ' + inv_method + ' ' + 'subj: ' + subj_data['subject'] + \
                     ' label_ind: ' + str(label_ind) + ' SNR: ' + str(SNRs[SNR_ind])[0:5])
    brain.add_label(label=subj_data['labels'][label_ind],borders=1,color='blue',alpha=0.8) 
    return brain

###############################################################
# Plot average of subjects results

# Plot median source metrics over cortex for SNR_ind
def plot_res_metrics_topo(R_emp, src, SNRs, res_metrics, labels, SNR_ind, data_path, subject, data_dir='', save_stc=False):
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    for inv_method in inv_methods:
        for c, metric in enumerate(['SD', 'PE']):
            scals = np.median(res_metrics[inv_method][metric][:,:,SNR_ind], axis=0).reshape(len(labels),1)
            stc = mne.simulation.simulate_stc(src, labels, scals, tmin=0, tstep=1)
            brain = stc.plot(subjects_dir=data_path, subject=subject, smoothing_steps=3, transparent=True, title=inv_method+' '+metric+' '+str(SNRs[SNR_ind]),
                             clim = {'kind' : 'value', 'lims' : [(0., 5.0, 7.0), (0., 0.75, 4.)][c]})
            if save_stc:
                stc.save(data_dir + metric + '_' + inv_method + '_' + str(SNRs[SNR_ind]))
    return brain


# Plot cumulative histograms
def plot_res_metrics_hist(res_metrics, inv_methods, SNR_ind):
    for c, metric in enumerate(['SD', 'PE']):
        data_to_plot = {}
        for inv_method in inv_methods:
            data_to_plot.update({inv_method : np.median(res_metrics[inv_method][metric][:, :, SNR_ind], axis=0)})
        fig = evaler.cumulative_plot(data_to_plot, ['Spatial dispersion SD (cm)', 'Localization error PE (cm)'][c], 
                                     cutoff = 1., plot_cutoff_line=False, labels=inv_methods, log_scale=False)
        plt.tight_layout()
        plt.grid(linestyle='--', alpha=0.2)
    return fig
    
 
def plot_medians(R_emp, res_metrics, figure_labels, metric_labels, SNRs_to_plot):
    """
    Plot median values for localization error and point spread. 
    10**-5 in SNRs_to_plot will be replaced with 0 and 10**3 with inf.
    """
    import matplotlib.ticker as mticker
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    subjects = list(R_emp.keys())
    colors = evaler.generate_colors(len(inv_methods))
    plot_SNR_0_inf = True
    for b, metric in enumerate(['PE', 'SD']):
        dpi,figsize = evaler.figure_preferences()
        fig = plt.figure(dpi=dpi,figsize=(5,4))
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
                                          '-', color=colors[c], label=figure_labels[c], linewidth=1.)
                plt.semilogx(SNRs_to_plot[0:2], median_subjects[0:2], '--', color=colors[c], linewidth=1.)
                plt.semilogx(SNRs_to_plot[len(SNRs_to_plot)-2:len(SNRs_to_plot)], 
                                          median_subjects[len(SNRs_to_plot)-2:len(SNRs_to_plot)],
                                          '--', color=colors[c], linewidth=1.)
            else:
                plt.semilogx(SNRs_to_plot, median_subjects, '-', color=colors[c], label=figure_labels[c])
        plt.legend()
        plt.ylabel(metric_labels[b])
        plt.xlabel('SNR')
        plt.ylim(0,9)
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

def plot_roc_auc(R_emp, roc_stats, SNR_ind, figure_labels, SNRs_to_plot, plot_limits=False, plot_SNR_0_inf=True):
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    roc_to_plot = [np.mean(roc_stats['roc'][inv_method][SNR_ind, :, :, :], axis=2) for inv_method in inv_methods]
    evaler.plot_roc(roc_to_plot, labels=figure_labels)
    plt.grid(linestyle='--', alpha=0.2)
    auc = {}
    for method in list(roc_stats['acu'].keys()):
        auc.update({method : np.array([np.min(roc_stats['acu'][method], axis=1), 
                                      np.mean(roc_stats['acu'][method], axis=1), 
                                      np.max(roc_stats['acu'][method], axis=1)])})
    evaler.plot_auc(auc, SNRs_to_plot, plot_limits=plot_limits, plot_SNR_0_inf=plot_SNR_0_inf)
    plt.grid(linestyle='--', alpha=0.2)
    plt.xlim(1.001*np.min(SNRs_to_plot), 0.999*np.max(SNRs_to_plot))
    plt.tight_layout()
    return 

def plot_auc_sigmoid_fit(R_emp, roc_stats, SNRs_to_plot):
    inv_methods = list((R_emp[list(R_emp.keys())[0]]['r_master']).keys())
    inv_methods.remove('SNRs')
    SNRs = R_emp[list(R_emp.keys())[0]]['r_master']['SNRs']
    for inv_method in inv_methods:
        x, auc_analytical, auc_numerical, h = evaler.plot_auc_fit(roc_stats['acu'],
                                                                  SNRs_to_plot, inv_method)
        x_inds = np.array([np.argmin(np.abs(x-SNR)) for SNR in SNRs])
        r2 = np.corrcoef(auc_analytical[x_inds], auc_numerical)**2
        print(r2)
    return 
