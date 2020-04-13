"""
Example API for use of Evaler repo.
"""
import evaler
import numpy as np
import mne

# Set SNR, SNRs_to_plot (for plotting 0 and inf), paths, waveform etc.
SNRs = np.hstack((0, np.logspace(-4,-2,num=7), np.logspace(-1.75, 1., num=17),
                       np.logspace(1.5, 2.0,num=2), np.inf))
SNRs = SNRs[0:len(SNRs):2]
SNRs = np.insert(SNRs,len(SNRs),np.inf)
SNRs[np.argmin(np.abs(SNRs-3.))] = 3.0
SNRs = list(SNRs)
SNRs_to_plot = SNRs[1:len(SNRs)-1]
SNRs_to_plot.insert(0, 10**-5)
SNRs_to_plot.append(10**3)
subjects = ['awmrc_004', 'awmrc_008', 'awmrc_010', 'awmrc_011', 'awmrc_021']
inv_methods = ['mixed_norm', 'MNE', 'dSPM', 'eLORETA', 'sLORETA']
subjects_dir = '/autofs/cluster/fusion/data/resting_state/subjects_mri/'
data_path = '/autofs/cluster/fusion/data/resting_state/preprocessed/subjects/'
n_epochs = 49
waveform = np.ones((1,101))

# Source estimation method to test
def inv_function(evoked, SNR, linear_method=False, inverse_operator=None):
    if linear_method in ['MNE', 'dSPM', 'eLORETA', 'sLORETA']:
        if SNR > 0:
            lambda2 = 1. / SNR**2
        if SNR == 0:
            lambda2 = 10**9
        if SNR == np.inf:
            lambda2 = 1/9
        estimate = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2,
                                                  linear_method, verbose='WARNING')
        return estimate.data
    else:
        # source = (result of your inverse method; np array of shape n_verts x n_times)
        return source

# Generate data
R_anal = evaler.get_analytical_R(subjects[0], subjects_dir, data_path, inv_methods[1], 
                          n_epochs, waveform, inv_function)
R_emp = evaler.get_empirical_R(subjects_dir, data_path, subjects, inv_methods, SNRs, 
                        n_epochs, waveform, inv_function, n_jobs=len(SNRs)*4)
r_master_ave = evaler.get_average_R(R_emp)
res_metrics = evaler.get_resolution_metrics(R_emp, subjects_dir, data_path, n_epochs)
roc_stats, roc_stats_all_subjects = evaler.get_roc(R_emp)

# Population plots
SNR_ind = 10
figure_labels = inv_methods
metric_labels = ['Localization error PE (cm)', 'Spatial dispersion SD (cm)']
subj_data = evaler.setup_subject(subjects_dir, subjects[0], data_path, n_epochs)
method = 'MNE'
evaler.plot_resolution_matrix(R=r_master_ave[method][:,:,SNR_ind], labels=subj_data['labels'],
                              title=method, SNR=SNRs[SNR_ind], vrange=(0., 0.08),
                              show_colorbar=False, show_labels=False)
fig_hist = evaler.plot_res_metrics_hist(res_metrics, inv_methods, SNR_ind)
fig_medians = evaler.plot_medians(R_emp, res_metrics, figure_labels, metric_labels, SNRs_to_plot)
evaler.plot_roc_auc(R_emp, roc_stats_all_subjects, SNR_ind, figure_labels, SNRs_to_plot=SNRs_to_plot,
             plot_limits=False, plot_SNR_0_inf=True)
evaler.plot_auc_sigmoid_fit(R_emp, roc_stats, SNRs_to_plot)
brain_topo = evaler.plot_res_metrics_topo(R_emp, SNRs, subj_data['settings'], res_metrics, 
                                   subj_data['labels'], SNR_ind, data_dir='', save_stc=False)

# Subject specific plots
brain = evaler.plot_label_activation(R_emp, inv_methods[1], subj_data, label_ind, SNR_ind)
evaler.circle_plot_resolution(center_labels=center_labels, R=R['rmaster'][inv_method][:,:,R['rmaster'][inv_method].shape[2]-1],
                              labels=labels, threshold=0.5, graph_threshold=0.5)

## Load data
#from sklearn.externals.joblib import load
#data_path = '/autofs/cluster/fusion/john/data/'
#R_emp = load(data_path + 'R_master_subjects_new')
