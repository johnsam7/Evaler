"""
Code for reproducing data and Figures for article "Spatial Fidelity of MEG/EEG Source Estimates: 
A Unified Evaluation Approach", Samuelsson et al., 2020.
"""
import evaler
import numpy as np
import mne
import pickle

# -------------------------
# Set SNR, SNRs_to_plot (for plotting 0 and inf), paths, waveform etc.
SNRs = np.hstack((0, np.logspace(-4,-2,num=7), np.logspace(-1.75, 1., num=17),
                       np.logspace(1.5, 2.0,num=2), np.inf))
SNRs[np.argmin(np.abs(SNRs-3.))] = 3.0
SNRs = list(SNRs)
SNRs_to_plot = SNRs[1:len(SNRs)-1] # SNRs_to_plot are used bc we cannot plot 0 and inf on a continuous axis
SNRs_to_plot.insert(0, 10**-5)
SNRs_to_plot.append(10**3)
subjects = ['a', 'b', 'c', 'd', 'e']
inv_methods = ['mixed_norm', 'MNE', 'dSPM', 'eLORETA', 'sLORETA']
data_path = './subj_data/' # Make sure current working folder is in the evaler folder
waveform = np.ones((1,101))
mindist = 3.0

# Create forward models for subjects
for subject in subjects:
    model = mne.read_bem_surfaces(data_path+subject+'/'+subject+'-model.fif')
    src = mne.read_source_spaces(data_path+subject+'/'+subject+'-src.fif')
    ave = mne.read_evokeds(data_path+subject+'/'+subject+'-ave.fif')[0]
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(ave.info, data_path+subject+'/'+subject+'-trans.fif', src,
                                    bem=bem, mindist=mindist, eeg=True, n_jobs=1)
    mne.write_forward_solution(data_path+subject+'/'+subject+'-fwd.fif', fwd, overwrite=True)

# -------------------------
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

# -------------------------
# Run simulation 
anal_method = 'MNE'
R_anal, R_emp_a = evaler.get_analytical_R(subjects[0], data_path, anal_method, inv_function)
R_emp = evaler.get_empirical_R(data_path, subjects, inv_methods, SNRs, waveform, inv_function, n_jobs=len(SNRs)*4)

# Get stats from resolution matrices
r_master_ave = evaler.get_average_R(R_emp)
res_metrics = evaler.get_resolution_metrics(R_emp, data_path)
roc_stats, roc_stats_all_subjects = evaler.get_roc(R_emp)

# -------------------------
# Plot Figures
subject = 'a'
labels, labels_unwanted = pickle.load(open(data_path+subject+'/labels', 'rb'))
figure_labels = inv_methods
metric_labels = ['Localization error PE (cm)', 'Spatial dispersion SD (cm)']
SNR_ind = np.argmin(np.abs(np.array(SNRs)-3))

# Analytical and empirical (lim(SNR) --> inf) resolution matrices (Figure 2)
evaler.plot_resolution_matrix(R=R_emp_a, labels=labels, title='R_emp, inf SNR, '+anal_method, SNR='inf', 
                              vrange=(0., 0.08), show_colorbar=False, show_labels=False)

evaler.plot_resolution_matrix(R=R_anal, labels=labels, title='Anaytical R, '+anal_method, SNR='inf', 
                              vrange=(0., 0.08), show_colorbar=False, show_labels=False)

# Empirical resolution matrices for MNE and MxNE (Figure 4)
method = 'MNE'
evaler.plot_resolution_matrix(R=r_master_ave[method][:,:,SNR_ind], labels=labels,title=method, 
                              SNR=SNRs[SNR_ind], vrange=(0., 0.08), show_colorbar=False, show_labels=False)
method = 'mixed_norm'
evaler.plot_resolution_matrix(R=r_master_ave[method][:,:,SNR_ind], labels=labels, title=method, 
                              SNR=SNRs[SNR_ind], vrange=(0., 0.08), show_colorbar=False, show_labels=False)

# Cumulative resolution metrics histograms (Figure 5)
fig_hist = evaler.plot_res_metrics_hist(res_metrics, inv_methods, SNR_ind)

# Median resolution metrics to SNR (Figure 6)
fig_medians = evaler.plot_medians(R_emp, res_metrics, figure_labels, metric_labels, SNRs_to_plot)

# ROC and AUC to SNR (Figure 7)
evaler.plot_roc_auc(R_emp, roc_stats_all_subjects, SNR_ind, figure_labels, SNRs_to_plot=SNRs_to_plot,
             plot_limits=False, plot_SNR_0_inf=True)
