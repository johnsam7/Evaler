
"""
Contains functions that reads fiff files and/or stc_ground_truth and return point spread, cross-talk, average error etc. 
"""

import numpy as np
from mayavi import mlab
from multiprocessing import Pool

import mne
from mne.minimum_norm import read_inverse_operator, prepare_inverse_operator
from mne.minimum_norm.inverse import _assemble_kernel
from mne.io import RawArray

from .source_space_tools import remove_overlap_in_labels, blurring
from .mne_simulation import get_raw_noise
from .settings import settings_class
from .plotting_tools import plot_topographic_parcellation


def setup(subjects_dir, subject, data_path, fname_raw, fname_fwd,
          fname_eve, fname_trans, fname_epochs, n_epochs, meg_and_eeg, plot_labels=False):
    #Save data paths in settings object
    settings = settings_class(
                 subjects_dir=subjects_dir,
                 subject=subject,
                 data_path=data_path, 
                 fname_raw=fname_raw,
                 fname_fwd=fname_fwd,
                 fname_eve=fname_eve,
                 fname_trans=fname_trans,
                 fname_epochs=fname_epochs,
                 meg_and_eeg=meg_and_eeg)
    
    #Read all labels and make them disjoint
    all_labels = mne.read_labels_from_annot(settings.subject(), 'laus500', subjects_dir=settings.subjects_dir())
    labels = []
    labels_unwanted = []
    for label in all_labels:
        if 'unknown' not in label.name and 'corpuscallosum' not in label.name:
            labels.append(label)
        else:
            labels_unwanted.append(label)
            print('Removing ' + label.name)
    
    # Reorder labels in terms of hemispheres
    labels_lh = [label for label in labels if label.hemi=='lh']
    labels_rh = [label for label in labels if label.hemi=='rh']
    labels = labels_lh + labels_rh
    
    # Load forward matrix
    fwd = mne.read_forward_solution(settings.fname_fwd())
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

    # Remove sources in unwanted labels (corpus callosum and unknown)
    sources_to_remove = np.array([])
    offset = fwd['src'][0]['nuse']
    vertnos_lh = fwd['src'][0]['vertno']
    vertnos_rh = fwd['src'][1]['vertno']
    for label in labels_unwanted:
        vertnos = label.vertices
        
        # Find vertices to remove from the gain matrix
        if label.hemi == 'lh':
            sources_to_remove = np.concatenate((sources_to_remove, np.where(np.in1d(vertnos_lh, vertnos))[0]))
            src_ind= 0
        if label.hemi == 'rh':
            sources_to_remove = np.concatenate((sources_to_remove, np.where(np.in1d(vertnos_rh, vertnos))[0] + offset))
            src_ind = 1
            
        # Correct src info
        fwd['src'][src_ind]['inuse'][vertnos] = 0
        fwd['src'][src_ind]['nuse'] = np.sum(fwd['src'][src_ind]['inuse'])
        fwd['src'][src_ind]['vertno'] = np.nonzero(fwd['src'][src_ind]['inuse'])[0]   
    source_to_keep = np.where(~np.in1d(range(fwd['sol']['data'].shape[1]), sources_to_remove.astype(int)))[0]
    fwd['sol']['data'] = fwd['sol']['data'][:,source_to_keep]
    fwd['nsource'] = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    
    # Plot labels
    if plot_labels:
        from surfer import Brain
        brain = Brain(subject_id=subject, hemi='lh', surf='inflated', subjects_dir=subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))
        brain.add_annotation('laus500')
    
    # Get epochs (n_epochs randomly picked from all available) from resting state recordings
    epochs = mne.read_epochs(settings.fname_epochs())
    epochs_to_use = np.random.choice(len(epochs), size=n_epochs, replace=False)
    
    return settings, labels, labels_unwanted, fwd, epochs_to_use


def _getInversionKernel(inverse_operator, nave=1, lambda2=1. / 9., method='MNE', label=None, pick_ori=None):
    inv = prepare_inverse_operator(inverse_operator,nave,lambda2,method)
    K = _assemble_kernel(inv,label,method,pick_ori)[0]
    return K


def _convert_real_resolution_matrix_to_labels(R, labels, label_verts):
    R_label = np.zeros((len(labels),len(labels)))
    for a, label in enumerate(labels):
        for b, label_r in enumerate(labels):
            R_label[b,a] = np.abs(np.sum(R[label_verts[label_r.name],:][:,label_verts[label.name]]))
    return R_label 


def standardize_columns(R, arg):
    if arg not in ['diag','max']:
        raise ValueError('arg must be either "diag" or "max". Breaking.')
    R_std = R.copy()
    for d in range(len(R_std.T)):
        if arg == 'diag':
            diag = R_std[d,d]
            R_std[:,d] = R_std[:,d]/diag
        if arg == 'max':
            R_std[:,d] = R_std[:,d]/np.max(R_std[:,d])
    return R_std


def standardize_rows(R):
    R_std = R.copy()
    for d in range(len(R_std)):
        diag = R_std[d,d]
        R_std[d,:] = R_std[d,:]/diag
    return R_std


def get_point_spread_matrix(R, arg):
    R_std = standardize_columns(np.abs(R), arg)
    return R_std


def get_crosstalk_matrix(R):
    R_std = standardize_rows(np.abs(R))
    return R_std


def get_R(inp):
    """
    General inverse solvers (empirical resolution)    
    """
    waveform, settings, labels, invmethod, epochs_to_use, labels_unwanted, SNR, \
        activation_labels, compute_analytical, inv_function = inp

    # Load forward matrix
    fwd = mne.read_forward_solution(settings.fname_fwd())
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

    # Remove sources in unwanted labels (corpus callosum and unknown)
    sources_to_remove = np.array([])
    offset = fwd['src'][0]['nuse']
    vertnos_lh = fwd['src'][0]['vertno']
    vertnos_rh = fwd['src'][1]['vertno']
    for label in labels_unwanted:
        vertnos = label.vertices
        
        # Find vertices to remove from the gain matrix
        if label.hemi == 'lh':
            sources_to_remove = np.concatenate((sources_to_remove, np.where(np.in1d(vertnos_lh, vertnos))[0]))
            src_ind= 0
        if label.hemi == 'rh':
            sources_to_remove = np.concatenate((sources_to_remove, np.where(np.in1d(vertnos_rh, vertnos))[0] + offset))
            src_ind = 1
            
        # Correct src info
        fwd['src'][src_ind]['inuse'][vertnos] = 0
        fwd['src'][src_ind]['nuse'] = np.sum(fwd['src'][src_ind]['inuse'])
        fwd['src'][src_ind]['vertno'] = np.nonzero(fwd['src'][src_ind]['inuse'])[0]   
    source_to_keep = np.where(~np.in1d(range(fwd['sol']['data'].shape[1]), sources_to_remove.astype(int)))[0]
    fwd['sol']['data'] = fwd['sol']['data'][:,source_to_keep]
    fwd['nsource'] = fwd['sol']['data'].shape[1]

    if activation_labels == None:
        activation_labels=labels
        
    # SNR has to be inf if we are to compute closed form R - otherwise different activations will be scaled differently
    if compute_analytical:
        SNR = np.inf
    
    # Create a dictionary linking labels with its vertices
    label_verts = {}
    for label in labels:
        if label.hemi == 'lh':
            hemi_ind = 0
            vert_offset = 0
        if label.hemi == 'rh':
            hemi_ind = 1     
            vert_offset = fwd['src'][0]['nuse']
        verts_in_src_space = label.vertices[np.isin(label.vertices,fwd['src'][hemi_ind]['vertno'])]
        inds = np.where(np.in1d(fwd['src'][hemi_ind]['vertno'],verts_in_src_space))[0]+vert_offset
        label_verts.update({label.name : inds})
        
############## Noise from epochs
    epochs = mne.read_epochs(settings.fname_epochs())
    epochs.pick_types(meg=True, eeg=True, exclude=[])
    goodies = np.array([c for c, ch in enumerate(epochs.info['ch_names']) if not ch in epochs.info['bads']])
    epochs.pick_types(meg=True, eeg=True)
    noise_epochs = epochs[[x for x in range(len(epochs)) if x not in epochs_to_use]]
    
    epochs = epochs[epochs_to_use]

    epochs_ave = epochs.average()
    noise = epochs_ave.data[:, 0:waveform.shape[1]]

    R_emp = np.zeros((len(labels),len(activation_labels)))
    R_label_vert = np.zeros((fwd['src'][0]['nuse']+fwd['src'][1]['nuse'], len(activation_labels)))

    # Find noise power for mags, grads and eeg (will be used for SNR scaling later)
    mod_ind = {'mag' : [], 'grad' : [], 'eeg' : []}
    noise_power = mod_ind.copy()
    for c, (meg, eeg) in enumerate([('mag', False), ('grad', False), (False, True)]):
        inds = mne.pick_types(epochs.info, meg = meg, eeg = eeg)
        mod_ind[list(mod_ind.keys())[c]] = inds
        noise_power[list(mod_ind.keys())[c]] = np.mean(np.linalg.norm(noise[inds, :], axis=1))
    
    # Compute noise covariance
    noise_cov = mne.compute_covariance(noise_epochs)
    
    # Create inverse operator if applied inverse method is linear
    if invmethod in ['MNE', 'dSPM', 'eLORETA', 'sLORETA']:
        inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, depth=None, fixed=True)
#        if SNR > 0:
#            lambda2 = 1. / SNR**2
#        else:
#            lambda2 = 10**9
    else:
        inverse_operator = None

    # Patch activation of each label
    for c,label in enumerate(activation_labels):
        inds = label_verts[label.name]
        G = fwd['sol']['data'][goodies, :][:, inds]
        act = np.repeat(waveform, repeats=len(inds), axis=0)*10**-8
        signal = np.dot(G,act)

        """
        Find average empirical SNR and invert to get average scaled SNR of grads, mags and EEG right. 
        Note that we take the average of the absolute value of the resolution matricx at each time point in order 
        to avoid problems with baseline correction. Therefore we should not scale the SNR with respect to the number 
        of time points in the waveform. If calculating time-averaged resolution matrix for constant activation function, 
        also scale SNR with respect to number of time points.
        """
        empirical_SNR = 0
        for key in list(mod_ind.keys()):
            signal_strength = np.mean(np.linalg.norm(signal[mod_ind[key], :], axis=1))
            empirical_SNR = empirical_SNR + 1. / 3. * signal_strength / noise_power[key]
        scaling = SNR / empirical_SNR
        if scaling == np.inf:
            sens = signal
            lambda2 = 1./9.
        else:
            sens = scaling * signal + noise 
#        sens = sens - np.repeat(np.mean(sens,axis=1).reshape(sens.shape[0],1), sens.shape[1], axis=1)
        evoked = epochs_ave.copy()
        evoked.data = sens
        evoked.set_eeg_reference('average', projection=True, verbose='WARNING')

        if invmethod == 'mixed_norm':
            estimate = mne.inverse_sparse.mixed_norm(evoked, fwd, noise_cov, alpha = 55, 
                                                        loose = 0, verbose='WARNING')
            lh_verts = estimate.vertices[0]
            rh_verts = estimate.vertices[1]
            src_inds = np.where(np.in1d(fwd['src'][0]['vertno'],lh_verts))[0]
            src_inds = np.concatenate((src_inds, np.where(np.in1d(fwd['src'][1]['vertno'],rh_verts))[0]+fwd['src'][0]['nuse']))
            
            for d,label_r in enumerate(labels):
                inds_r = label_verts[label_r.name]
                dipoles_in_label = np.where(np.isin(src_inds, inds_r))[0]
                # Let resolution matrix of the entry be the mean of that entry over all time chunks
                if dipoles_in_label.shape[0] > 0:
                    R_emp[d,c] = np.mean(np.abs(estimate.data[dipoles_in_label,:]))
            R_label_vert[src_inds,c] = np.mean(np.abs(estimate.data), axis=1)
            if np.sum(R_emp[:,c]) == 0.:
                break_point
                raise Exception('Estimate was equal to zero. Aborting.')
        else:
#            if invmethod in ['MNE', 'dSPM', 'eLORETA', 'sLORETA']:
#                estimate = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, invmethod, verbose='WARNING')
#                source = estimate.data
            # Inv_function is a user specified inverse method that maps evoked -> array (n_labels x n_labels)
            source = inv_function(evoked, SNR, invmethod, inverse_operator)
            if np.sum(np.isnan(source))>0:
                break_point
                raise Exception('Nan value found in source estimate. Aborting.')
            # Loop through each label and populate entries in resolution matrix with the mean of the absolute over time.       
            for d,label_r in enumerate(labels):
                inds_r = label_verts[label_r.name]
                # Each entry is the average of all dipole amplitudes in the patch over time
                R_emp[d,c] = np.mean(np.abs(source[inds_r,:]))
            # Resolution matrix without summing of patch vertices, resulting in shape (n_vertices, n_labels)
            R_label_vert[:,c] = np.mean(np.abs(source), axis=1)
            
        print('\n ' + settings.subject() + ', ' + invmethod + ': ' + str(c/len(activation_labels) * 100) + ' % done... \n')

    if compute_analytical:
        
        inv = mne.minimum_norm.prepare_inverse_operator(inverse_operator, nave=1, lambda2=lambda2, method=invmethod)
        if invmethod == 'MNE':
            K = mne.minimum_norm.inverse._assemble_kernel(inv,label=None,method=invmethod,pick_ori=None)[0]
        elif invmethod == 'dSPM' or invmethod == 'sLORETA':
            inv_matrices = mne.minimum_norm.inverse._assemble_kernel(inv,label=None,method=invmethod,pick_ori=None)
            K = np.dot(np.diag(inv_matrices[1].flatten()), inv_matrices[0])
        else:
            print('Resolution matrix on closed form is only available for linear methods; MNE, dSPM and sLORETA. Returning only empirical...')
            return R_emp
        R = np.dot(K, fwd['sol']['data'][goodies, :])
        R_analytical = _convert_real_resolution_matrix_to_labels(R, labels, label_verts)
        return R_emp, np.abs(R_analytical), np.abs(R)
    
    else:
        return (np.abs(R_emp), R_label_vert)


def remove_diagonal(A):
    A_off_diagonal = A.copy()
    np.fill_diagonal(A_off_diagonal, 9999999999999999)
#    outp = np.array([]).reshape(A.shape[0]-1, 0)
#    for c, column in enumerate(A.T):
#        list_column = list(column)
#        list_column.pop(c)
#        list_column = np.array(list_column)
#        outp = np.concatenate((outp, list_column.reshape(len(list_column), 1)), axis=1)
#    return outp
    bool_ind = A_off_diagonal.T==9999999999999999
    return A_off_diagonal.T[np.where(~bool_ind)].reshape(A_off_diagonal.shape[1], A_off_diagonal.shape[0]-1).T  


def get_average_cross_talk_map(R):
    R = standardize_rows(R)
    acm = np.median(R,axis=1)
    return acm


def get_average_point_spread(R,arg):
    R = standardize_columns(R,arg)
    #Remove diagonal elements when standardizing wrt max value in each column
    R_copy = remove_diagonal(R)
    acm = np.median(R_copy,axis=0)
    return acm


def get_spatial_dispersion(R, src, settings, labels):
    rr = np.concatenate((src[0]['rr'][src[0]['vertno']],
                         src[1]['rr'][src[1]['vertno']]), axis=0)
    peak_positons = rr[np.argmax(np.abs(R), axis=0), :]
#    #Hauk
#    dist = np.linalg.norm(np.repeat(rr.reshape(1, len(rr), 3), repeats=rr.shape[0], axis=0)
#                          - np.repeat(rr.reshape(rr.shape[0], 1, 3), repeats=len(rr), axis=1), axis=2)
#    SD_hauk = np.sqrt(np.divide(np.diag(np.dot(dist.T, R**2)), np.sum(R**2, axis=0)))*10
#    #Molins
#    dist = np.linalg.norm(np.repeat(rr.reshape(1, len(rr), 3), repeats=rr.shape[0], axis=0)
#                          - np.repeat(rr.reshape(rr.shape[0], 1, 3), repeats=len(rr), axis=1), axis=2)
#    SD_molins = np.sqrt(np.divide(np.diag(np.dot(dist.T**2, R**2)), np.sum(R**2, axis=0)))*100    
    #Samuelsson
    dist = np.linalg.norm(np.repeat(peak_positons.reshape(1, len(peak_positons), 3), repeats=rr.shape[0], axis=0)
                          - np.repeat(rr.reshape(rr.shape[0], 1, 3), repeats=len(peak_positons), axis=1), axis=2)
    SD = np.divide(np.diag(np.dot(dist.T, R)), np.sum(np.abs(R), axis=0))*100
#    r_dist = np.linalg.norm(np.repeat(rr.reshape((rr.shape[0], rr.shape[1], 1)), repeats=peak_positons.shape[0], axis=2) - \
#                       np.repeat(peak_positons.reshape((peak_positons.shape[0], peak_positons.shape[1], 1)), repeats=rr.shape[0], axis=2).T, axis=1)
#    mean_dist = np.mean(r_dist, axis=0)
#    SD = np.divide(SD, mean_dist)
    return SD
    

def get_spherical_coge(R, settings, labels):
    src = mne.setup_source_space(subject=settings.subject(), surface='sphere', spacing='ico5',
                                 subjects_dir=settings.subjects_dir(), add_dist=False)

    labels_divide = [[(c, label) for c, label in enumerate(labels) if label.hemi=='lh'],
                 [(c, label) for c, label in enumerate(labels) if label.hemi=='rh']]
    source_ind_hemi = [np.array([label[0] for label in labels_divide[0]]),
                       np.array([label[0] for label in labels_divide[1]])]

    spherical_coge = {'coge' : [], 'cog_vector_norm' : []}
    for c, hemi in enumerate(['lh', 'rh']):
        src_sphere = src[c]
        
        # Get vertices, move center of sphere to origo
        rr = src_sphere['rr'] - np.mean(src_sphere['rr'], axis=0)
        radius = np.mean(np.linalg.norm(rr, axis=1)) 
        
        # Get hemispherical resolution matrix and label centers
        R_hemi = R[source_ind_hemi[c],:][:, source_ind_hemi[c]]
        labels_hemi = labels_divide[c]
        label_center = []
        for label in labels_hemi:
            verts = label[1].vertices
            label_center.append(np.mean(rr[verts,:],axis=0))
        label_center = np.array(label_center)             

        # Calculate center of gravity and error on spherical surface
        for c, col in enumerate(R_hemi.T):
            center_of_gravity = np.dot(col, label_center)/np.sum(col)
            center_of_gravity = center_of_gravity*radius/np.linalg.norm(center_of_gravity)
            error = np.linalg.norm(label_center[c] - center_of_gravity)
            surface_error = 2 * radius * np.arcsin(error / (2 * radius))
            spherical_coge['cog_vector_norm'].append(np.linalg.norm(center_of_gravity) / radius)
            spherical_coge['coge'].append(surface_error)
            
    spherical_coge['cog_vector_norm'] = np.array(spherical_coge['cog_vector_norm'])
    spherical_coge['coge'] = 100*np.array(spherical_coge['coge'])    
    return spherical_coge

def get_label_center_points(settings, labels, src, src_space_sphere):
    labels_divide = [[(c, label) for c, label in enumerate(labels) if label.hemi=='lh'],
                 [(c, label) for c, label in enumerate(labels) if label.hemi=='rh']]
    
    center_points = []
    center_vertices = []
    for c, hemi in enumerate(['lh', 'rh']):
        src_sphere = src_space_sphere[c]
        
        # Get vertices, move center of sphere to origo
        rr = src_sphere['rr'] - np.mean(src_sphere['rr'], axis=0)
        
        # Get hemispherical resolution matrix and label centers
        labels_hemi = labels_divide[c]
        for label in labels_hemi:
            label_verts = label[1].vertices
            label_center = np.mean(rr[label_verts,:],axis=0)
            sources_in_label = np.array([vert for vert in label[1].vertices if vert in src[c]['vertno']])
            center_vertex = sources_in_label[np.argmin(np.linalg.norm(label_center-rr[sources_in_label, :], axis=1))]
            center_vertices.append(center_vertex + c*src[0]['np'])
            
    center_vertices = np.array(center_vertices)
    center_points = np.concatenate((src[0]['rr'], src[1]['rr']), axis=0)[center_vertices, :]
    
    return center_points, center_vertices 


def get_peak_dipole_error(R_vl, src, src_space_sphere, settings, labels):
    """
    Return error (in cm) between center of patch, found by spherical inflation, and peak reconstruction.
    """
    
    label_center_points = get_label_center_points(settings, labels, src, src_space_sphere)[0]
    max_sources = np.argmax(np.abs(R_vl), axis=0)
    rr = np.concatenate((src[0]['rr'][src[0]['vertno']], 
                         src[1]['rr'][src[1]['vertno']]), axis=0)
    errors = np.linalg.norm(rr[max_sources, :] - label_center_points, axis=1)*100
    
    return errors

        

def get_label_center(labels, src):

    label_center = []

    for label in labels:
        verts = label.vertices
        if label.hemi=='lh':
            label_center.append(np.mean(src[0]['rr'][verts,:],axis=0))
        if label.hemi=='rh':
            label_center.append(np.mean(src[1]['rr'][verts,:],axis=0))
        
    label_center = np.array(label_center)
    
    return label_center

def get_3d_error(R, src, labels, case='patch_patch'):

    if case not in ['point_patch', 'point_point', 'patch_patch']:
        raise Exception('case argument must either be point_patch, point_point or patch_patch')
    
    label_center = get_label_center(labels, src) 
    
    if case == 'point_patch':
        max_sources = np.argmax(np.abs(R), axis=0)
        rr = np.concatenate((src[0]['rr'][src[0]['vertno']], 
                             src[1]['rr'][src[1]['vertno']]), axis=0)
        errors = np.linalg.norm(rr[max_sources,:] - label_center, axis=1)

    if case == 'point_point':
        max_sources = np.argmax(np.abs(R), axis=0)
        errors = []
        rr = np.concatenate((src[0]['rr'][src[0]['vertno']], 
                             src[1]['rr'][src[1]['vertno']]), axis=0)
        for c, max_source in enumerate(max_sources):
            errors.append(np.linalg.norm(rr[max_source,:] - rr[c, :]))

    if case == 'patch_patch':
        max_sources = np.argmax(np.abs(R), axis=0)
        errors = []
        for c, max_source in enumerate(max_sources):
            errors.append(np.linalg.norm(label_center[max_source] - label_center[c]))

    return 100*np.array(errors)
    
    
def get_center_of_gravity_error(R, src, labels):        
    label_center = get_label_center(labels, src)            
    coge = []
    center_of_gravity_list = []
    cog_closest_source = []

    for c, col in enumerate(R.T):
        center_of_gravity = np.dot(col, label_center)/np.sum(col)
        error = np.linalg.norm(label_center[c] - center_of_gravity)
        coge.append(error)
        center_of_gravity_list.append(center_of_gravity)
        cog_closest_source.append(np.argmin(np.linalg.norm(np.repeat(center_of_gravity.reshape((1,3)),len(labels),axis=0)-label_center, axis=1)))
        
    return 100*np.array(coge), center_of_gravity_list, cog_closest_source


def amplitude_reconstruction_error(R):
#    ar = np.abs(np.max(R,axis=0)/np.max(R)-1)
    ar = np.abs(np.log10(np.max(R,axis=0)/np.max(R)))
    return ar


def resolution_map_parcellation(settings, R, labels, res_argument, arg='max', clims = [2,50,98], transparent=True, plot=False):
    if res_argument not in ['point_spread','cross_talk']:
        raise ValueError('res_argument has to be either point_spread or cross_talk')
    if res_argument == 'point_spread':
        acm = get_average_point_spread(R, arg)
    if res_argument == 'cross_talk':
        acm = get_average_cross_talk_map(R)
        
    if plot:
        plot_topographic_parcellation(acm, settings, labels, clims = clims, transparent=transparent)
    return acm

def resolution_map(settings, R, res_argument, arg='max', fpath='', print_surf=False):
    if res_argument not in ['point_spread','cross_talk']:
        raise ValueError('res_argument has to be either point_spread or cross_talk')
    if res_argument == 'point_spread':
        acm = get_average_point_spread(R, arg)
    if res_argument == 'cross_talk':
        acm = get_average_cross_talk_map(R)
    
    src = mne.read_forward_solution(settings.fname_fwd())['src']
    src_joined = join_source_spaces(src)
    scalars = blurring(acm, src_joined)
    brain = mlab.triangular_mesh(src_joined['rr'][:, 0], src_joined['rr'][:, 1], src_joined['rr'][:, 2], src_joined['tris'], scalars = scalars)
    if print_surf:
        if len(fpath) == 0:
            raise ValueError('Must provide fpath to where plyfile will be printed.')
        else:
            print_ply(fpath, src_joined, scalars, vmax = True, vmin = True)
    return acm,brain 


def get_r_master(SNRs, waveform, settings, labels, inv_methods, epochs_to_use, labels_unwanted,
                 fwd, activation_labels=None, inv_function=None, n_jobs=1):
    if  n_jobs < 2*len(SNRs):
        SNR_njobs = n_jobs
        activation_jobs = 1
    else:
        SNR_njobs = len(SNRs)
        activation_jobs = np.floor_divide(n_jobs, len(SNRs))
    compute_analytical = False
    r_master = {}
    r_master_vl = {}
    iterations = np.floor_divide(len(SNRs), SNR_njobs)
    N_remainder = np.mod(len(SNRs), SNR_njobs)
    vertno = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    
    SNR_iterations = [SNRs[iteration*SNR_njobs : (iteration + 1) * SNR_njobs] for iteration in range(0, iterations)]
    if not N_remainder == 0:
        SNR_iterations.append(SNRs[SNR_njobs*iterations:len(SNRs)])

    for inv_method in inv_methods:
        if activation_labels == None:
            r_tensor = np.zeros((len(labels), len(labels), len(SNRs)))
            r_tensor_vl = np.zeros((vertno, len(labels), len(SNRs)))
            activation_labels = labels
        else:
            r_tensor = np.zeros((len(labels), len(activation_labels), len(SNRs)))
            r_tensor_vl = np.zeros((vertno, len(activation_labels), len(SNRs)))
        print('Computing resolution matrices for inverse method ' + inv_method + '...')
        for group_count, SNR_group in enumerate(SNR_iterations):
            from joblib import Parallel, delayed
            myfunc = delayed(get_R)
            parallel = Parallel(n_jobs=SNR_njobs*activation_jobs)
            activation_chunks = [list(np.array_split(np.array(activation_labels),activation_jobs)[i]) 
                                            for i in range(activation_jobs)]
            inp_group = []
            for SNR in SNR_group:
                for k in range(activation_jobs):
                    inp_group.append((SNR,activation_chunks[k]))
            out = parallel(myfunc((waveform, settings, labels, inv_method, epochs_to_use, labels_unwanted,
                                   inp[0], inp[1], compute_analytical, inv_function)) for inp in inp_group)

            for c, SNR in enumerate(SNR_group):
                R_emp = np.array([]).reshape(len(labels),0)
                R_emp_vl = np.array([]).reshape(vertno,0)
                for d, activation_chunk in enumerate(activation_chunks):
                    R_emp = np.concatenate((R_emp, out[c*activation_jobs+d][0]), axis=1)
                    R_emp_vl = np.concatenate((R_emp_vl, out[c*activation_jobs+d][1]), axis=1)
                r_tensor[:, :, c + group_count*len(SNR_group)] = R_emp
                r_tensor_vl[:, :, c + group_count*len(SNR_group)] = R_emp_vl
        r_master.update({inv_method : r_tensor})
        r_master_vl.update({inv_method : r_tensor_vl})
    
    r_master_vl.update({'SNRs' : SNRs})
    r_master.update({'SNRs' : SNRs})
    return r_master, r_master_vl


def get_roc_statistics(r_master, inv_methods):
    roc_stats = {'roc' : {}, 'acu' : {}, 'all_stats' : {}}
    
    for inv_method in inv_methods:
        R_tensor = r_master[inv_method]
        roc_list = {'roc' : [], 'acu' : [], 'all_stats' : []}
        
        for c,SNR in enumerate(list(r_master['SNRs'])):
            roc, acu, all_stats = get_roc(R_tensor[:,:,c])
            roc_list['roc'].append(roc)
            roc_list['acu'].append(acu)
            roc_list['all_stats'].append(all_stats)            
            
        roc_stats['roc'].update({inv_method : roc_list['roc']})
        roc_stats['acu'].update({inv_method : roc_list['acu']})
        roc_stats['all_stats'].update({inv_method : roc_list['all_stats']})
        
    return roc_stats


def get_roc(R):
    n_T = 100
    ROC = np.zeros((2,n_T+3))    
    R_max = standardize_columns(R, arg='max')
    true_estimates = np.diag(R_max)
    R_max_off_diagonals = remove_diagonal(R_max)
    acu = []
    all_stats = {'TP' : [], 'FN' : [], 'TN' : [], 'FP' : []}

    for c,T in enumerate(np.linspace(-0.01,1.01,n_T)):
        TP = np.sum(true_estimates > T)
        FN = len(true_estimates) - TP
        TN = np.sum(R_max_off_diagonals <= T)
        FP = np.sum(R_max_off_diagonals >= T)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        ROC[0,c] = FPR
        ROC[1,c] = TPR
        all_stats['TP'].append(TP)
        all_stats['FN'].append(FN)
        all_stats['TN'].append(TN)
        all_stats['FP'].append(FP)

    for stats in list(all_stats.keys()):
        all_stats[stats] = np.array(all_stats[stats])
    
    acu = np.abs(np.trapz(y=ROC[1,:], x=ROC[0,:], dx=0.001))
    return ROC, acu, all_stats


def get_source_metrics(r_master, fwd, labels, inv_method, settings, case='patch_patch', arg='diag'):
#    source_metrics = {'psf_ave' : {}, 'coge' : {}, 'rel_amp' : {}, 'sphere_error' : {}, 
#                      '3d_error' : {}}
    source_metrics = {'psf_ave' : {}, '3d_error' : {}}

    for c, SNR in enumerate(r_master['SNRs']):
        # Get right resolution matrix from r_master
        R = r_master[inv_method][:,:,c]

        # Compute source metrics
        source_metrics['psf_ave'].update({str(SNR) : get_average_point_spread(R, arg=arg)})
        source_metrics['3d_error'].update({str(SNR) : get_3d_error(R, fwd['src'], labels, case=case)})
#        source_metrics['coge'].update({str(SNR) : get_center_of_gravity_error(R, fwd['src'], labels)[0]})
#        source_metrics['rel_amp'].update({str(SNR) : amplitude_reconstruction_error(R)})
#        sphere_met = get_spherical_coge(R, settings, labels)
#        source_metrics['sphere_error'].update({str(SNR) : sphere_met['coge']})
#        source_metrics['sphere_vector_norm'].update({str(SNR) : sphere_met['cog_vector_norm']})

    return source_metrics


def count_sources_in_labels(labels, fwd):
    vert_nrs = []
    for label in labels: 
        if label.hemi == 'lh': 
            vert_nrs.append(np.sum(np.isin(label.vertices, fwd['src'][0]['vertno']))) 
        if label.hemi == 'rh': 
            vert_nrs.append(np.sum(np.isin(label.vertices, fwd['src'][1]['vertno']))) 
    return np.array(vert_nrs)

