#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:12:55 2019

@author: ju357
"""

import mne
import numpy as np


def get_forward_model(subjects_dir = '/autofs/cluster/fusion/data/FreeSurfer', \
                      subject = 'jgs-20160519', meg_dir = '/autofs/cluster/fusion/data/MEG_EEG/john/170505/', \
                      raw_fname = 'assr_43_223_si_2_raw-1.fif', trans_fname = 'trans.fif', mindist = 5.0, eeg = True):
    """ 
    Creates a forward model with fixed dipole orientations. If default is being called, the resulting forward model 
    will be for a healthy 25 year old male subject in a 306 channel Elekta system with an MEG-compatible 58 channel EasyCap EEG.
    """
    
    conductivity=(0.3, 0.006, 0.3)
    raw = mne.io.read_raw_fif(meg_dir+raw_fname,preload=True)
    src = mne.setup_source_space(subject, spacing='ico5',subjects_dir=subjects_dir,add_dist=False)
    model = mne.make_bem_model(subject=subject, ico=4,conductivity=conductivity,subjects_dir=subjects_dir) 
    bem = mne.make_bem_solution(model) 
    fwd = mne.make_forward_solution(raw.info, meg_dir+trans_fname, src, bem=bem, mindist=mindist, eeg=eeg, n_jobs=1)
    fwd_fix = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=False)
    
    return raw,fwd_fix
    

def forward_operation(fwd,f,t,verts,fs):
    """
    Does a forward operation. Input function f is the waveform, t is the time and verts are the vertices associated
    with each waveform.
    """
    
    G = fwd['sol']['data'][:,verts]
    if len(G.shape) == 1:
        G = G.reshape(G.shape[0],1)
    if len(f.shape) == 1:
        f = f.reshape(1,f.shape[0])
    signal = np.dot(G,f)
    
    return signal


def get_noise(raw, events, event_duration, n_t, n_epochs):
    """
    Returns background brain activity from interstimulus data that can be superposed with simulated data. In case of bad 
    channels in raw, the data in the closest good channel (of same type) will be used.
    ----------------
    Inputs:
    raw = the raw file containing the trigger channel
    events = events object
    event_duration = duration of events
    n_t = number of timesteps in one epoch
    n_epochs = number of total epochs of noise wanted
    Outputs:
        noise = the noise in shape (n_sensors,n_t*n_epochs)
    """
    
    
    
    raw.pick_types(meg=True, eeg=True, exclude=[])
    raw_data = raw.get_data()
    fs = raw.info['sfreq']
    n_events = events.shape[0]
    n_tot = int(n_t*n_epochs)
    n_channels = len(mne.pick_types(raw.info, meg=True, eeg=True, exclude=[]))
    
    noise = np.array([]).reshape(n_channels,0)
    event_inds = events[:,0] - events[0,0]
    c = 0
    
    while noise.shape[1]<n_tot: 
        i = np.remainder(c,n_events-1)
        add_on = np.zeros((n_channels,int(0.5*fs)))
        add_on = raw_data[:,event_inds[i]+int(event_duration*fs):event_inds[i]+int((event_duration+0.5)*fs)]
        noise = np.concatenate((noise,add_on),axis=1)
        c = c + 1
    noise = noise[:,0:n_tot]
    
    return noise


def get_raw_noise(raw, events, event_duration, n_t):
    """
    Creates background brain activity (noise) from interstimulus data that can be superposed with simulated data.
    Returns a noise file as raw and noise covariance object.
    ----------------
    Inputs:
        raw = the raw file containing the trigger channel
        events = events object
        event_duration = duration of events
        n_t = number of timesteps in one epoch
    Outputs:
        raw_noise = raw object - noise
        noise_cov = noise covariance object
    """

    
    c=1
    event_inds = events[:,0] - raw.first_samp
    event_times = event_inds/raw.info['sfreq']
    raw_temp = raw.copy()
    if event_times[0] > n_t/raw.info['sfreq']:
        raw_temp.crop(tmin=0, tmax=(n_t-1)/raw.info['sfreq'])
    else:
        raw_temp.crop(tmin=0, tmax=event_times[0])
    t_steps = len(raw_temp.times)
    raws = [raw_temp]
    
    while t_steps < n_t:
        t_steps = t_steps+len(raw_temp.times)
        if t_steps > n_t:
            tmax = event_times[c-1]+event_duration + (t_steps - n_t - 1)/raw.info['sfreq']
        else:
            tmax = event_times[c]
        raw_temp = raw.copy()
        raw_temp.crop(tmin=event_times[c-1]+event_duration, tmax=tmax)
        raws.append(raw_temp)
        c = c+1
    
    raw_noise = mne.concatenate_raws(raws)
    
    
    return raw_noise


def oscillation(t,f=6.0):
    return np.sin(f*2*np.pi*t)


def simulate_activation(t_noise, t_event, labels, verts, settings, dipole_waveforms='oscillation', SNR=3.0, n_epochs = 20, plot=False):
    """
    Simulates activation (dipole wave_forms) at location specifed by verts or labels if verts is an empty list, and calculates their output in sensor space 
    superposed with noise taken from intertrial real data.
    -----------
    Inputs:
        t_noise = duration of event of data from which the noise we be loaded. Noise will be loaded starting after t_noise
        t_event = duration of whole new event, including prestim
        dipole_waveforms = array of dipole waveforms of shape (len(dipoles), len(times)) or (len(labels), len(times)). If nothing is provided, 
            it will activate an oscillation of frequency f=6 Hz.
        verts = list of vertices to be activated (give empty list if using labels)
        labels = list of labels to be activated (give empty list if using vertices)
        SNR = SNR of simulations, defined as the quota of the total energy of the simulated signal to noise
        n_epochs = desired amount of epochs to be simulated
        plot = plot simulated signal in sensor space (topographic view), with noise in sensor space (topographic view), activated vertices in source space
    """  
    
    #Check that vertices or labels have been given and get vertices from labels
    if len(verts) == 0:
        if len(labels) == 0:
            raise Exception('Must provide either vertices or labels for site of activation! Breaking.')
        else:
            n = len(labels)
            verts = []
            for label in labels:
                verts.append(label.vertices)

    #Turn vertices into a nested list of vertices            
    if len(labels) == 0:
        n = len(verts)
        verts = [[vert] for vert in verts]

    #Load raw file and set time
    raw = mne.io.read_raw_fif(settings.fname_raw(),preload=True)
    fs = raw.info['sfreq']
    t = np.linspace(0.,t_event,int(fs*(t_event-0.))+1)
    n_t = len(t)

    #Check that the number of dipole wave forms is the same as the number of vertices or labels  
    if dipole_waveforms == 'oscillation':
        dipole_waveforms = np.zeros((n,len(t)))
        dipole_waveforms = oscillation(t=np.repeat(t.reshape(1,len(t)),repeats=n,axis=0))
        
    if dipole_waveforms.shape[0] != len(verts):
        raise Exception('dipole_waveforms must be of the same size as verts')
            
    #Compute forward model
    fwd = mne.read_forward_solution(settings.fname_fwd())
    fwd = mne.convert_forward_solution(fwd,surf_ori=True,force_fixed=True)
    
    #Find vertices that are active in forward model. If none, take closest vertex.
    wave_forms = np.array([]).reshape(0,dipole_waveforms.shape[1])
    active_verts = np.array([])
    for c,vertex_group in enumerate(verts):
        if len(vertex_group) == 1:
            verts_in_fwd = np.array([np.argmin(np.abs(fwd['src'][0]['vertno']-vertex_group[0]))])
            wave_forms = dipole_waveforms
        else:
            in_fwd = vertex_group[np.isin(vertex_group,fwd['src'][0]['vertno'])]
            verts_in_fwd = np.array([np.argmin(np.abs(fwd['src'][0]['vertno']-vert)) for vert in in_fwd])
            temp = np.repeat(dipole_waveforms[c,:].reshape(1,len(dipole_waveforms[c,:])),repeats=len(verts_in_fwd),axis=0)
            wave_forms = np.concatenate((wave_forms,temp),axis=0)
            
        active_verts = np.concatenate((active_verts,verts_in_fwd),axis=0).astype(int)
        
    #Filter noise data from raw and find epochs and time
    raw.filter(l_freq=1.0,h_freq=70.0)
    events = mne.read_events(settings.fname_eve()) 
    if len(events)<n_epochs:
        print('\n number of events fewer than n_epochs, using '+str(len(events))+' events instead. \n')
        n_epochs = len(events)
    noise = get_noise(raw, events, t_noise, n_t, n_epochs)
    
    #Simulate signal
    signal = forward_operation(fwd = fwd, f=wave_forms, t=t, verts = active_verts, fs = fs)    
    if len(dipole_waveforms.shape) == 1:
        t = t.reshape(1,len(t))
        t = np.repeat(t,n,axis=0)
        signal = forward_operation(fwd = fwd, f = dipole_waveforms.reshape(1,len(dipole_waveforms)), \
                                   t=t, verts = active_verts, fs = fs)   
    
    #Build data; adjust SNR 
    data = np.zeros((signal.shape[0],0))
    trigger = np.zeros((1,0))
    chn_types = [mne.pick_types(raw.info, meg='mag', eeg=False, exclude=[]), \
                   mne.pick_types(raw.info, meg='grad', eeg=False, exclude=[]), \
                   mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])]
    for epoch in range(n_epochs):
        n_samp_pre = n_t - signal.shape[1]
        noise_epoch = noise[:,epoch*n_t:(epoch+1)*n_t]
        signal = np.concatenate((np.zeros((noise_epoch.shape[0],n_samp_pre)),signal),axis=1)
        data_epoch = np.zeros(signal.shape)
        for chs in chn_types:
            E_signal = np.sum(signal[chs,:]**2)
            E_noise = np.sum(noise_epoch[chs,:]**2)
            adjustment_factor = SNR*E_noise/E_signal
            data_epoch[chs,:] = np.sqrt(adjustment_factor)*signal[chs,:]+noise_epoch[chs,:]
        data = np.concatenate((data,data_epoch),axis=1)
        a = np.zeros((1,data_epoch.shape[1]))
        a[0] = 1
        trigger = np.concatenate((trigger,a),axis=1)
        
    #Plot data
    if plot:
        plot_event_topography(signal, raw, events, tmin=0., tmax=t_event)
        plot_event_topography(data_epoch, raw, events, tmin=0., tmax=t_event)
        plot_3d_dipoles(fwd, dipole_waveforms, t, verts)
    
    data_dir = {'mag' : data[chn_types[0],:], 'grad' : data[chn_types[1],:], 'eeg' : data[chn_types[2],:],}    
    return data_dir, signal, trigger

def save_raw(fpath, fname, raw, data_dir, trigger, write = True):
    """
    Save simulated data to fpath in raw (fname) and event files, with STI101 as the trigger channel.
    """
    data_reformat = np.zeros((len(raw.info["ch_names"]),trigger.shape[1]))

    chn_types = [mne.pick_types(raw.info, meg='mag', eeg=False, exclude=[]), \
               mne.pick_types(raw.info, meg='grad', eeg=False, exclude=[]), \
               mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])]
    data_reformat[chn_types[0],:] = data_dir['mag']
    data_reformat[chn_types[1],:] = data_dir['grad']
    data_reformat[chn_types[2],:] = data_dir['eeg']

    trigger_channel = raw.info['ch_names'].index('STI101') 
    data_reformat[trigger_channel,:] = trigger    

    raw_new = mne.io.RawArray(data_reformat,raw.info)
    events = mne.find_events(raw_new, stim_channel='STI101',consecutive=False, verbose=True, initial_event=True)
    if write:
        raw_new.save(fpath+fname+'.fif',overwrite=True)
        mne.write_events(fpath+fname+'-eve.fif', events)       
        print('Saved data to '+fpath)  
        
    print('Finished.')  
    return raw_new, events


#def patch_activation(center, radius, nave, src, fwd, raw, cov, bem, trans_file, \
#                             subject, plot_patch = False, topo_plot = False):
#    activated_verts = source_space_tools.find_verts(center=center, radius=10, src=src, plot_patch = plot_patch)
#    stc_data = np.tile(10**5./len(src['rr'])*1.*10**-9,(len(activated_verts),1))#Am/vertex
#    stc_d = np.tile(stc_data, (1,1))
#    stc_d = np.repeat(stc_d,len(activated_verts),axis=0)
#    stc_all = mne.SourceEstimate(stc_data, vertices=[activated_verts,np.array([])], \
#         tmin=tmin, tstep=tstep, subject=subject) 
#    evoked = simulate_evoked(fwd, stc_all, raw.info, cov=cov, nave=nave, iir_filter=None, random_state=None)
#    evoked.pick_types(meg=True, eeg=False,exclude='bads')
#    if topo_plot == True:
#        evoked.plot_topomap()
#    return evoked





