#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:49:47 2019

@author: ju357
"""

from mne.datasets import sample

###################################################################################################
# CMNESettings class
###################################################################################################

class settings_class(object):
    """the settings object

    Attributes:
        _data_path: Path to the MEG data
        _subjects_dir: Path to MRI data
        _subject: Subject ID
        _repo_path: Repository path
        _fname_raw: Raw file
        _fname_inv: Inverse file
        _fname_fwd: Forward file
        _fname_eve: Event file
        _fname_cov: Covariance file
        _fname_trans: Trans file
        _modality: The selected modality meg or meg-eeg
    """
    
    _large_memory = True # If all epoch data fit into the Ram
    
    #define default sample input
    subject = 'sample'
    fname_raw = 'sample_audvis_raw.fif'
    fname_inv = 'sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif'
    fname_fwd = 'sample_audvis-meg-eeg-oct-6-fwd.fif'
    fname_eve = 'sample_audvis_raw-eve.fif'
    fname_cov = 'sample_audvis-cov.fif'
    fname_trans = 'sample_audvis_raw-trans.fif'
    fname_epochs = ' '

    
    ###############################################################################################
    # Constructor
    ###############################################################################################
    def __init__(self,
                 subjects_dir=None,
                 subject=subject,
                 data_path=None, 
                 fname_raw=fname_raw,
                 fname_fwd=fname_fwd,
                 fname_eve=fname_eve,
                 fname_cov=fname_cov,
                 fname_trans=fname_trans,
                 fname_epochs=fname_epochs,
                 meg_and_eeg=True):
        """Return a settings object."""

        if data_path is None:
            data_path = sample.data_path() + '/MEG/sample/'
            
        if subjects_dir is None:
            subjects_dir = sample.data_path() + 'subjects'
        
        self._meg_and_eeg = meg_and_eeg
        self._data_path = data_path
        self._fname_raw = self._data_path + fname_raw
        self._fname_fwd = self._data_path + fname_fwd
        self._fname_eve = self._data_path + fname_eve
        self._fname_cov = self._data_path + fname_cov
        self._fname_trans = self._data_path + fname_trans
        self._subjects_dir = subjects_dir
        self._subject = subject
        self._fname_epochs = self._data_path + fname_epochs
                                
    
    ###############################################################################################
    # Getters and setters
    ###############################################################################################    
    def fname_epochs(self):
        """
        Returns the selected modality
        """
        return self._fname_epochs

    def meg_and_eeg(self):
        """
        Returns the selected modality
        """
        return self._meg_and_eeg
            
    def data_path(self):
        """
        Returns the data path
        """
        return self._data_path
            
    def fname_raw(self):
        """
        Returns the raw file name
        """
        return self._fname_raw
        
    def fname_fwd(self):
        """
        Returns the forward operator file name
        """
        return self._fname_fwd
    
    def fname_eve(self):
        """
        Returns the event file name
        """
        return self._fname_eve

    def fname_cov(self):
        """
        Returns the covariance file name
        """
        return self._fname_cov

    def fname_trans(self):
        """
        Returns the trans file name
        """
        return self._fname_trans

    def subjects_dir(self):
        """
        Returns the subjects dir file name
        """
        return self._subjects_dir

    def subject(self):
        """
        Returns the subject file name
        """
        return self._subject




    
    
