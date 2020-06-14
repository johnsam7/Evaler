# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:14:30 2018

@author: ju357
"""

import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib as plt
import mne
import warnings


def add_verts(src, verts_to_add_hemi):
    """
    Add vertices in verts_to_add_hemi to src, which can be used to activate
    sources before a forward model is built.

    Parameters
    ----------
    verts_to_add_hemi : list
        A list containing two nested lists that each contains the vertices to
        add to each hemisphere.
    src : bool, optional
        List of source space objets (left and right hemispheres).

    Returns
    -------
    new_src
        New source spaces with activated sources.
    """
    new_src = src.copy()
    for hemi in range(2):
        for vert in verts_to_add_hemi[hemi]:
            new_src[hemi]['inuse'][vert] = 1
            new_src[hemi]['nuse'] = new_src[hemi]['nuse'] + 1
            new_src[hemi]['vertno'] = np.nonzero(new_src[hemi]['inuse'])[0]
    
    return new_src


def fill_empty_labels(src, labels):
    labels_divide = [[(c, label) for c, label in enumerate(labels) if label.hemi=='lh'],
                 [(c, label) for c, label in enumerate(labels) if label.hemi=='rh']]
    verts_to_add_hemi = []
    zero_labels = []
    for c, hemi in enumerate(['lh', 'rh']):
        verts_to_add = []
        labels_hemi = labels_divide[c]
        for label in labels_hemi:
            label_verts = label[1].vertices
            sources_in_label = np.array([vert for vert in label[1].vertices if vert in src[c]['vertno']])
            if len(sources_in_label) == 0:
                verts_to_add.append(label[1].vertices[0])
                zero_labels.append(label)
        verts_to_add_hemi.append(verts_to_add)
    src_new = add_verts(src, verts_to_add_hemi)
    print('Empty labels that have been filled: ', end='')
    for label in zero_labels: print(label[1].name + ', ',  end='')
    return src_new


def print_surf(fpath,rr,tris,scals=[],color=np.array([True])):
    """Prints a ply file from a triangulated surface based on rr and tris."""

    if scals == []:
        scals = np.ones(rr.shape[0])

    scal_face = np.zeros(tris.shape[0])
    for c,x in enumerate(scal_face):
        scal_face[c] = np.mean(scals[tris[c]])
    
    #Convert scalars to color map
    
    array_list = []
    if color.all():
        for c,x in enumerate(tris):
            color = [int(np.ceil(256.0*val))-1 for val in plt.cm.jet(scal_face[c])]
            data_tup = ([x[0], x[1], x[2]], color[0], color[1], color[2])        
            array_list.append(data_tup)    
    else:
        for c,x in enumerate(tris):
            data_tup = ([x[0], x[1], x[2]], color[0], color[1], color[2])        
            array_list.append(data_tup)   
    
    faces_colored = np.array(array_list, dtype=[('vertex_indices', 'int32', (3,)),
                                                    ('red', 'int32'), ('green', 'int32'), ('blue', 'int32')])
    rr_list = []
    for c,x in enumerate(rr):
        color = [int(np.ceil(256.0*val))-1 for val in plt.cm.jet(scals[c])]
        temp_tuple = (x[0], x[1], x[2], color[0], color[1], color[2])            
        rr_list.append(temp_tuple)
    for c,x in enumerate(rr):
        temp_tuple = (x[0], x[1], x[2], color[0], color[1], color[2])            
        rr_list.append(temp_tuple)
    
    vertex = np.array(rr_list,dtype=[('x', 'float64'), ('y', 'float64'), ('z', 'float64'),
                                         ('red', 'int32'), ('green', 'int32'), ('blue', 'int32')])
    
    el_vert = PlyElement.describe(vertex,'vertex')
    el_face = PlyElement.describe(faces_colored,'face')

    PlyData([el_vert, el_face], text=True).write(fpath)
    print(fpath)
    
    return


def print_ply(fpath, src, scals, vmax = True, vmin = True):
    """Takes a source object and scalar vertex values, transforms them to face values
    and saves them as a colored PLV file that can be opened with a software like Meshlab.
    src is the source space object, scals is the vertex scalar map and fpath is the path and
    the filename """
    
    if vmin == True:
        vmin = np.min(scals)
    if vmax == True:
        vmax = np.max(scals)-vmin
    #Color faces instead of vertices - associate each face with scalar average of surrounding verts.
    scals = (scals-vmin)/vmax
    
    print_surf(fpath,src['rr'],src['tris'],scals=scals)
        
    return
    

def print_sphere(fpath,centres,radii,color='white'):
    """prints a ply file of a sphere with centre in centres [(x,y,z),...] and radius in radii [r0,r1,...] (mm) with color white, red, green or blue [rgb]"""
    if color == 'white':
        color = np.array([255,255,255])
    if color == 'red':
        color = np.array([255,0,0])
    if color == 'green':
        color = np.array([0,255,0])
    if color == 'blue':
        color = np.array([0,0,255])
        
    rr = np.array([]).reshape(0,3)
    tris = np.array([]).reshape(0,3)
    for k in range(len(radii)):
        radius = radii[k]
        centre = centres[k]
        sphere = PlyData.read('./ply_files/sphere.ply')
        verts = np.array([list(dat)[0:3] for dat in sphere['vertex']._data])
        verts = verts/np.max(abs(verts))
        verts = verts*radius
        verts = verts + centre
        verts = verts/1000.0
        faces = np.array([list(dat)[0:3] for dat in sphere['face']._data])
        faces = faces[:,0,:] + int(k*sphere['vertex'][:].shape[0])
        rr = np.vstack((verts,rr))
        tris = np.vstack((tris,faces))
        
    print_surf(fpath,rr,tris.astype(int),color=color)

    return


def calculate_area(rr,tris):
    """Takes rr - an array of position of vertices and tris - indices of vertices that deliniates
    triangle face and returns the total area of the tesselation surface."""
    
    area = 0.0
    for row in tris:
        v1=rr[row[1],:]-rr[row[0],:]
        v2=rr[row[2],:]-rr[row[0],:]
        nml = np.cross(v1,v2)
        area = area + np.linalg.norm(nml)/2.0            

    print('Total surface area: ' + np.str(area))
    
    return area

def solid_angles(obs_points, rr, tris, n_jobs=1):
    from joblib import Parallel, delayed
    
    def get_solid_angs(obs_points, rr, tris):
        solid_angles = []
        for c, obs_point in enumerate(obs_points):
            solid_angles.append(calculate_solid_angle(rr, tris, obs_point))
            print(str(c/len(obs_points)*100)+' % complete         \r',end='')
        return solid_angles

    myfunc = delayed(get_solid_angs)
    parallel = Parallel(n_jobs=n_jobs)
    obs_points_chunks = np.array_split(obs_points, n_jobs)
    out = parallel(myfunc(obs_points_chunk, rr, tris) for obs_points_chunk in obs_points_chunks)

    a = []
    solid_fi = np.array([a+fi for fi in out]).flatten()
    
    insideout = np.zeros(solid_fi.shape)
    for c, angle in enumerate(solid_fi):
        if np.abs(angle-4*np.pi) < 0.01:
            insideout[c] = 1
        elif np.abs(angle) < 0.01:
            insideout[c] = 0
        else:
            insideout[c] = 0.5

    return solid_fi, insideout

def calculate_solid_angle(rr, tris, obs_point):
    RR = (rr[tris]-obs_point)
    norms = np.linalg.norm(RR[:,:,:],axis=2)
    solid_angle = np.sum(2*np.arctan(np.sum(np.multiply(RR[:,0,:],np.cross(RR[:,1,:],RR[:,2,:])),axis=1)/ \
            (norms[:,0]*norms[:,1]*norms[:,2] + np.sum(np.multiply(RR[:,0,:],RR[:,1,:]),axis=1)*norms[:,2] + \
            np.sum(np.multiply(RR[:,0,:], RR[:,2,:]),axis=1)*norms[:,1] + np.sum(np.multiply(RR[:,1,:], RR[:,2,:]),axis=1)*norms[:,0])))
    return solid_angle
    
def calculate_normals(rr, tris, solid_angle_calc=False, obs_point=np.zeros(3), print_info=True):
    """Takes rr - an array of position of vertices and tris - indices of vertices that deliniates
    triangle face and returns vertex normals based on an (unweighted) average of neighboring face normals."""
    A = []
    area_list = []
    area = 0.0
    count=0
    nan_vertices = []
    for x in range(len(rr)):
        A.append([])
    solid_angle = 0
    for row in tris:
        v1=rr[row[1],:]-rr[row[0],:]
        v2=rr[row[2],:]-rr[row[0],:]
        nml = np.cross(v1,v2)
        area = area + np.linalg.norm(nml)/2.0
        area_list.append(area)
        nn_fc = nml/np.linalg.norm(nml)
        A[row[0]].append(nn_fc)
        A[row[1]].append(nn_fc)
        A[row[2]].append(nn_fc)
        if solid_angle_calc==True:
            R1 = rr[row[0]] - obs_point
            R2 = rr[row[1]] - obs_point
            R3 = rr[row[2]] - obs_point
            
            solid_angle = solid_angle + 2*np.arctan(np.dot(R1,np.cross(R2,R3))/ \
                (np.linalg.norm(R1)*np.linalg.norm(R2)*np.linalg.norm(R3) + np.dot(R1,R2)*np.linalg.norm(R3) + \
                np.dot(R1,R3)*np.linalg.norm(R2) + np.dot(R2,R3)*np.linalg.norm(R1)))  
    
    if solid_angle_calc==True:
        print('solid_angle at the point of observation estimated to:')
        print(solid_angle)    
        
    nn = np.zeros((len(rr),3))
    for c, ele in enumerate(A):
        vert_norm = np.zeros(3)
        for vec in ele:
            vert_norm = vert_norm + vec
        vert_norm = vert_norm/np.linalg.norm(vert_norm)
        nn[c,:]=vert_norm
        
    for c, ele in enumerate(A):
        if np.isnan(nn[c,:]).any(): #np.linalg.norm(nn[c,:]) == 0:
            neighbor_rows = np.where(tris==c)[0]
            neighbors = np.unique(tris[neighbor_rows])
            neighbors = neighbors[np.where(neighbors!=c)]
            normal = np.mean(nn[neighbors,:],axis=0)
            nn[c,:] = normal/np.linalg.norm(normal)
            count = count+1
            
            if np.isnan(nn[c,:]).any():
                nan_vertices.append(c)
            
    if print_info:
        print('number of nan normals that have been smoothed = ' + str(count))
        print('Remaining NAN normals = ' + str(len(nan_vertices)))                
        print('Total surface area: ' + np.str(area))
    
    return (nn,area,area_list,nan_vertices)


def add_ply_tesselation_to_source_space(src_orig, ply_filepath):
    """ Takes a PLY file, transforms it into a source space object and adds it to an existing source space (src_orig).
    Input:  ply_filepath = path (string) to ply file
    Output: src = source space with ply surface in the first and only element"""
    
    plydata_marty = PlyData.read(ply_filepath)
    rr = np.array([list(dat)[:3] for dat in plydata_marty['vertex']._data])/1000
    fc = np.array([dat for dat in plydata_marty['face'].data['vertex_indices']])
    src = src_orig.copy()[1]
    src['inuse'] = np.ones(len(rr)).astype(int)
    src['nn'] = np.concatenate((np.zeros((len(rr),2)),np.ones((len(rr),1))),axis=1)
    src['np'] = len(rr)
    src['ntri'] = len(fc)
    src['nuse'] = len(rr)
    src['nuse_tri'] = len(fc)
    src['rr'] = rr
    src['tris'] = fc
    src['use_tris'] = fc
    src['vertno'] = np.nonzero(src['inuse'])[0]
    src_out = src_orig.copy()
    src_out[0] = join_source_spaces(src_orig)
    src_out[1] = src
    
    return src_out

    
def join_source_spaces(src_orig):
    if len(src_orig)!=2:
        raise ValueError('Input must be two source spaces')
        
    src_joined=src_orig.copy()
    src_joined=src_joined[0]
    src_joined['inuse'] = np.concatenate((src_orig[0]['inuse'],src_orig[1]['inuse']))
    src_joined['nn'] = np.concatenate((src_orig[0]['nn'],src_orig[1]['nn']),axis=0)
    src_joined['np'] = src_orig[0]['np'] + src_orig[1]['np']
    src_joined['ntri'] = src_orig[0]['ntri'] + src_orig[1]['ntri']
    src_joined['nuse'] = src_orig[0]['nuse'] + src_orig[1]['nuse']
    src_joined['nuse_tri'] = src_orig[0]['nuse_tri'] + src_orig[1]['nuse_tri']
    src_joined['rr'] = np.concatenate((src_orig[0]['rr'],src_orig[1]['rr']),axis=0)
    triangles_0 = len(src_orig[0]['rr'])
    src_joined['tris'] = np.concatenate((src_orig[0]['tris'],src_orig[1]['tris']+triangles_0),axis=0)
    src_joined['use_tris'] = np.concatenate((src_orig[0]['use_tris'],src_orig[1]['use_tris']+triangles_0),axis=0)
    src_joined['vertno'] = np.nonzero(src_joined['inuse'])[0]

    return src_joined    
    
        
def neighbor_dictionary(src):
    """
    Takes as source space as input and returns a dictionary with every vertex as
    keys and its neighbors as values
    """
    
    keys = range(0,src['np'])
    values = [ [] for i in range(src['np']) ]
    my_dic = dict(zip(keys,values))
    
    for triangles in src['tris']:
        for vert in triangles:
            for neighbor in triangles:
                my_dic[vert].append(neighbor)
    
    for key,val in my_dic.items():
        val = list(set(val))
        my_dic[key] = val
        
    return my_dic


def blurring(x, src, smoothing_steps=1, spread = True):
    """
    Takes a vector source estimate (x) and source object as input and outputs a
    blurred estimate blurred_x. If spread = True, it will keep 
    smoothing until all nan elements of the cortex has been smoothed.
    """
    if len(x) != len(src['rr']):
        y = np.zeros(src['np'])
        y[:] = np.nan
        y[src['vertno']] = x
        x = y

    
    neighbor_dic = neighbor_dictionary(src)
    blurred_x = x.copy()
    verts = np.array(range(0,src['np']))
    c=0
    
    if spread:
        print('computing smoothing steps...')
        while(np.sum(np.isnan(blurred_x)) > 0):
            for vert_ind in verts[np.where(np.isnan(blurred_x))[0]]:
                neighbors = neighbor_dic[vert_ind]
                blurred_x[vert_ind] = np.nanmean(blurred_x[neighbors])
            c = c+1
        
    else:
        print('computing smoothing steps...')
        for c in range(0,smoothing_steps):
            print(c)
            temp = blurred_x.copy()
            for vert_ind,dipole in enumerate(x):
                neighbors = neighbor_dic[vert_ind]
                temp[vert_ind] = np.nanmean(blurred_x[neighbors])
            blurred_x = temp.copy()
            c = c+1
                
    return blurred_x


def blurred_sensitivity_map(fwd, sensor_index, src_space_index, fpath, smoothing_steps, vmax, vmin, spread = False):
    """
    Returns the blurred source estimation and prints a sensitivity map to fpath.
    """
    
    if src_space_index == 'both':
        src = join_source_spaces(fwd['src'])
        sensitivities = np.linalg.norm(fwd['sol']['data'][sensor_index,:],axis=0)

    else:
        src = fwd['src'][src_space_index]    
        
        if src_space_index == 1:
            sensitivities = np.linalg.norm(fwd['sol']['data'][sensor_index, \
                                           fwd['src'][0]['nuse']:fwd['src'][0]['nuse']+ \
                                            fwd['src'][1]['nuse']],axis=0)
        else:
            sensitivities = np.linalg.norm(fwd['sol']['data'][sensor_index, \
                                           0:fwd['src'][0]['nuse']],axis=0)
        
    scalars = np.empty((src['np']))
    scalars[:] = np.nan
    scalars[src['inuse']==1] = sensitivities
    
    blurred_x = blurring(x=scalars, src=src, smoothing_steps=smoothing_steps, spread = spread)
    
    print('vmax = '+str(vmax))
    print_ply(fname=fpath,src=src,scals=blurred_x,vmax = vmax, vmin=vmin)
    
    print('finished printing sensitivity map to /ply_files/'+fpath)
    
    return blurred_x    

    
def read_all_labels(settings, label_path, prefix='', threshold=False):
    import glob
    from itertools import compress
    
    if threshold:
        label_names = glob.glob(label_path+'*thresh.label')
    else:
        label_names = glob.glob(label_path+'*.label')
        thresh_names = glob.glob(label_path+'*thresh.label')
        label_names = list(compress(label_names,[not name in thresh_names for name in label_names]))
        
    all_labels = [mne.read_label(filename=label_name, subject=settings.subject()) for label_name in label_names] 
    #remove prefix
    for label in all_labels:
        if prefix in label.name:
            label.name=label.name[len(prefix):len(label.name)]
    
    return all_labels

def remove_cortical_labels(all_labels):
    label_names = [label.name for label in all_labels]
    try:
        cort_ind = np.array([label_names.index('rh.cortex-rh'), label_names.index('lh.cortex-lh')])
        inds = np.where(~np.isin(range(len(all_labels)),cort_ind))[0]
        labels = [all_labels[i] for i in inds]
    except:
        labels = all_labels
    return labels
    
def remove_overlap_in_labels(settings, labels, vertlim=20):
    from collections import Counter
    labels_org = [label.copy() for label in labels]
    verts_tot_lh = np.array([]).reshape(1,0) 
    verts_tot_rh = np.array([]).reshape(1,0) 
    for label in labels:
        verts=label.vertices 
        if label.hemi == 'lh': 
            verts_tot_lh=np.hstack((verts_tot_lh,verts.reshape(1,len(verts)))) 
        if label.hemi == 'rh': 
            verts_tot_rh=np.hstack((verts_tot_rh,verts.reshape(1,len(verts)))) 
            
    verts_tot_rh = verts_tot_rh.flatten()
    verts_tot_lh = verts_tot_lh.flatten()
    overlapping_vertices = len(verts_tot_rh) - len(np.unique(verts_tot_rh)) + \
                                len(verts_tot_lh) - len(np.unique(verts_tot_lh))
    print(str(overlapping_vertices) + ' overlapping vertices found, removing...')
    overlapping_verts_rh = [item for item, count in Counter(verts_tot_rh).items() if count > 1]
    overlapping_verts_lh = [item for item, count in Counter(verts_tot_lh).items() if count > 1]
    
    #remove overlapping verts from all labels
    disjoint_labels = []
    for label in labels:
        if label.hemi == 'lh':
            label.vertices = label.vertices[~np.isin(label.vertices,overlapping_verts_lh)]
        if label.hemi == 'rh':
            label.vertices = label.vertices[~np.isin(label.vertices,overlapping_verts_rh)]
        disjoint_labels.append(label)
    
    #add overlapping verts to ONE label to avoid lack of label coverage over cortex
    for vert in overlapping_verts_lh:
        for c,label in enumerate(labels_org):
            if label.hemi == 'lh':
                if vert in label.vertices:
                    disjoint_labels[c].vertices = np.insert(disjoint_labels[c].vertices,0,vert)
                    break
    for vert in overlapping_verts_rh:
        for c,label in enumerate(labels_org):
            if label.hemi == 'rh':
                if vert in label.vertices:
                    disjoint_labels[c].vertices = np.insert(disjoint_labels[c].vertices,0,vert)
                    break
                
    #check that vertices are no longer overlapping and cover entirety of cortex
    a = np.array([]).reshape(1,0)
    fwd = mne.read_forward_solution(settings.fname_fwd())
    offset = fwd['src'][0]['np']
    for label in disjoint_labels:
        if label.hemi == 'lh':
            a = np.concatenate((a,label.vertices.reshape(1,len(label.vertices))),axis=1)
        if label.hemi == 'rh':
            a = np.concatenate((a,label.vertices.reshape(1,len(label.vertices))+offset),axis=1)
    if a.shape[1] == np.unique(a).shape[0]:
        print('Labels are now non-overlapping.')
        if a.shape[1] == fwd['src'][0]['np']+fwd['src'][1]['np']:
            print('Labels cover entirety of cortex.')
        else:
            warnings.warn('Labels are not covering entirety of cortex.')
    else:
        warnings.warn('Labels are overlapping. This could potentially cause problems when plotting resolution metrics over brain surface.')
        
    #remove labels with less than vertlim
    vert_count = 0
    for label in disjoint_labels:
        if len(label.vertices) < vertlim:
            disjoint_labels.remove(label)
            vert_count = vert_count + len(label.vertices)

    print(str(vert_count)+' vertices have been removed due to small label sizes.')
            
    return disjoint_labels            
