# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:49:21 2021

@author: jamesnj1
"""

import itertools
from operator import itemgetter
import pandas as pd
import numpy as np


#assuming that the lat/long are in degrees, not radians
def column_haversine(geo):
    geo         = np.deg2rad(geo)
    lat         = geo[:,0]
    lng         = geo[:,1]
    diff_lat    = np.diff(lat)
    diff_lng    = np.diff(lng)
    
    d = np.sin(diff_lat/2)**2 + np.cos(lat[1:])*np.cos(lat[:-1]) * np.sin(diff_lng/2)**2
    d = 2 * 6378137 * np.arcsin(np.sqrt(d)) #distance in meters
    return d



#first ensure that the date-time field, "dtg", is a pandas datetime:
#   df['dtg'] = pd.to_datetime(df['dtg'])
def id_geotemporal_associations(df, geo_difference_limit, time_difference_limit):
    '''
    Parameters
    ----------
    df : pandas dataframe with at least 4 columns: 
        ID (id), date-time (dtg), latitude (y), longitude (x)
    geo_difference_limit : integer
        DESCRIPTION: distance in meters that one considers to be an association
    time_difference_limit : integer
        DESCRIPTION: time in  minutes that one considers to be an association

    Returns
    -------
    associationMatrix : pandas dataframe
        DESCRIPTION: array with all entities as rows and columns with number of associations
                        as the [row, column] entry
    associationList : pandas dataframe
        DESCRIPTION: list with entities in column 1 and column 2, and the number of association
                        between these entities as column 3

    '''
    tmp = df[['id','dtg','x','y']].copy().sort_values(by='dtg')
    tmp = tmp.loc[~pd.isna(tmp['id'])].reset_index(drop=True)
    
    #where do IDs change
    different_seq_ids = tmp['id'].ne(tmp['id'].shift().bfill()).astype(bool).tolist()
    
    #where is the time difference less than or equal to the time_difference_limit
    time_difference_idx = (
        tmp['dtg'].diff() <= pd.Timedelta(time_difference_limit, unit='minutes')
        ).astype(bool).tolist()
    
    #where is the geo distance less than or equal to the 
    geo_difference_idx  = (
        column_haversine(np.array(tmp[['x','y']].values.tolist())) <= geo_difference_limit
        ).astype(bool).tolist()
    geo_difference_idx = np.insert(geo_difference_idx, 0, False)
    
    #find indices where all three are true
    idxs = (
        np.asarray(different_seq_ids) * np.asarray(time_difference_idx) * np.asarray(geo_difference_idx)
        )
    idxs[np.where(idxs==True)[0]-1] = True # take indices just previous as well
    
    tmp = tmp.loc[idxs]
    
    sequentialIndices = list()
    for k,g in itertools.groupby(enumerate(tmp.index), lambda x: x[1] - x[0]):
        sequentialIndices.append(list(map(itemgetter(1),g)))
    
    #pre-allocate array for a connectivity matrix
    associationMatrix = pd.DataFrame(
        data=np.zeros((len(tmp.id.unique()),len(tmp.id.unique()))),
        index=tmp.id.unique(),
        columns = tmp.id.unique(),
        dtype=int
        )
    
    associationList = dict([('count',[]), ("entities", [])])
    count = 0
    for entry in sequentialIndices:
        entities = tmp['id'].loc[entry]
        
        for entity in entities.unique():
            associationMatrix.loc[entity,entity] += len(entities.loc[entities==entity])
            not_entity = entities.loc[entities != entity].unique()
            not_entities = {n: len(entities.loc[entities==n]) for n in not_entity}
            
            for k,v in not_entities.items():
                if (set([entity,k]) not in associationList['entities']):
                    associationList['entities'].append(set([entity,k]))
                    associationList['count'].append(0)
                associationMatrix.loc[entity,k] += v
                idx = np.where([set([entity,k]) == ii for ii in associationList['entities']])[0][0]
                associationList['count'][idx] += v
                
        count += 1
        
        if (count%200) == 0:
            print("Completed " + str(count) + "/" + str(len(sequentialIndices)) + " groupings")
            
    associationList['entities']  = [tuple(i) for i in associationList['entities']]
    associationList = pd.DataFrame(associationList)
    associationList = associationList.assign(
        entity1 = [i[0] for i in associationList['entities'].tolist()],
        entity2 = [i[1] for i in associationList['entities'].tolist()]
        )
    associationList = associationList[['entity1', 'entity2', 'count']]
    
    return associationMatrix, associationList