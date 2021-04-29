#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:04:56 2021

Package to display how a point source should appear at any location on the 
TESS detector. Two options: TESS_PRF, referencing the actual Pixel Response
Function models on MAST (or downloaded locally). Gaussian_PRF will generate 
an Gaussian PRF model. If you want to know how a star will appear in a given 
Target Pixel File (TPF)), initialize the PRF object for the center of the PRF 
(in terms of row/column), then call the `resample` function to position the 
star at a specific location within the PRF (row/column relative to TPF corner).

@author: keatonb
"""
import numpy as np
from bs4 import BeautifulSoup
import requests
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import os
from glob import glob

class _PRF:
    """Pixel Response Function base class
    
    To be inherited by classes
     - TESS_PRF
     - Gaussian_PRF
     
    resample function positions an oversampled PRF model at a certain location
    in a TPF, relative to TPF corner.
    """
    def __init__(self):
        self.prf = None
    
    def resample(self, sourcecol, sourcerow, stampsize=(11,11), 
                 supersamplefactor = 9):
        """Sample TESS PRF at location within TPF
        
        sourcex (float): col position of star (relative to TPF)
        sourcey (float): row position of star (relative to TPF)
        stampsize (int,int): (height,width) of TPF
        supersamplefactor (int): interpolate supersampled before downsampling (default 9)
        """
        #Check that PRF has been initialized
        if self.prf is None:
            raise RuntimeError("PRF not available. Instantiate as subclass" 
                               "(e.g., TESS_PRF, Gaussian_PRF)")
        
        #Center PRF on origin
        prfcol = np.arange(-self.prf.shape[1]/2.+.5,self.prf.shape[1]/2.+.5)
        prfrow = np.arange(-self.prf.shape[0]/2.+.5,self.prf.shape[0]/2.+.5)

        #Convert to relative locations in TPF
        relprfcol = prfcol/self.prfnsamp + sourcecol
        relprfrow = prfrow/self.prfnsamp + sourcerow

        #Supersample the stamp, then downsample
        supercol = (np.arange(stampsize[1]*supersamplefactor) + 0.5) / supersamplefactor
        superrow = (np.arange(stampsize[0]*supersamplefactor) + 0.5) / supersamplefactor

        #Interpolate PRF values onto supersampled stamp pixels
        interppix = RectBivariateSpline(relprfrow,relprfcol,self.prf)
        interped = interppix(superrow,supercol) #Interpolate

        #Replace values outside PRF with zeros
        interped[:,np.where(supercol < np.min(relprfcol))] = 0
        interped[:,np.where(supercol > np.max(relprfcol))] = 0
        interped[np.where(superrow < np.min(relprfrow)),:] = 0
        interped[np.where(superrow > np.max(relprfrow)),:] = 0

        #Downsample to TESS TPF resolution
        output = np.zeros([int(s/supersamplefactor) for s in interped.shape])
        for i in range(supersamplefactor):
            for j in range(supersamplefactor):
                output += interped[i::supersamplefactor,j::supersamplefactor]
        output /= supersamplefactor**2.
        return output
    
class TESS_PRF(_PRF):
    """TESS Pixel Response Function object
    
    """
    def __init__(self,cam,ccd,sector,colnum,rownum, localdatadir = None):
        """Get TESS PRF for detector location, sector
        
        Downloads relevant PRF files from the MAST archive by default
        
        ***To use pre-downloaded local files, give directory containing
        subdirectories of format "cam#_ccd#/" as localdatadir, appropriate
        for sector of interest (separate for Sectors 1-3, 4+)
        
        inputs:
         - cam (int): TESS camera number
         - ccd (int): TESS ccd number
         - sector (int): TESS sector number
         - colnum (float): column number near target
         - rownum (float): row number near target
        """
        self.cam,self.ccd,self.sector,self.colnum,self.rownum = cam,ccd,sector,colnum,rownum
        self.prfnsamp = 9 #samples/pixel for TESS PRFs
        
        #Get PRF file info
        subdir = f'cam{int(cam)}_ccd{int(ccd)}/'
        filelist = None #local and online options
        
        if localdatadir is None:
            #Different PRFs for Sectors 1-3, and 4+
            #https://heasarc.gsfc.nasa.gov/docs/tess/observing-technical.html#point-spread-function
            if sector < 4:
                url = 'https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0001/' + subdir
            else:
                url = 'https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0004/' + subdir
            ext = 'fits'

            #Get list of available PRF files
            #https://stackoverflow.com/a/34718858
            def listFD(url, ext=''):
                page = requests.get(url).text
                soup = BeautifulSoup(page, 'html.parser')
                return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

            filelist = [file for file in listFD(url, ext)]
        else:
            filelist = glob(os.path.join(localdatadir, subdir) + '*.fits')
            
        cols = np.array([int(file[-9:-5]) for file in filelist])
        rows = np.array([int(file[-17:-13]) for file in filelist])

        #Bilinear interpolation between four nearest PRFs
        dist = np.sqrt((rows-rownum)**2. + (cols-colnum)**2.)
        nearestinds = np.argsort(dist)[:4]
        #Following https://stackoverflow.com/a/8662355
        points = []
        for ind in nearestinds:
            hdulist = fits.open(filelist[ind])
            prf = hdulist[0].data
            points.append((cols[ind],rows[ind],prf))
            hdulist.close()
        points = sorted(points)
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        self.prf = (q11 * (x2 - colnum) * (y2 - rownum) +
                    q21 * (colnum - x1) * (y2 -  rownum) +
                    q12 * (x2 - colnum) * ( rownum - y1) +
                    q22 * (colnum - x1) * ( rownum - y1)
                    ) / ((x2 - x1) * (y2 - y1) + 0.0)
        
class Gaussian_PRF(_PRF):
    """Gaussian Pixel Response Function object
    
    """
    def __init__(self,sigma=1.,npixels=15,nsamp=9):
        """Generate a synthetic, Gaussian Pixel Response Function
        
        inputs:
         - sigma (float, default 1.0): sigma of the Gaussian, in TESS pixels
         - npixels (int, default 15): width of sampled PRF in TESS pixels
         - nsamp (int, default 9): number of PRF samples per TESS pixel
        """
        self.sigma = sigma
        #force integers
        self.npixels = int(npixels)
        self.prfnsamp = int(nsamp)
        
        #center coords
        x0,y0 = (self.npixels/2.,self.npixels/2.) #in pixel units
        row,col = np.meshgrid(np.arange(self.npixels*self.prfnsamp)+0.5, 
                              np.arange(self.npixels*self.prfnsamp)+0.5)
        row /= self.prfnsamp
        col /= self.prfnsamp
        #2D Gaussian
        self.prf = np.exp(-(((row-x0)/(self.sigma))**2+((col-y0)/(self.sigma))**2)/2)