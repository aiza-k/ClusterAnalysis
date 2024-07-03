# coding=utf-8

import numpy as np
import pandas as pd
from osgeo import gdal

rasterfn = r"D:\AGB_project\aiza\esa.cci_N40W110_2010.2017-2020.tif"
outf = r"D:\AGB_project\aiza\average.csv"
raster = gdal.Open(rasterfn) # load tiff file

RasterCount = raster.RasterCount # get band number

year_list = [] # init year list
value_list = [] # init value list
for i in range(1,RasterCount+1): # read each band
    band = raster.GetRasterBand(i) # get band
    band_name = band.GetDescription() # get band name

    array = band.ReadAsArray() # read as array
    array = np.array(array,dtype=float) # convert array to float
    array[array<=0] = np.nan # for AGB, set 0 to nan
    array_mean = np.nanmean(array) # calculate mean, ignore nan

    year_list.append(band_name) # append year
    value_list.append(array_mean) # append value
df = pd.DataFrame({'year':year_list,'value':value_list}) # create dataframe
df.to_csv(outf,index=False) # save dataframe to csv


