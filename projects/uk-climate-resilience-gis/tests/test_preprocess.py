import tempfile
import unittest
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from resilience.preprocess import validated_join,zonal_raster


class HarmonisationTests(unittest.TestCase):
    def test_duplicate_join_cannot_inflate_population(self):
        with self.assertRaises(ValueError):
            validated_join(pd.DataFrame({'id':['a']}),pd.DataFrame({'id':['a','a'],'population':[20,20]}),'id')

    def test_missing_join_cannot_silently_drop_a_neighbourhood(self):
        with self.assertRaises(ValueError):
            validated_join(pd.DataFrame({'id':['a','b']}),pd.DataFrame({'id':['a'],'population':[20]}),'id')

    def test_partial_pixels_are_weighted_by_intersection_area(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'test.tif'
            with rasterio.open(path,'w',driver='GTiff',width=2,height=1,count=1,dtype='float32',crs=27700,transform=from_origin(0,100,100,100),nodata=-9999) as dst:
                dst.write(np.array([[1.,3.]],dtype='float32'),1)
            areas=gpd.GeoDataFrame({'lsoa21cd':['a']},geometry=[box(25,0,125,100)],crs=27700)
            result=zonal_raster(areas,path).iloc[0]
            self.assertAlmostEqual(result['mean'],1.5)
            self.assertAlmostEqual(result.coverage,1.)
