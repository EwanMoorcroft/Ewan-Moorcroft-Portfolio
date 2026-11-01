import unittest
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from resilience.network import access_distances, walk_allowed


class WalkingTests(unittest.TestCase):
    def fixtures(self):
        graph=nx.DiGraph()
        for u,v,d in [(1,2,100),(2,3,100),(3,4,100)]:
            graph.add_edge(u,v,weight=d);graph.add_edge(v,u,weight=d)
        edges=gpd.GeoDataFrame({'u':[1,2,3],'v':[2,3,4],'length':[100.,100.,100.], 'forward':[True]*3,'backward':[True]*3},
            geometry=[LineString([(0,0),(100,0)]),LineString([(100,0),(100,100)]),LineString([(100,100),(0,100)])],crs=27700)
        return graph,edges

    def test_detour_and_connectors_are_counted(self):
        graph,edges=self.fixtures()
        origins=gpd.GeoDataFrame({'origin_id':['a']},geometry=[Point(0,-10)],crs=27700)
        entrances=gpd.GeoDataFrame(geometry=[Point(0,110)],crs=27700)
        result=access_distances(graph,edges,origins,entrances,150,100)
        self.assertAlmostEqual(result.iloc[0].distance_m,320)
        self.assertTrue(result.iloc[0].known)

    def test_same_edge_shortcut_is_exact(self):
        graph,edges=self.fixtures()
        origins=gpd.GeoDataFrame({'origin_id':['a']},geometry=[Point(20,0)],crs=27700)
        entrances=gpd.GeoDataFrame(geometry=[Point(80,0)],crs=27700)
        self.assertAlmostEqual(access_distances(graph,edges,origins,entrances,150,100).iloc[0].distance_m,60)

    def test_far_origin_is_unknown_not_inaccessible(self):
        graph,edges=self.fixtures()
        origins=gpd.GeoDataFrame({'origin_id':['a']},geometry=[Point(-1000,0)],crs=27700)
        entrances=gpd.GeoDataFrame(geometry=[Point(80,0)],crs=27700)
        result=access_distances(graph,edges,origins,entrances,150,100)
        self.assertFalse(result.iloc[0].known)

    def test_private_access_and_carriageways_are_excluded(self):
        self.assertFalse(walk_allowed({'highway':'motorway'}))
        self.assertFalse(walk_allowed({'highway':'footway','access':'private'}))
        self.assertFalse(walk_allowed({'highway':'cycleway'}))
        self.assertTrue(walk_allowed({'highway':'cycleway','foot':'yes'}))

