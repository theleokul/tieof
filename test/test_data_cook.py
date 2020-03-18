import os

import pytest
import numpy as np
from dineof._data_cook import DataCook

file_abspath = os.path.abspath(__file__)
basedir = os.path.dirname(file_abspath)


class TestDataCook:

    @classmethod
    def setup_class(cls):
        cls.dc = DataCook(os.path.join(basedir, 'shapes/baikal.shp'),
                          os.path.join(basedir, '2019/Input'),
                          'chlor_a')

    @pytest.fixture
    def geo_matrices(self):

        lons = np.linspace(-180, 180, 2000)
        lats = np.linspace(-90, 90, 2000)[::-1]

        yield np.meshgrid(lons, lats)

    def test_form_cut_mask_on_bounds(self, geo_matrices):
        lons, lats = geo_matrices

        lons_cut_mask = self.dc.form_cut_mask_on_bounds(lons, bounds=(103, 110))
        lons = lons[lons_cut_mask]
        assert np.all(103 < lons) and np.all(lons < 110)

        lats_cut_mask = self.dc.form_cut_mask_on_bounds(lats, bounds=(51, 56))
        lats = lats[lats_cut_mask]
        assert np.all(51 < lats) and np.all(lats < 56)
