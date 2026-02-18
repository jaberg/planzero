from . import est_nir

if __name__ == '__main__':
    est = est_nir.EstSectorEmissions()
    max_gap = est.max_gap_2005()
