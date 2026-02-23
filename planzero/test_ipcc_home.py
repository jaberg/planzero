from . import ipcc_home

def test_smoke():
    obj = ipcc_home.stacked_area('foo')
    print(obj.as_html())
