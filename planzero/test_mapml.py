from .mapml import  *


_test_point_output = """<mapml-viewer projection="OSMTILE"
             zoom="4"
             lon="2.5"
             lat="1.5"
             width="400" height="300"
            controls>
            <map-layer
            src="https://maps4html.org/web-map-doc/demo/data/osm.mapml"
            label="Open Street Map"
            checked
        ></map-layer> <map-layer label="Features" checked><map-meta name="projection" content="OSMTILE"></map-meta><map-feature>
          <map-featurecaption>My Label</map-featurecaption>
          <map-geometry cs="gcrs">
            <map-point class="point">
              <map-coordinates>4.5 3.5</map-coordinates>
            </map-point>
          </map-geometry>
        </map-feature></map-layer>
         </mapml-viewer>"""


def test_point():
    viewer = MapML_Viewer(
        zoom=4,
        lat=1.5,
        lon=2.5,
        width=400,
        height=300,
        controls=True,
        layers=[
            OpenStreetMap(),
            FeatureLayer(
                features=[
                    Point(lat=3.5, lon=4.5, label="My Label"),
                ]),
        ])
    print(viewer.as_html())
    assert viewer.as_html() == _test_point_output
