from pydantic import BaseModel

class HTML_element(BaseModel):
    def as_html(self):
        return str(self) # default


def _as_html(thing):
    if isinstance(thing, str):
        return thing
    else:
        return thing.as_html()


class MapML_Layer(BaseModel):
    label: str
    checked: bool = True


class OpenStreetMap(MapML_Layer):
    label: str = "Open Street Map"
    def as_html(self):
        return f'''<map-layer
            src="https://maps4html.org/web-map-doc/demo/data/osm.mapml"
            label="{self.label}"
            {"checked" if self.checked else ""}
        ></map-layer>'''


class MapML_Feature(BaseModel):
    label: str | HTML_element


class Point(MapML_Feature):
    map_geometry: str = "gcrs"
    map_point_class: str = "point"
    lat: float
    lon: float
    map_properties: list[str | HTML_element] = []

    def as_html(self):
        map_properties = self.map_properties_as_html()
        return f'''<map-feature>
          <map-featurecaption>{_as_html(self.label)}</map-featurecaption>
          <map-geometry cs="{self.map_geometry}">
            <map-point class="{self.map_point_class}">
              <map-coordinates>{self.lon} {self.lat}</map-coordinates>
            </map-point>{map_properties}
          </map-geometry>
        </map-feature>'''

    def map_properties_as_html(self):
        return ''.join(
            f'<map-properties>{_as_html(map_property)}</map-properties>'
            for map_property in self.map_properties)


class FeatureLayer(MapML_Layer):
    label: str = "Features"

    features: list[MapML_Feature]

    def as_html(self):
        rval = []
        rval.append(
            f' <map-layer label="{self.label}"'
            f' {"checked" if self.checked else ""}>')
        rval.append('<map-meta name="projection" content="OSMTILE"></map-meta>')
        rval.extend(feature.as_html() for feature in self.features)
        rval.append('</map-layer>')
        return ''.join(rval)


class MapML_Viewer(BaseModel):
    zoom: int
    lat: float
    lon: float
    width: int | str
    height: int | str
    controls: bool
    layers: list[MapML_Layer]

    def as_html(self):
        layers = ''.join(layer.as_html() for layer in self.layers) 
        return f'''<mapml-viewer projection="OSMTILE"
             zoom="{self.zoom}"
             lon="{self.lon}"
             lat="{self.lat}"
             width="{self.width}" height="{self.height}"
            {"controls" if self.controls else ""}>
            {layers}
         </mapml-viewer>'''
