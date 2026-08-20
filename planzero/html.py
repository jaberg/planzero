from pydantic import BaseModel, ConfigDict
from .enums import GHG

class StrictBaseModel(BaseModel):

    model_config = ConfigDict(extra='forbid')


class HTML_element(StrictBaseModel):

    def __str__(self):
        return self.as_html()


class HTML_raw(HTML_element):
    raw: str

    def as_html(self):
        return self.raw


class HTML_P(HTML_element):
    elements:list[HTML_element]

    def as_html(self):
        body = ''.join(elem.as_html() for elem in self.elements)
        return f'<p>{body}</p>'

class HTML_UL(HTML_element):
    lis: list[HTML_element]

    def as_html(self):
        lis = ''.join(f'<li>{li.as_html()}</li>' for li in self.lis)
        return f'<ul>{lis}</ul>'

import markdown

class HTML_Markdown(HTML_element):
    content: str

    def as_html(self):
        html_output = markdown.markdown(self.content)
        return html_output


import latex2mathml.converter
class HTML_Math_Latex(HTML_element):

    latex:str
    display:str = 'inline'  # 'inline' or 'block'

    def as_html(self):
        mathml = latex2mathml.converter.convert(self.latex, display=self.display)
        return mathml


html_by_ghg = {
    GHG.CO2:HTML_Math_Latex(latex=r'\mathrm{CO}_2').as_html(),
    GHG.CH4:HTML_Math_Latex(latex=r'\mathrm{CH}_4').as_html(),
    GHG.N2O:HTML_Math_Latex(latex=r"\mathrm N_2 \mathrm O").as_html(),
    GHG.HFCs:'HFCs',
    GHG.PFCs:'PFCs',
    GHG.SF6:HTML_Math_Latex(latex=r'\mathrm{SF}_6').as_html(),
    GHG.NF3:HTML_Math_Latex(latex=r'\mathrm{NF}_3').as_html(),
}

class EChartTitle(StrictBaseModel):
    text:str
    subtext:str
    left:str = 'center'


class EChartXAxis(StrictBaseModel):
    name: str = 'Year'
    nameLocation: str = 'middle'
    nameGap: int = 30
    data: list[int]
    min:float|str|None = None
    max:float|str|None = None


class EChartYAxis(StrictBaseModel):
    name: str
    nameLocation: str = 'middle'
    nameGap: int = 40
    min:float|str|None = None
    max:float|str|None = None


class EChartLineStyle(StrictBaseModel):
    width: int|None = 2
    type: str | None = None # solid, dashed, dotted
    color: str | None = None
    lineWidth:int|None = None
    opacity:int|None = None


class EChartItemStyle(StrictBaseModel):
    color: str | None = None


class EChartSeriesDataElem(StrictBaseModel):
    value: float
    url: str | None


class EChartSeriesBase(StrictBaseModel):
    name: str|None = None
    type: str = 'line'
    yAxisIndex: int|None = None
    lineStyle: EChartLineStyle | None = EChartLineStyle(width=2)
    itemStyle: EChartItemStyle | None = None
    areaStyle: dict|None = None

    data: list[EChartSeriesDataElem]|list[float]|list[list[float]]

    xAxisId:str|None = None
    yAxisId:str|None = None
    symbol:str|None = None

    stack: str|None = None


def EChartSeriesData(sts, times, v_unit, url):
    assert times
    values = sts.query(times).to(v_unit).magnitude
    return [{'value': float(vv) if vv == vv else 0, 'url': url}
            for vv in values]


class EChartSeriesStackElem(EChartSeriesBase):
    catpath: str | None = None
    list_raw_data: list[float] | None = None
    select: dict = {'itemStyle': {'borderWidth': 20}}
    stack: str = 'Total'
    areaStyle: dict = {}
    emphasis: dict = {'focus': 'series'}
    data: list[EChartSeriesDataElem]


class StackedAreaEChart(HTML_element):
    div_id:str
    width_px:int = 800
    height_px:int = 600

    title: EChartTitle
    xAxis: EChartXAxis
    yAxis: EChartYAxis | list[EChartYAxis]

    stacked_series: list[EChartSeriesBase]
    other_series: list[EChartSeriesBase]
    legend: dict | None = None

    def save_as(self, filepath):
        # called from Makefile to create snapshots for posts
        with open(filepath, 'w') as ofile:
            ofile.write(self.as_html())

    def as_html(self):
        newline = '\n'
        return f"""
        <div id="{self.div_id}" style="width: {self.width_px}px; height: {self.height_px}px; margin: 0 auto;">
        </div>
        <script>
        var mychart_{self.div_id} = echarts.init(
            document.getElementById('{self.div_id}'),
            null,
            {{renderer: 'canvas', hoverLayerThreshold: 0}});
        var option_{self.div_id};
        option_{self.div_id} = {{
            title: {self.title.model_dump(exclude_none=True)},
            xAxis: {self.xAxis.model_dump(exclude_none=True)},
            yAxis: {[ya.model_dump(exclude_none=True) for ya in self.yAxis]
                    if isinstance(self.yAxis, list)
                    else self.yAxis.model_dump(exclude_none=True)},
            series: {
                [foo.model_dump(exclude_none=True)
                for foo in (self.stacked_series + self.other_series)]},
            tooltip: {{
                trigger: 'item',
                axisPointer: {{
                  type: 'cross',
                  label: {{
                    backgroundColor: '#6a7985'
                  }}
                }},
                formatter: params => {{
                    return params.seriesName;
                }},
            }},
        }}
        if ({1 if self.legend is not None else 0}) {{
            option_{self.div_id}["legend"] = {self.legend};
        }}
        option_{self.div_id} && mychart_{self.div_id}.setOption(option_{self.div_id});

        mychart_{self.div_id}.on('click', function(params) {{
          // Console log to see what data is available
          console.log(params);

          // params.data contains the array for that point: [x, y, url]
          // So the URL is at index 2
          var url = params.data.url;

          if (url) {{
            // Open in new tab
            //window.open(url, '_blank');

            // OR open in same tab:
            window.location.href = url;
          }}
        }});

        window.addEventListener('resize', function() {{
          mychart_{self.div_id}.resize();
        }});
        </script>
        <div>
        """


class EChartMatrixXY(StrictBaseModel):
    data:list[object]|None = None
    length:int|None = None
    levelSize:int
    label:dict[str,object]|None = None
    show:bool


class EChartMatrixCorner(StrictBaseModel):
    data: list[dict[str,object]]
    label: dict[str,object]


class EChartMatrixBodyDataElem(StrictBaseModel):
    coord: str|list[int|None]
    value: str
    label: dict[str, object]
    coordClamp: bool|None = None
    mergeCells: bool|None = None


class EChartMatrixBody(StrictBaseModel):
    data: list[EChartMatrixBodyDataElem]
    label: dict[str,object]|None = None


class EChartMatrix(StrictBaseModel):
    x:EChartMatrixXY
    y:EChartMatrixXY
    corner:EChartMatrixCorner
    body:EChartMatrixBody
    top: int|str = 0
    bottom: int|str = 0
    width: int|str = '100%'
    left: int|str = 'center'


class EChartToolTip(StrictBaseModel):
    trigger:str

class EChartDataZoomElem(StrictBaseModel):
    type:str
    xAxisIndex:str|int
    throttle:int
    top: int|str|None = None
    bottom: int|str|None = None
    width: int|str|None = None
    left: int|str|None = None
    right: int|str|None = None
    height: int|str|None = None


class EChartGrid(StrictBaseModel):
    id:str
    coordinateSystem:str
    coord:list[int, int]
    top:int|str|None = None
    bottom:int|str|None = None
    left:int|str|None = None
    width:int|str|None = None
    containLabel:bool

class EChartMatrixXAxis(StrictBaseModel):
    type:str
    id:str
    gridId:str
    scale:bool
    axisTick:dict[str, object]
    axisLabel:dict[str, object]
    axisLine:dict[str, object]
    splitLine:dict[str, object]
    min:float|str|None = None
    max:float|str|None = None
    boundaryGap:bool|None = None # from confidence-band example, not sure what it does


class EChartMatrixYAxis(StrictBaseModel):
    id:str
    gridId:str
    scale:bool
    interval:int
    axisTick:dict[str, object]
    axisLabel:dict[str, object]
    axisLine:dict[str, object]
    min:float|str|None = None
    max:float|str|None = None


class GridLinkElem(StrictBaseModel):
    gridId:str
    url:str

class UncertainSparklineMatrixEChart(HTML_element):
    # Enforce strict field checks
    model_config = ConfigDict(extra='forbid')

    div_id:str
    width:str|int = '100%'
    height:str|int = '600px'

    matrix: EChartMatrix
    tooltip: EChartToolTip
    dataZoom: list[EChartDataZoomElem]
    grid: list[EChartGrid]
    xAxis: list[EChartMatrixXAxis]
    yAxis: list[EChartMatrixYAxis]
    series: list[EChartSeriesBase]
    grid_links: list[GridLinkElem] = []

    def save_as(self, filepath):
        # called from Makefile to create snapshots for posts
        with open(filepath, 'w') as ofile:
            ofile.write(self.as_html())

    def list_helper(self, lst):
        tmp = ',\n'.join(lst_ii.model_dump_json(exclude_none=True)
                       for lst_ii in lst)
        return f'[{tmp}]'

    def as_html(self):
        newline = '\n'
        return f"""
        <div id="{self.div_id}" style="width: {self.width}; height: {self.height}; margin: 0 auto;">
        </div>
        <script>
        var mychart_{self.div_id} = echarts.init(
            document.getElementById('{self.div_id}'),
            null,
            {{renderer: 'canvas', hoverLayerThreshold: 0}}); // still need?
        var option_{self.div_id};
        option_{self.div_id} = {{
            matrix: {self.matrix.model_dump_json(indent=2, exclude_none=True)},
            tooltip: {self.tooltip.model_dump_json(indent=2, exclude_none=True)},
            dataZoom: {self.list_helper(self.dataZoom)},
            grid: {self.list_helper(self.grid)},
            xAxis: {self.list_helper(self.xAxis)},
            yAxis: {self.list_helper(self.yAxis)},
            series: {self.list_helper(self.series)},
        }}
        mychart_{self.div_id}.setOption(option_{self.div_id});

        // 2. Attach a click listener to the underlying ZRender instance
        mychart_{self.div_id}.getZr().on('click', function (params) {{
            // 1. Define your array of grid to URL mappings
            const gridLinks = {self.list_helper(self.grid_links)};

            // Extract the raw [x, y] pixel coordinates of the mouse click
            const pixelLoc = [params.offsetX, params.offsetY];

            // 3. Loop over your array to see which grid contains this pixel
            for (let ii = 0; ii < gridLinks.length; ii++) {{
                const mapping = gridLinks[ii];

                // containPixel takes an object specifying the component type/id, and the pixel array
                if (mychart_{self.div_id}.containPixel({{ gridId: mapping.gridId }}, pixelLoc)) {{

                    // Optional: Log it for debugging
                    console.log(`Clicked inside ${{mapping.gridId}}. Routing to ${{mapping.url}}`);

                    // 4. Redirect the user
                    window.location.href = mapping.url;

                    // Stop looping once we found the match
                    break;
                }}
            }}
        }});


        // Listen to all clicks on the canvas
        mychart_{self.div_id}.getZr().on('click', function (params) {{
          console.log("clickZr");
        }});

      console.log("loading");
        window.addEventListener('resize', function() {{
          mychart_{self.div_id}.resize();
          console.log("resizing");
        }});
        </script>
        <div>
        """



import inspect
GITHUB_WORKSPACE = os.environ.get('GITHUB_WORKSPACE')

def coderef_filepath(obj):
    file_path = inspect.getsourcefile(obj)
    if file_path.startswith('/mnt/planzero'):
        file_path = file_path[len('/mnt/'):]
    elif file_path.startswith('/content/planzero'):
        file_path = file_path[len('/content/'):]
    elif GITHUB_WORKSPACE and file_path.startswith(f'{GITHUB_WORKSPACE}/planzero'):
        file_path = file_path[len(f'{GITHUB_WORKSPACE}/'):]
    else:
        assert 0, file_path
    return file_path


def coderef_url(obj):
    file_path = coderef_filepath(obj)
    lines, line_number = inspect.getsourcelines(obj)
    url = f'https://github.com/jaberg/planzero/blob/main/{file_path}#L{line_number}'
    return url
