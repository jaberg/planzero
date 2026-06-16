from pydantic import BaseModel, ConfigDict

class HTML_element(BaseModel):

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


class EChartTitle(BaseModel):
    text:str
    subtext:str
    left:str = 'center'


class EChartXAxis(BaseModel):
    name: str = 'Year'
    nameLocation: str = 'middle'
    nameGap: int = 30
    data: list[int]


class EChartYAxis(BaseModel):
    name: str
    nameLocation: str = 'middle'
    nameGap: int = 40


class EChartLineStyle(BaseModel):
    width: int = 2
    type: str | None = None
    color: str | None = None


class EChartItemStyle(BaseModel):
    color: str | None = None


class EChartSeriesDataElem(BaseModel):
    value: float
    url: str | None


class EChartSeriesBase(BaseModel):
    name: str
    type: str = 'line'
    yAxisIndex: int = 0
    lineStyle: EChartLineStyle | None = EChartLineStyle(width=2)
    itemStyle: EChartItemStyle | None = None
    data: list[float | EChartSeriesDataElem]


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


class EChartMatrixXY(BaseModel):
    #data
    length:int
    levelSize:int
    label:dict[str,object]|None = None
    show:bool


class EChartMatrixCorner(BaseModel):
    data: list[dict[str,object]]
    label: dict[str,object]


class EChartMatrixBodyDataElem(BaseModel):
    coord: str|list[int]
    value: str
    label: dict[str, object]


class EChartMatrixBody(BaseModel):
    data: list[EChartMatrixBodyDataElem]
    label: dict[str,object]|None = None


class EChartMatrix(BaseModel):
    x:EChartMatrixXY
    y:EChartMatrixXY
    corner:EChartMatrixCorner
    body:EChartMatrixBody
    top: int|str = 0
    bottom: int|str = 0
    width: int|str = '100%'
    left: int|str = 'center'


class EChartToolTip(BaseModel):
    trigger:str

class EChartDataZoomElem(BaseModel):
    type:str
    xAxisIndex:str|int
    throttle:int
    top: int|str|None = None
    bottom: int|str|None = None
    width: int|str|None = None
    left: int|str|None = None


class EChartGrid(BaseModel):
    # TODO
    pass

class EChartSeries(BaseModel):
    # TODO
    pass


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
    xAxis: list[EChartXAxis]
    yAxis: list[EChartYAxis]
    series: list[EChartSeries]

    def save_as(self, filepath):
        # called from Makefile to create snapshots for posts
        with open(filepath, 'w') as ofile:
            ofile.write(self.as_html())

    def as_html(self):
        newline = '\n'
        return f"""
        <div id="{self.div_id}" style="width: {self.width}; height: {self.height}; margin: 0 auto;">
        </div>
        <script>
        const False = false;
        var mychart_{self.div_id} = echarts.init(
            document.getElementById('{self.div_id}'),
            null,
            {{renderer: 'canvas', hoverLayerThreshold: 0}}); // still need?
        var option_{self.div_id};
        option_{self.div_id} = {{
            matrix: {self.matrix.model_dump(exclude_none=True)},
            tooltip: {self.tooltip.model_dump(exclude_none=True)},
            dataZoom: {[elem.model_dump(exclude_none=True) for elem in self.dataZoom]},
            grid: {[elem.model_dump(exclude_none=True) for elem in self.grid]},
            xAxis: {[elem.model_dump(exclude_none=True) for elem in self.xAxis]},
            yAxis: {[elem.model_dump(exclude_none=True) for elem in self.yAxis]},
            series: {[elem.model_dump(exclude_none=True) for elem in self.series]},
        }}
        mychart_{self.div_id}.setOption(option_{self.div_id});

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



import inspect

def coderef_filepath(obj):
    file_path = inspect.getsourcefile(obj)
    if file_path.startswith('/mnt/planzero'):
        file_path = file_path[len('/mnt/'):]
    elif file_path.startswith('/content/planzero'):
        file_path = file_path[len('/content/'):]
    else:
        assert 0, file_path
    return file_path

def coderef_url(obj):
    file_path = coderef_filepath(obj)
    lines, line_number = inspect.getsourcelines(obj)
    url = f'https://github.com/jaberg/planzero/blob/main/{file_path}#L{line_number}'
    return url
