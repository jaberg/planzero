from pydantic import BaseModel

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


class EChartSeriesBase(BaseModel):
    name: str
    type: str = 'line'
    lineStyle: EChartLineStyle | None = EChartLineStyle(width=2)
    itemStyle: EChartItemStyle | None = None
    data: list[float]


class EChartSeriesStackDataElem(BaseModel):
    value: float
    url: str


class EChartSeriesStackElem(EChartSeriesBase):
    list_raw_data: list[float]
    select: dict = {'itemStyle': {'borderWidth': 20}}
    stack: str = 'Total'
    areaStyle: dict = {}
    emphasis: dict = {'focus': 'series'}
    data: list[EChartSeriesStackDataElem]


class StackedAreaEChart(HTML_element):
    div_id:str
    width_px:int = 800
    height_px:int = 600

    title: EChartTitle
    xAxis: EChartXAxis
    yAxis: EChartYAxis

    stacked_series: list[EChartSeriesBase]
    other_series: list[EChartSeriesBase]

    def as_html(self):
        newline = '\n'
        return f"""
        <div id="{self.div_id}" style="width: {self.width_px}px; height: {self.height_px}px; margin: 0 auto;">
        </div>
        <script>
        var mychart_{self.div_id} = echarts.init(document.getElementById('{self.div_id}'));
        var option_{self.div_id};
        option_{self.div_id} = {{
            title: {self.title.model_dump(exclude_none=True)},
            xAxis: {self.xAxis.model_dump(exclude_none=True)},
            yAxis: {self.yAxis.model_dump(exclude_none=True)},
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
