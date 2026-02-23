from . import ipcc_canada
from .html import (
    EChartTitle,
    EChartXAxis,
    EChartYAxis,
    EChartSeriesStackElem,
    EChartSeriesBase,
    EChartLineStyle,
    EChartItemStyle,
    StackedAreaEChart)

def stacked_area(div_id='chart2'):
    return StackedAreaEChart(
        div_id=div_id,
        title=EChartTitle(
            text='Canadian Emissions by IPCC Sector',
            subtext='Click on data points to explore sectors, clicking regions would be nice but does not work yet'),
        xAxis=EChartXAxis(data=ipcc_canada.echart_years()),
        yAxis=EChartYAxis(name='Emissions (Mt CO2e)'),
        stacked_series=[
            EChartSeriesStackElem(**series)
            for series in ipcc_canada.echart_series_all_Mt()],
        other_series=[
            EChartSeriesBase(
                name='Target',
                lineStyle=EChartLineStyle(type='dotted', color='#606060'),
                itemStyle=EChartItemStyle(color='#606060'),
                data=ipcc_canada.CNZEAA_targets()),
            EChartSeriesBase(
                name='Net Total (without LULUCF)',
                lineStyle=EChartLineStyle(color='#303030'),
                itemStyle=EChartItemStyle(color='#303030'),
                data=ipcc_canada.net_emissions_total_without_LULUCF()),
            EChartSeriesBase(
                name='Net Total (with LULUCF)',
                lineStyle=EChartLineStyle(color='#303030'),
                itemStyle=EChartItemStyle(color='#303030'),
                data=ipcc_canada.net_emissions_total()),
        ])
