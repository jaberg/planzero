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
