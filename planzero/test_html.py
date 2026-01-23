from .html import HTML_Math_Latex

def test_math_latex():

    elem = HTML_Math_Latex(latex=r"x = {-b \pm \sqrt{b^2-4ac} \over 2a}")
    mathml = elem.as_html()
    assert mathml == r"""<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline"><mrow><mi>x</mi><mo>&#x0003D;</mo><mrow><mfrac><mrow><mo>&#x02212;</mo><mi>b</mi><mi>&#x000B1;</mi><msqrt><mrow><msup><mi>b</mi><mn>2</mn></msup><mo>&#x02212;</mo><mn>4</mn><mi>a</mi><mi>c</mi></mrow></msqrt></mrow><mrow><mn>2</mn><mi>a</mi></mrow></mfrac></mrow></mrow></math>"""
