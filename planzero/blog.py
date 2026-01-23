from pydantic import BaseModel
import datetime
import sympy

_classes = []
_blogs_by_url_filename = {}

class BlogPost(BaseModel):

    date: datetime.datetime
    title: str
    about: str
    url_filename: str
    author: str

    def __init__(self, **kwargs):
        if 'about' not in kwargs:
            kwargs = dict(kwargs, about=self.__class__.__doc__)
        super().__init__(**kwargs)

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        _classes.append(cls)

from sympy.printing.mathml import mathml
from functools import partial
pres = partial(mathml, printer='presentation')

class GHG_Emissions(BlogPost):
    """
    What are greenhouse gases and what do they have to do with
    climate?  This is, I hope, the first post in a series developing
    various plans to achieve a net-zero economy in Canada. It outlines the
    terms in which net-zero is defined, and documents planzero's simple
    climate model.
    """
    equations: dict[str, str]
    a: str

    def __init__(self):
        CO2 = sympy.symbols("CO_2")
        CH4 = sympy.symbols("CH_4")
        N2O = sympy.symbols("N2O") # TODO: use latex2mathml and put the subscript in the middle
        CO2e = sympy.symbols("CO2e")
        N = sympy.symbols("N")
        N0 = sympy.symbols("N0")
        C = sympy.symbols("C")
        C0 = sympy.symbols("C0")
        W = sympy.symbols("W")
        m = sympy.symbols("m")
        t = sympy.symbols("t")
        Cf = sympy.Function("C")(t)
        years = sympy.symbols("years")
        GWP_100 = sympy.symbols("GWP_100")
        delta_C_left = pres(sympy.Derivative(C, t))

        with sympy.evaluate(False):

            equations = dict(
                CO2=pres(CO2),
                CO2e=pres(CO2e),
                CH4=pres(CH4),
                N2O=pres(N2O),
                C=pres(C),
                C0=pres(C0),
                W_m2=pres(W / m ** 2),
                CO2_df=pres(5.35 * sympy.ln(C / C0)), # W/m^2
                delta_C_left=delta_C_left,
                CH4_delta_C_right=pres(-C / (12 * years)),
                CH4_df=pres(0.036 * (sympy.sqrt(C) - sympy.sqrt(C0))), # W/m^2
                N2O_df=pres(0.12 * (sympy.sqrt(C) - sympy.sqrt(C0))), # W/m^2
                N2O_delta_C_right=pres(-C / (114 * years)),
                GWP_100=pres(GWP_100),
                HFC_delta_C_right=pres(-C / (14 * years)),
                HFC_df=pres(0.16 * C),
                PFC_df=pres(0.08 * C),
                SF6_df=pres(0.57 * C),
                NF3_df=pres(0.21 * C),
                )

        super().__init__(
            date=datetime.datetime(2026, 1, 21),
            title="A Simple Model of Greenhouse Gas Emissions",
            url_filename="2026-01-21-unfccc",
            author="James Bergstra",
            a="bar",
            equations=equations)


def init_blogs_by_url_filename():
    for cls in _classes:
        obj = cls()
        _blogs_by_url_filename[obj.url_filename] = obj
