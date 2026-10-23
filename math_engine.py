"""
Symbolic math engine using SymPy.

The input comes from an LLM / end user, and sympify() evaluates text as Python,
so the expression is validated against a strict whitelist BEFORE parsing.
"""
import re

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

MAX_LEN = 200
_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s=]*$")
_ALLOWED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
    "pi": sp.pi, "E": sp.E, "oo": sp.oo, "I": sp.I,
}
_GLOBALS = {"Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _validate(expr_str: str):
    if len(expr_str) > MAX_LEN:
        raise ValueError("Expression too long")
    if not _ALLOWED_CHARS.match(expr_str) or "__" in expr_str:
        raise ValueError("Unsupported characters in expression")
    for name in re.findall(r"[A-Za-z_]\w*", expr_str):
        if name not in _ALLOWED_FUNCS and len(name) > 2:
            raise ValueError(f"Unsupported name: {name}")


def simplify_expression(expr_str: str):
    """Simplifies a mathematical expression symbolically."""
    try:
        expr_str = (expr_str or "").strip()
        _validate(expr_str)
        if "=" in expr_str:                       # "lhs = rhs"  ->  Eq
            lhs, rhs = expr_str.split("=", 1)
            expr = sp.Eq(parse_expr(lhs, local_dict=_ALLOWED_FUNCS, global_dict=_GLOBALS, transformations=_TRANSFORMS),
                         parse_expr(rhs, local_dict=_ALLOWED_FUNCS, global_dict=_GLOBALS, transformations=_TRANSFORMS))
            simplified = sp.simplify(expr)
        else:
            expr = parse_expr(expr_str, local_dict=_ALLOWED_FUNCS, global_dict=_GLOBALS, transformations=_TRANSFORMS)
            simplified = sp.simplify(expr)
        return {"success": True, "original": str(expr),
                "simplified": str(simplified), "latex": sp.latex(simplified)}
    except Exception as e:
        return {"success": False, "error": str(e)}
