"""
Exact Symbolic Math Engine using SymPy.
Handles differential equations, symbolic integration, limits, and exact physics formula parsing.
"""
import sympy as sp

def simplify_expression(expr_str: str):
    """Simplifies mathematical or physical expressions symbolically."""
    try:
        expr = sp.sympify(expr_str)
        simplified = sp.simplify(expr)
        return {
            "success": True,
            "original": str(expr),
            "simplified": str(simplified),
            "latex": sp.latex(simplified)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}