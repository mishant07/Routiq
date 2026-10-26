"""Safe calculator tool for mathematical evaluations."""
import ast
import math
import operator

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": math.factorial,
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable or function: {node.id}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            operand = _eval_node(node.operand)
            return _SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.Call):
        func = _eval_node(node.func)
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    else:
        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")


def calculate_expression(expression: str) -> str:
    """Evaluate a mathematical expression safely (e.g. 'sqrt(144) + 5 * 2', '2**10', 'sin(pi/2)').

    Args:
        expression: The mathematical expression string to evaluate.

    Returns:
        The evaluated numerical result or an error message.
    """
    clean_expr = expression.strip()
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_node(parsed.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error for '{expression}': {str(e)}"
