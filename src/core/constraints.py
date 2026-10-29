"""Constraint parsing and evaluation helpers.

The web UI accepts feature equations as text, so this module deliberately uses
an allow-listed AST evaluator rather than Python ``eval``. Objective bounds are
validated here as well so every API path applies identical semantics.
"""

import ast
import csv
import math
import operator
import re

import numpy as np


CONSTRAINT_TOLERANCE = 1e-7
MAX_EXPRESSION_LENGTH = 500
MAX_AST_NODES = 96

_SAFE_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_SAFE_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_CMPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def get_categorical_choices(feature):
    """Return categorical levels without treating commas inside a level as separators."""
    name = feature.get('name', 'Unnamed feature')
    supplied = feature.get('choices')

    if supplied is not None:
        if not isinstance(supplied, (list, tuple)):
            raise ValueError(f"Feature '{name}' categorical choices must be a list.")
        raw_choices = [str(value) for value in supplied if value is not None and str(value) != '']
    else:
        range_value = str(feature.get('range', '') or '')
        try:
            raw_choices = next(csv.reader([range_value], skipinitialspace=True), [])
        except csv.Error as exc:
            raise ValueError(f"Feature '{name}' has invalid categorical values.") from exc
        raw_choices = [value.strip() for value in raw_choices if value.strip()]

    choices = []
    seen = set()
    for value in raw_choices:
        if value not in seen:
            seen.add(value)
            choices.append(value)
    if not choices:
        raise ValueError(f"Feature '{name}' requires one or more categories.")
    return choices


def encode_categorical_series(series, feature):
    """Encode a categorical Series and reject levels missing from the declared choices."""
    choices = get_categorical_choices(feature)
    mapping = {choice: idx for idx, choice in enumerate(choices)}
    values = series.astype(str)
    encoded = values.map(mapping)
    if encoded.isna().any():
        unknown = list(dict.fromkeys(values[encoded.isna()].tolist()))
        preview = ', '.join(repr(value) for value in unknown[:5])
        if len(unknown) > 5:
            preview += ', ...'
        raise ValueError(
            f"Feature '{feature.get('name', 'Unnamed feature')}' contains categorical "
            f"value(s) missing from its choices: {preview}."
        )
    return encoded.to_numpy(), choices


def encode_categorical_training_series(series, feature):
    """Encode BO training data while preserving levels outside the current search domain.

    Configured choices are assigned the first codes so candidate generation can remain
    restricted to ``0..len(configured_choices)-1``. Historical levels that were removed
    from the feature configuration receive additional, distinct codes and can therefore
    still contribute to model fitting without being proposed as future experiments.
    """
    configured_choices = get_categorical_choices(feature)
    values = series.astype(str)
    observed_choices = list(dict.fromkeys(values.tolist()))
    model_choices = configured_choices + [
        value for value in observed_choices if value not in configured_choices
    ]
    mapping = {choice: idx for idx, choice in enumerate(model_choices)}
    return values.map(mapping).to_numpy(), model_choices


def _parse_expression(expression):
    if not isinstance(expression, str):
        raise ValueError("Constraint expression must be text.")
    expression = expression.strip()
    if not expression:
        raise ValueError("Constraint expression cannot be empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Constraint expression exceeds {MAX_EXPRESSION_LENGTH} characters."
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid constraint syntax: {exc.msg}.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Constraint expression is too complex.")
    return tree


def safe_eval_ast(node, context):
    """Evaluate arithmetic/comparisons using a strict AST allow-list."""
    if isinstance(node, ast.Expression):
        return safe_eval_ast(node.body, context)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only finite numeric constants are permitted.")
        value = float(node.value)
        if not math.isfinite(value):
            raise ValueError("Constraint constants must be finite.")
        return value
    if isinstance(node, ast.Num):  # pragma: no cover - Python < 3.8 compatibility
        value = float(node.n)
        if not math.isfinite(value):
            raise ValueError("Constraint constants must be finite.")
        return value
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unknown variable in constraint: {node.id}")
        value = float(context[node.id])
        if not math.isfinite(value):
            raise ValueError(f"Variable '{node.id}' must contain a finite number.")
        return value
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(safe_eval_ast(node.operand, context))
    if isinstance(node, ast.BinOp):
        op = _SAFE_BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported arithmetic operator: {type(node.op).__name__}")
        left = safe_eval_ast(node.left, context)
        right = safe_eval_ast(node.right, context)
        if isinstance(node.op, ast.Div) and abs(right) <= np.finfo(float).eps:
            raise ValueError("Division by zero is not permitted in constraints.")
        value = op(left, right)
        if not math.isfinite(float(value)):
            raise ValueError("Constraint arithmetic produced a non-finite value.")
        return value
    if isinstance(node, ast.Compare):
        left = safe_eval_ast(node.left, context)
        for comparison, comparator in zip(node.ops, node.comparators):
            op = _SAFE_CMPS.get(type(comparison))
            if op is None:
                raise ValueError(
                    f"Unsupported comparison operator: {type(comparison).__name__}"
                )
            right = safe_eval_ast(comparator, context)
            if not op(left, right):
                return False
            left = right
        return True
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _normalize_single_equals(expression):
    """Keep the legacy ``A = 1`` syntax while preserving <=, >=, != and ==."""
    return re.sub(r"(?<![<>=!])=(?!=)", "==", expression)


def _feature_pattern(feature_name):
    # Unlike \b, these boundaries also work when a name ends in ')' or '%'.
    return rf"(?<![\w]){re.escape(feature_name)}(?![\w])"


def _replace_feature_names(expression, row_dict, features_config):
    context = {}
    replaced = expression
    active_names = [
        str(feature.get("name"))
        for feature in features_config or []
        if feature.get("name")
    ]
    for index, feature_name in enumerate(sorted(active_names, key=len, reverse=True)):
        placeholder = f"_var_{index}"
        pattern = _feature_pattern(feature_name)
        if re.search(pattern, replaced):
            if feature_name not in row_dict:
                raise ValueError(f"Missing value for feature '{feature_name}'.")
            try:
                value = float(row_dict[feature_name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature '{feature_name}' must be numeric in a numerical constraint."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"Feature '{feature_name}' must be finite.")
            context[placeholder] = value
            replaced = re.sub(pattern, placeholder, replaced)
    return replaced, context


def compile_numerical_expression(expr_str, feature_names):
    """Compile a feature equation into a safe AST and placeholder mapping."""
    clean = _normalize_single_equals(str(expr_str).strip())
    placeholders = {}
    replaced = clean
    for index, feature_name in enumerate(
        sorted([str(name) for name in feature_names if name], key=len, reverse=True)
    ):
        placeholder = f"_var_{index}"
        pattern = _feature_pattern(feature_name)
        if re.search(pattern, replaced):
            placeholders[placeholder] = feature_name
            replaced = re.sub(pattern, placeholder, replaced)
    tree = _parse_expression(replaced)
    allowed_names = set(placeholders)
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    unknown = sorted(referenced_names - allowed_names)
    if unknown:
        raise ValueError(f"Unknown feature name(s): {', '.join(unknown)}.")
    if not any(isinstance(node, ast.Compare) for node in ast.walk(tree)):
        raise ValueError("A numerical constraint must include a comparison operator.")
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name,
        ast.Constant, ast.Num, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.UAdd, ast.USub, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq,
        ast.NotEq,
    )
    unsupported = [
        type(node).__name__ for node in ast.walk(tree)
        if not isinstance(node, allowed_nodes)
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported expression element(s): {', '.join(sorted(set(unsupported)))}."
        )
    return tree, placeholders


def eval_numerical_expression(expr_str, row_dict, features_config):
    """Safely evaluate one numerical feature constraint against a candidate row."""
    clean = _normalize_single_equals(str(expr_str).strip())
    replaced, context = _replace_feature_names(clean, row_dict, features_config)
    tree = _parse_expression(replaced)
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    unknown = sorted(referenced_names - set(context))
    if unknown:
        raise ValueError(f"Unknown feature name(s): {', '.join(unknown)}.")
    if not any(isinstance(node, ast.Compare) for node in ast.walk(tree)):
        raise ValueError("A numerical constraint must include a comparison operator.")
    return bool(safe_eval_ast(tree, context))


def validate_feature_constraints(numerical_rules, features_config):
    """Validate all numerical rules before an expensive optimizer run starts."""
    feature_names = [feature.get("name") for feature in features_config or []]
    for index, expression in enumerate(numerical_rules or [], start=1):
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Numerical constraint {index} cannot be empty.")
        try:
            compile_numerical_expression(expression, feature_names)
        except ValueError as exc:
            raise ValueError(f"Invalid numerical constraint {index}: {exc}") from exc


def validate_feature_ranges(features_config):
    """Validate the Range / Values syntax used by BO and TransBO."""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    continuous_pattern = re.compile(
        rf"^\[\s*({number})\s*,\s*({number})\s*\]$"
    )
    regular_pattern = re.compile(
        rf"^\[\s*({number})\s*,\s*({number})\s*\]\s*(\d+)$"
    )
    for index, feature in enumerate(features_config or [], start=1):
        name = feature.get('name') or f'Feature {index}'
        feature_type = feature.get('type', 'continuous')
        range_value = str(feature.get('range', '')).strip()
        if feature_type == 'continuous':
            match = continuous_pattern.fullmatch(range_value)
            if not match:
                raise ValueError(
                    f"Feature '{name}' requires a continuous range like [min, max]."
                )
            low, high = map(float, match.groups())
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"Feature '{name}' requires min < max.")
        elif feature_type == 'regular':
            match = regular_pattern.fullmatch(range_value)
            if not match:
                raise ValueError(
                    f"Feature '{name}' requires a regular range like [min, max]steps."
                )
            low, high, steps = float(match.group(1)), float(match.group(2)), int(match.group(3))
            if not (math.isfinite(low) and math.isfinite(high) and low < high and steps >= 2):
                raise ValueError(
                    f"Feature '{name}' requires min < max and at least 2 steps."
                )
        elif feature_type == 'discrete':
            parts = [part.strip() for part in range_value.strip('[]').split(',') if part.strip()]
            if not parts:
                raise ValueError(f"Feature '{name}' requires one or more discrete values.")
            try:
                values = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(
                    f"Feature '{name}' contains a non-numeric discrete value."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Feature '{name}' values must be finite.")
        elif feature_type == 'categorical':
            get_categorical_choices(feature)
        else:
            raise ValueError(f"Feature '{name}' has unknown type '{feature_type}'.")


def check_constraints(row_dict, numerical_rules, disallowed_rules, features_config):
    """Return whether a decoded candidate satisfies all feature constraints."""
    feature_types = {feature['name']: feature.get('type') for feature in features_config}
    for rule in disallowed_rules or []:
        if not rule:
            continue
        violation = True
        for condition in rule:
            feature_name = condition.get("feature")
            feature_value = condition.get("value")
            matches = feature_name in row_dict
            if matches and feature_types.get(feature_name) in {'continuous', 'discrete', 'regular'}:
                try:
                    matches = math.isclose(float(row_dict[feature_name]), float(feature_value),
                                           rel_tol=0.0, abs_tol=CONSTRAINT_TOLERANCE)
                except (TypeError, ValueError):
                    matches = False
            elif matches:
                matches = str(row_dict[feature_name]).strip() == str(feature_value).strip()
            if not matches:
                violation = False
                break
        if violation:
            return False

    for expression in numerical_rules or []:
        try:
            if not eval_numerical_expression(expression, row_dict, features_config):
                return False
        except ValueError as exc:
            # API routes validate first; this protects internal callers as well.
            print(f"Error evaluating numerical constraint '{expression}': {exc}")
            return False
    return True


def validate_objective_constraints(objective_constraints, objectives_config):
    """Validate and normalize raw-scale objective bounds."""
    if objective_constraints in (None, []):
        return []
    if not isinstance(objective_constraints, list):
        raise ValueError("Objective constraints must be provided as a list.")

    valid_objectives = {
        str(objective.get("name"))
        for objective in objectives_config or []
        if objective.get("name")
    }
    normalized = []
    for index, constraint in enumerate(objective_constraints, start=1):
        if not isinstance(constraint, dict):
            raise ValueError(f"Objective constraint {index} must be an object.")
        objective = str(constraint.get("objective", "")).strip()
        operator_value = str(constraint.get("operator", "")).strip()
        operator_value = {"≥": ">=", "≤": "<="}.get(operator_value, operator_value)
        if objective not in valid_objectives:
            raise ValueError(
                f"Objective constraint {index} references unavailable objective '{objective}'."
            )
        if operator_value not in (">=", "<="):
            raise ValueError(
                f"Objective constraint {index} must use either '>=' or '<=' ."
            )
        try:
            limit = float(constraint.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Objective constraint {index} requires a numeric boundary."
            ) from exc
        if not math.isfinite(limit):
            raise ValueError(
                f"Objective constraint {index} requires a finite boundary."
            )
        normalized.append(
            {"objective": objective, "operator": operator_value, "value": limit}
        )
    return normalized


def check_objective_constraints(
    predicted_dict, objective_constraints, tolerance=CONSTRAINT_TOLERANCE
):
    """Evaluate validated raw predictions and return structured violations."""
    violations = []
    for constraint in objective_constraints or []:
        objective = constraint["objective"]
        operator_value = constraint["operator"]
        limit = float(constraint["value"])
        predicted = predicted_dict.get(objective)
        try:
            predicted_float = float(predicted)
        except (TypeError, ValueError):
            predicted_float = math.nan
        if not math.isfinite(predicted_float):
            violations.append(
                {
                    "objective": objective,
                    "operator": operator_value,
                    "limit": limit,
                    "predicted": None,
                    "reason": "Missing or non-finite prediction",
                }
            )
            continue
        violated = (
            operator_value == ">=" and predicted_float < limit - tolerance
        ) or (
            operator_value == "<=" and predicted_float > limit + tolerance
        )
        if violated:
            violations.append(
                {
                    "objective": objective,
                    "operator": operator_value,
                    "limit": limit,
                    "predicted": predicted_float,
                }
            )
    return not violations, violations
