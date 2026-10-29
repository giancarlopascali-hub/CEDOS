"""Portable, versioned CSV exchange for EDOS and TransBO setup panels."""

import csv
import io
import math

from src.core.constraints import (
    get_categorical_choices,
    validate_feature_constraints,
    validate_feature_ranges,
    validate_objective_constraints,
)


SCHEMA_VERSION = '1'
MODULES = {'doe', 'bo', 'sa', 'tbo'}
HEADER = (
    'schema_version', 'source_module', 'record_type', 'group_id', 'name',
    'type', 'range_or_values', 'target', 'importance', 'included',
    'expression', 'operator', 'value',
)


def _text(value):
    return '' if value is None else str(value)


def _included(value, row_number):
    if value == '':
        return True
    if value.lower() not in {'true', 'false'}:
        raise ValueError(f'Row {row_number}: included must be true or false.')
    return value.lower() == 'true'


def _finite_optional(value, label, row_number, *, percent=False):
    if value == '':
        return
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f'Row {row_number}: {label} must be a number.') from exc
    if not math.isfinite(number) or (percent and not 0 <= number <= 100):
        detail = ' between 0 and 100' if percent else ''
        raise ValueError(f'Row {row_number}: {label} must be finite{detail}.')


def serialize_setup_csv(setup, module):
    """Write one module's live Setup state without altering draft values."""
    if module not in MODULES:
        raise ValueError('Unknown setup module.')
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=HEADER, lineterminator='\r\n')
    writer.writeheader()

    def write(kind, **fields):
        writer.writerow({
            'schema_version': SCHEMA_VERSION,
            'source_module': module,
            'record_type': kind,
            **{key: _text(value) for key, value in fields.items()},
        })

    write('setup')
    for feature in setup.get('features', []):
        write('feature', name=feature.get('name'), type=feature.get('type'),
              range_or_values=feature.get('range'),
              included=str(feature.get('included', True)).lower())
    for objective in setup.get('objectives', []):
        write('objective', name=objective.get('name'), type=objective.get('type'),
              target=objective.get('target'), importance=objective.get('importance'),
              included=str(objective.get('included', True)).lower())
    constraints = setup.get('constraints') or {}
    for expression in constraints.get('numerical', []):
        if _text(expression).strip():
            write('numerical_constraint', expression=expression)
    for index, conditions in enumerate(constraints.get('disallowed', []), start=1):
        for condition in conditions:
            write('disallowed_condition', group_id=f'rule_{index}',
                  name=condition.get('feature'), value=condition.get('value'))
    for constraint in setup.get('objective_constraints', []):
        write('objective_constraint', name=constraint.get('objective'),
              operator=constraint.get('operator'), value=constraint.get('value'))
    return '\ufeff' + output.getvalue()


def parse_setup_csv(csv_text, destination, columns=None):
    """Validate a Setup CSV and map supported records to a destination module.

    Dataset values are deliberately absent here. BO may search outside the
    values and categories observed in previous experiments.
    """
    if destination not in MODULES:
        raise ValueError('Unknown destination module.')
    if not isinstance(csv_text, str) or '\x00' in csv_text:
        raise ValueError('The setup CSV contains invalid text.')
    reader = csv.DictReader(io.StringIO(csv_text.lstrip('\ufeff'), newline=''),
                            restkey='_extra', strict=True)
    if reader.fieldnames is None or len(reader.fieldnames) != len(HEADER) or set(reader.fieldnames) != set(HEADER):
        raise ValueError('This is not an EDOS setup CSV, or its columns have changed.')

    features, objectives, numerical, objective_constraints = [], [], [], []
    groups = {}
    source = None
    warnings = []
    try:
        for number, row in enumerate(reader, start=2):
            if row.get('_extra') is not None or any(value is None for value in row.values()):
                raise ValueError(f'Row {number}: incorrect number of CSV columns.')
            if not any(_text(value).strip() for value in row.values()):
                continue
            if row['schema_version'] != SCHEMA_VERSION:
                raise ValueError(f'Row {number}: unsupported setup CSV version.')
            if row['source_module'] not in MODULES:
                raise ValueError(f'Row {number}: unknown source module.')
            if source is not None and row['source_module'] != source:
                raise ValueError(f'Row {number}: mixed source modules are not allowed.')
            source = row['source_module']
            kind = row['record_type']
            name = row['name']
            if kind == 'setup':
                continue
            if kind == 'feature':
                if not name.strip():
                    raise ValueError(f'Row {number}: feature name is required.')
                if row['type'] not in {'continuous', 'regular', 'discrete', 'categorical'}:
                    raise ValueError(f'Row {number}: unknown feature type.')
                features.append({'name': name, 'type': row['type'],
                                 'range': row['range_or_values'],
                                 'included': _included(row['included'], number)})
            elif kind == 'objective':
                if not name.strip():
                    raise ValueError(f'Row {number}: objective name is required.')
                if row['type'] not in {'maximize', 'minimize', 'target'}:
                    raise ValueError(f'Row {number}: unknown objective type.')
                _finite_optional(row['target'], 'target', number)
                _finite_optional(row['importance'], 'importance', number, percent=True)
                objectives.append({'name': name, 'type': row['type'],
                                   'target': row['target'], 'importance': row['importance'],
                                   'included': _included(row['included'], number)})
            elif kind == 'numerical_constraint':
                if row['expression'].strip():
                    numerical.append(row['expression'])
            elif kind == 'disallowed_condition':
                if not row['group_id'] or not name or not row['value']:
                    raise ValueError(f'Row {number}: disallowed condition needs group, feature and value.')
                groups.setdefault(row['group_id'], []).append({'feature': name, 'value': row['value']})
            elif kind == 'objective_constraint':
                if not name:
                    raise ValueError(f'Row {number}: objective constraint needs a name.')
                operator = {'≥': '>=', '≤': '<='}.get(row['operator'], row['operator'])
                if operator not in {'>=', '<='}:
                    raise ValueError(f'Row {number}: objective constraint must use >= or <=.')
                _finite_optional(row['value'], 'objective constraint value', number)
                objective_constraints.append({'objective': name, 'operator': operator,
                                              'value': row['value']})
            else:
                raise ValueError(f'Row {number}: unknown record type {kind!r}.')
    except csv.Error as exc:
        raise ValueError(f'Invalid CSV quoting: {exc}.') from exc

    names = [item['name'] for item in features + objectives]
    if len(names) != len(set(names)):
        raise ValueError('Feature and objective names must be unique within the setup CSV.')
    if source is None:
        raise ValueError('The setup CSV has no settings rows.')

    # Validate declared domains independently of historical dataset values.
    for feature in features:
        if feature['range'].strip():
            validate_feature_ranges([feature])
        elif source in {'doe', 'bo', 'tbo'}:
            warnings.append(f"Feature '{feature['name']}' has no range or values yet.")
    validate_feature_constraints(numerical, features)
    feature_by_name = {feature['name']: feature for feature in features}
    for conditions in groups.values():
        for condition in conditions:
            feature = feature_by_name.get(condition['feature'])
            if feature is None or feature['type'] != 'categorical':
                raise ValueError(f"Disallowed condition references unavailable categorical feature '{condition['feature']}'.")
            if condition['value'] not in get_categorical_choices(feature):
                raise ValueError(f"Disallowed condition uses an unknown value for '{condition['feature']}'.")
    for constraint in objective_constraints:
        objective = next((item for item in objectives if item['name'] == constraint['objective']), None)
        if objective is not None and not objective['included']:
            raise ValueError(f"Objective constraint references excluded objective '{constraint['objective']}'.")
        if constraint['value']:
            validate_objective_constraints([constraint], objectives)
        elif constraint['objective'] not in {item['name'] for item in objectives}:
            raise ValueError(f"Objective constraint references unavailable objective '{constraint['objective']}'.")
        else:
            warnings.append(f"Objective constraint for '{constraint['objective']}' has no boundary yet.")

    if destination in {'bo', 'sa', 'tbo'}:
        available = list(columns or [])
        if not available:
            raise ValueError('Load an experimental dataset before importing BO, TransBO or SA setup.')
        if len(available) != len(set(available)):
            raise ValueError('Dataset column names must be unique to import setup.')
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError('Setup names missing from the dataset: ' + ', '.join(missing))

    if destination == 'doe':
        dropped = [item['name'] for item in features + objectives if not item['included']]
        if dropped:
            warnings.append('DoE has no Include control; excluded rows were omitted: ' + ', '.join(dropped))
        features = [item for item in features if item['included']]
        objectives = [item for item in objectives if item['included']]
        active_features = {item['name'] for item in features}
        try:
            validate_feature_constraints(numerical, features)
        except ValueError as exc:
            raise ValueError('DoE cannot restore a constraint on an excluded feature.') from exc
        if any(condition['feature'] not in active_features
               for conditions in groups.values() for condition in conditions):
            raise ValueError('DoE cannot restore a disallowed combination on an excluded feature.')
        if objective_constraints:
            warnings.append('DoE has no objective constraints; those rows were not applied.')
        objective_constraints = []
    if destination == 'sa':
        for feature in features:
            if feature['type'] == 'regular':
                feature['type'] = 'discrete'
                warnings.append(f"SA treats regular feature '{feature['name']}' as discrete.")
            if feature['range']:
                warnings.append(f"SA does not use a range or value list for '{feature['name']}'.")
                feature['range'] = ''
        if numerical or groups:
            warnings.append('SA has no feature constraints; those rows were not applied.')
        numerical, groups = [], {}
    if destination == 'tbo' and groups:
        warnings.append('TransBO has no disallowed-combination controls; those rows were not applied.')
        groups = {}
    if destination in {'doe', 'bo', 'tbo'} and source == 'sa':
        warnings.append('SA files have no feature ranges; existing ranges are retained where available.')

    return {
        'source_module': source,
        'destination_module': destination,
        'features': features,
        'objectives': objectives,
        'constraints': {'numerical': numerical, 'disallowed': list(groups.values())},
        'objective_constraints': objective_constraints,
        'warnings': warnings,
    }
