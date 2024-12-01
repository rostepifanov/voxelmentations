import pytest

import numpy as np
import voxelmentations as V

@pytest.mark.core
def test_Identity_CASE_to_dict():
    p = 1
    always_apply = True

    expected_state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'Identity',
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.Identity(p=p, always_apply=always_apply)
    output_state_dict = transformation.to_dict()

    assert output_state_dict == expected_state_dict

@pytest.mark.core
def test_Identity_CASE_from_dict():
    p = 0.5
    always_apply = True

    state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'Identity',
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.from_dict(state_dict)

    assert pytest.approx(transformation.p) == p
    assert transformation.always_apply == always_apply

@pytest.mark.core
def test_Sequential_CASE_to_dict():
    p = 1
    always_apply = True

    expected_state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'Sequential',
            'transformations': [
                {
                    '__class_fullname__': 'Identity',
                    'p': p,
                    'always_apply': always_apply,
                }
            ],
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.Sequential([
        V.Identity(p=p, always_apply=always_apply),
    ], p=p, always_apply=always_apply)

    output_state_dict = transformation.to_dict()

    assert output_state_dict == expected_state_dict

@pytest.mark.core
def test_Sequential_CASE_from_dict():
    p = 0.5
    always_apply = True

    state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'Sequential',
            'transformations': [
                {
                    '__class_fullname__': 'Identity',
                    'p': p,
                    'always_apply': always_apply,
                }
            ],
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.from_dict(state_dict)

    assert pytest.approx(transformation.p) == p
    assert transformation.always_apply == always_apply

    assert len(transformation.transformations) == 1

    assert pytest.approx(transformation[0].p) == p
    assert transformation[0].always_apply == always_apply

@pytest.mark.core
def test_OneOf_CASE_to_dict():
    p = 1
    always_apply = True

    expected_state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'OneOf',
            'transformations': [
                {
                    '__class_fullname__': 'Identity',
                    'p': p,
                    'always_apply': always_apply,
                }
            ],
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.OneOf([
        V.Identity(p=p, always_apply=always_apply),
    ], p=p, always_apply=always_apply)

    output_state_dict = transformation.to_dict()

    assert output_state_dict == expected_state_dict

@pytest.mark.core
def test_OneOf_CASE_from_dict():
    p = 0.5
    always_apply = True

    state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'OneOf',
            'transformations': [
                {
                    '__class_fullname__': 'Identity',
                    'p': p,
                    'always_apply': always_apply,
                }
            ],
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.from_dict(state_dict)

    assert pytest.approx(transformation.p) == p
    assert transformation.always_apply == always_apply

    assert len(transformation.transformations) == 1

    assert pytest.approx(transformation[0].p) == p
    assert transformation[0].always_apply == always_apply

@pytest.mark.core
def test_Nested_Compose_CASE_to_dict():
    p = 1
    always_apply = True

    expected_state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'Sequential',
            'transformations': [
                {
                    '__class_fullname__': 'OneOf',
                    'transformations': [
                        {
                            '__class_fullname__': 'Identity',
                            'p': p,
                            'always_apply': always_apply,
                        },
                    ],
                    'p': p,
                    'always_apply': always_apply,
                }
            ],
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.Sequential([
        V.OneOf([
            V.Identity(p=p, always_apply=always_apply),
        ], p=p, always_apply=always_apply),
    ], p=p, always_apply=always_apply)

    output_state_dict = transformation.to_dict()

    assert output_state_dict == expected_state_dict

@pytest.mark.core
def test_Nested_Compose_CASE_from_dict():
    p = 0.5
    always_apply = True

    state_dict = {
        '__version__': V.__version__,
        'transformation': {
            '__class_fullname__': 'Sequential',
            'transformations': [
                {
                    '__class_fullname__': 'OneOf',
                    'transformations': [
                        {
                            '__class_fullname__': 'Identity',
                            'p': p,
                            'always_apply': always_apply,
                        },
                    ],
                    'p': p,
                    'always_apply': always_apply,
                }
            ],
            'p': p,
            'always_apply': always_apply,
        }
    }

    transformation = V.from_dict(state_dict)

    assert pytest.approx(transformation.p) == p
    assert transformation.always_apply == always_apply

    assert len(transformation.transformations) == 1

    assert pytest.approx(transformation[0].p) == p
    assert transformation[0].always_apply == always_apply

    assert len(transformation[0].transformations) == 1

    assert pytest.approx(transformation[0][0].p) == p
    assert transformation[0][0].always_apply == always_apply
