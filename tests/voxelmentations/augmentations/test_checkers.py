import pytest

import voxelmentations.augmentations.checkers as C

def test_prepare_float_CASE_wrong_type():
    with pytest.raises(ValueError):
        C.prepare_float(None, '')

def test_prepare_float_CASE_int():
    output = C.prepare_float(1, '')

    assert pytest.approx(output) == 1.

def test_prepare_inrange_zero_one_float_CASE_correct():
    output = C.prepare_inrange_zero_one_float(0.5, '')

    assert pytest.approx(output) == 0.5

def test_prepare_inrange_zero_one_float_CASE_greater_than_one_value():
    with pytest.raises(ValueError):
        C.prepare_inrange_zero_one_float(1.5, '')

def test_prepare_int_asymrange_CASE_wrong_type():
    with pytest.raises(ValueError):
        C.prepare_int_asymrange(None, '', 0)

def test_prepare_int_asymrange_CASE_float():
    with pytest.raises(ValueError):
        C.prepare_int_asymrange(1., '', 0)

def test_prepare_int_asymrange_CASE_low():
    with pytest.raises(ValueError):
        C.prepare_int_asymrange(-1, '', 0)

def test_prepare_int_asymrange_CASE_int():
    output = C.prepare_int_asymrange(1, '', 0)
    expected = (0, 1)

    assert output == expected

def test_prepare_int_asymrange_CASE_tuple_int():
    output = C.prepare_int_asymrange((1, 2), '', 0)
    expected = (1, 2)

    assert output == expected

def test_prepare_int_asymrange_CASE_tuple_AND_wrong_type():
    with pytest.raises(ValueError):
        C.prepare_int_asymrange((None, None), '', 0)

def test_prepare_int_asymrange_CASE_tuple_int_AND_low():
    with pytest.raises(ValueError):
        C.prepare_int_asymrange((-1, 2), '', 0)

def test_prepare_int_asymrange_CASE_tuple_int_AND_long():
    with pytest.raises(ValueError):
        C.prepare_int_asymrange((0, 1, 2), '', 0)

def test_prepare_float_asymrange_CASE_wrong_type():
    with pytest.raises(ValueError):
        C.prepare_float_asymrange(None, '', 0.)

def test_prepare_float_asymrange_CASE_int():
    with pytest.raises(ValueError):
        C.prepare_float_asymrange(1, '', 0.)

def test_prepare_float_asymrange_CASE_low():
    with pytest.raises(ValueError):
        C.prepare_float_asymrange(-1., '', 0.)

def test_prepare_float_asymrange_CASE_float():
    output = C.prepare_float_asymrange(1., '', 0.)
    expected = (0., 1.)

    assert pytest.approx(output) == expected

def test_prepare_float_asymrange_CASE_tuple_float():
    output = C.prepare_float_asymrange((1., 2.), '', 0.)
    expected = (1., 2.)

    assert pytest.approx(output) == expected

def test_prepare_float_asymrange_CASE_tuple_AND_wrong_type():
    with pytest.raises(ValueError):
        C.prepare_float_asymrange((None, None), '', 0.)

def test_prepare_float_asymrange_CASE_tuple_float_AND_low():
    with pytest.raises(ValueError):
        C.prepare_float_asymrange((-1., 2.), '', 0.)

def test_prepare_int_asymrange_CASE_tuple_float_AND_long():
    with pytest.raises(ValueError):
        C.prepare_float_asymrange((0., 1., 2.), '', 0.)
