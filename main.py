from datetime import datetime, timedelta
from math import sqrt

import pandas as pd

from model.sat_pos_calc import SatellitePositionCalc

NAV_FILE = 'data/brdc2360.18n'
SP3_FILE = 'data/brdc2360.sp3'
POS_FILE = 'data/brdc2360.pos'

calc = SatellitePositionCalc(NAV_FILE)

PRN_LIST = [f'G{i:02n}' for i in range(1, 33)]

sp3 = pd.read_table(SP3_FILE, sep=r'\s+', header=0)


def calc_3d_point_error(p1, p2):
    dx_dy_dz = [i - j for i, j in zip(p1, p2)]
    ds = sqrt(sum([k ** 2 for k in dx_dy_dz]))
    return ds, dx_dy_dz


def calc_sod(t):
    return t.hour * 3600 + t.minute * 60 + t.second


def evaluate_precision(prn, calc_time, computed_pos, allow_print=True):
    """Evaluate error of instant satellite position by compare calculated and .sp3 data"""
    sod = calc_sod(calc_time)
    try:
        xyz_sp3 = sp3[(sp3['prn'] == prn) & (sp3['sod'] == sod)].values[0, -3:]
    except IndexError:
        print(f'No Such SP3 Data in {calc_time}(sod {sod}) of prn {prn}')
        return

    sp3_err, sp3_delta = calc_3d_point_error(xyz_sp3, computed_pos)
    if allow_print:
        print(f'Calculated Position of {prn} in {calc_time}\n{computed_pos}')
        print(f'Difference of X Y Z between .sp3 and calculated position\n{sp3_delta}')
        print(f'Error between .sp3 and calculated position {sp3_err}')
        print()
    return sp3_err


def test_calc_instant_pos():
    prn = "G01"
    nav_times = [
        datetime(2018, 8, 24, 0, 0, 0),
        datetime(2018, 8, 24, 1, 30, 0),
    ]
    calc_time = datetime(2018, 8, 24, 1, 0, 0)

    for nav_time in nav_times:
        xyz = calc.calc_specified_time(prn, nav_time=nav_time, calc_time=calc_time)
        err = evaluate_precision(prn, calc_time, xyz, True)
    pass


def test_diff_time_span():
    prn = 'G01'

    prn_nav_data = calc.RINEX.get_nav_data(PRN=prn)
    nav_times = list(map(lambda bl: bl.toc, prn_nav_data))

    errs_list = []
    for nav_time in nav_times:
        calc_times = [nav_time + timedelta(minutes=i * 30) for i in range(-2, 3)]
        errs_diff_span = []
        for calc_time in calc_times:
            xyz = calc.calc_specified_time(prn, nav_time, calc_time)
            err = evaluate_precision(prn, calc_time, xyz, allow_print=False)
            if err:
                errs_diff_span.append(err)
        errs_list.append(errs_diff_span)
    return errs_list
    pass


def test_sat_orbits():
    start_date_time = datetime(2018, 8, 24, 0, 0, 0)
    end_date_time = datetime(2018, 8, 24, 23, 45, 0)
    orbit_dict = {}
    for prn in PRN_LIST:
        df_orbit = calc.calc_sat_orbit(prn, start_date_time, end_date_time)
        orbit_dict[prn] = df_orbit
        print(f'{prn} finished')
    pass


if __name__ == '__main__':
    test_sat_orbits()
