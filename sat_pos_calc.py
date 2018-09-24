import pandas as pd
from datetime import datetime, timedelta
from math import sqrt, sin, atan2, cos
import numpy as np

from data_structure.nav_data_structure import RINEXNavigationData
from data_structure.sat_pos_data_structure import PositionData

# Predefined Parameters
GM = 3.986005e14
omega_e = 7.292115e-5  # angular velocity of earth rotation

NAV_FILE = 'data/brdc2360.18n'
SP3_FILE = 'data/brdc2360.sp3'
POS_FILE = 'data/brdc2360.pos'

CALC_FILE = 'data/orbit.txt'

# RINEX Navigation Data

data_format = [
    'PRN', 'toc', 'a0', 'a1', 'a2',
    'IODE', 'Crs', 'delta_n', 'M0',
    'Cuc', 'e', 'Cus', 'sqrt_A',
    'toe', 'Cic', 'Omega0', 'Cis',
    'i0', 'Crc', 'omega', 'Omega_dot',
    'di_dt', 'L2', 'GPS_week', 'L2P_mark',
    'sat_precision', 'health_state', 'TGD', 'IODC_age',
    'send_time', 'sim_range', 'reserve1', 'reserve2'
]


class SatellitePositionCalc:

    def __init__(self):
        self.RINEX = RINEXNavigationData(NAV_FILE)
        self.sp3 = PositionData(SP3_FILE, "SP3")
        self.pos = PositionData(POS_FILE, "POS")

    def satellite_position(self, prn, nav_time, calc_time):
        """
        Calculate satellite position using nav_date in nav_time
        after calc_time in sec
        """

        condition = lambda bl: bl.toc == nav_time and bl.PRN == prn
        orbit_data = next(filter(condition, self.RINEX.blocks))

        [
            PRN, toc, a0, a1, a2,
            IODE, Crs, delta_n, M0,
            Cuc, e, Cus, sqrt_A,
            toe, Cic, Omega0, Cis,
            i0, Crc, omega, Omega_dot,
            di_dt, L2, GPS_week, L2P_mark,
            sat_precision, health_state, TGD, IODC_age,
            send_time, sim_range, reserve1, reserve2
        ] = [getattr(orbit_data, name) for name in data_format]

        # t: sec_delay in gps week (s)

        # t = toe + dt
        t = self.get_gps_time(calc_time)
        dt = t - toe

        # Computation Steps
        n0 = sqrt(GM) / sqrt_A ** 3

        n = n0 + delta_n

        M = M0 + n * dt

        E = self.iterateE(M, e)

        f = atan2(sqrt(1 - e ** 2) * sin(E), cos(E) - e)

        u_prime = omega + f

        delta_u_r_i = np.array([
            [Cuc, Cus],
            [Crc, Crs],
            [Cic, Cis],
        ]) @ np.array([
            cos(2 * u_prime), sin(2 * u_prime)
        ])

        u, r, i = np.array([
            u_prime,
            sqrt_A ** 2 * (1 - e * cos(E)),
            i0 + di_dt * dt
        ]) + delta_u_r_i

        x = r * cos(u)
        y = r * sin(u)

        L = Omega0 + Omega_dot * dt - omega_e * t

        coor = [
            x * cos(L) - y * cos(i) * sin(L),
            x * sin(L) + y * cos(i) * cos(L),
            y * sin(i)
        ]

        return coor

    def compare_with_sp3(self, prn, sod, computed_pos, print_dxdydz=False):

        # XYZ_pos = self.pos.get_pos_item(PRN=prn, SOD=sod).XYZ

        try:
            XYZ_sp3 = self.sp3.get_pos_item(PRN=prn, SOD=sod).XYZ

        except Exception:
            sp3_err = 0
            print(f"No Such SP3 Data in sod {sod}")
        else:
            print(f'Calculated Satellite Position {computed_pos}')
            # pos_err, pos_delta = self.calc_3d_point_error(XYZ_pos, computed_pos)
            sp3_err, sp3_delta = self.calc_3d_point_error(XYZ_sp3, computed_pos)

            # print(f'Error between .pos_calc {pos_err}')
            print(f'Error between .sp3 and calculated position {sp3_err}')
            if print_dxdydz:
                # print(f'Delta between .pos_calc  {pos_delta}')
                print(f'Delta between .sp3 and calculated position {sp3_delta}')

        return sp3_err

    def calc_sat_pos_in_diff_time(self, prn, nav_times, calc_times):
        errs = []
        for nav_time in nav_times:
            for calc_time in calc_times:
                print(f"Navigation  {nav_time}\n"
                      f"Calculation {calc_time}")
                # Second of day
                tgt_sod = self.sod(calc_time)
                xyz = self.satellite_position(prn, nav_time, calc_time)
                err = self.compare_with_sp3(prn, tgt_sod, xyz, True)
                errs.append(err)
                print()

        return np.array(errs).reshape(len(nav_times), len(calc_times))

    def _calc_whole_day_sat_pos(self, prn, f):
        start = datetime(2018, 8, 24, 0, 0, 0)
        end = datetime(2018, 8, 24, 23, 45, 00)
        nums = int((end - start).total_seconds() / 30)
        calc_times = [start + timedelta(seconds=30 * i) for i in range(nums + 1)]

        prn_nav_data = self.RINEX.get_nav_data(PRN=prn)
        for calc_time in calc_times:
            nav_time = min(prn_nav_data, key=lambda bl: abs(bl.toc - calc_time)).toc
            print(f"PRN {prn}: Navigation {nav_time} => Calculation {calc_time}")
            # Second of day
            tgt_sod = self.sod(calc_time)
            xyz = self.satellite_position(prn, nav_time, calc_time)
            # compare_with_sp3(prn, tgt_sod, xyz)

            x, y, z = xyz
            print(f'{tgt_sod:>8.2f}{prn:>5}{x:>15.3f}{y:>15.3f}{z:>15.3f}', file=f)
            # print()

    def calc_whole_day_sats_pos(self):
        prns = [f'G{i:02n}' for i in range(1, 33)]
        with open(CALC_FILE, 'w')as f:
            print(f'{"sod":>8}{"prn":>5}{"x":>15}{"y":>15}{"z":>15}', file=f)
            for prn in prns:
                self._calc_whole_day_sat_pos(prn, f)

    def analysis_errors(self):
        prn = 'G01'

        prn_nav_data = self.RINEX.get_nav_data(PRN=prn)
        nav_times_errs = pd.DataFrame(index=[f"{i*30} minutes" for i in range(-2, 3)])
        for nav in prn_nav_data:
            nav_times = [nav.toc]
            calc_times = [nav_times[0] + timedelta(minutes=i * 30) for i in range(-2, 3)]
            errs = self.calc_sat_pos_in_diff_time(prn, nav_times, calc_times)
            nav_times_errs[nav.toc] = errs.T
        return nav_times_errs

    @staticmethod
    def calc_3d_point_error(p1, p2):
        dxdydz = [i - j for i, j in zip(p1, p2)]
        return sqrt(sum([(k ** 2) for k in dxdydz])), dxdydz

    @staticmethod
    def get_gps_time(tgt_t):
        return (tgt_t.weekday() + 1) * 86400 + tgt_t.hour * 3600 + tgt_t.minute * 60 + tgt_t.second

    @staticmethod
    def iterateE(M, e):
        """
        Using Newton iteration to compute E
        Equation: E = M + e * sin(E)
        """
        E = M - e
        while True:
            lst = E
            E = E - (E - e * sin(E) - M) / (1 - e * cos(E))
            if abs(E - lst) < 1e-6:
                break
        return E

    @staticmethod
    def sod(t):
        return t.hour * 3600 + t.minute * 60 + t.second


if __name__ == '__main__':
    pos_calc = SatellitePositionCalc()

    prn = "G01"

    nav_times = [
        datetime(2018, 8, 24, 0, 0, 0),
        datetime(2018, 8, 24, 1, 30, 0),
    ]
    calc_times = [datetime(2018, 8, 24, 1, 0, 0)]

    pos_calc.calc_sat_pos_in_diff_time(prn, nav_times, calc_times)

    """
    Navigation  2018-08-24 00:00:00
    Calculation 2018-08-24 01:00:00
    Calculated Satellite Position [-13205383.694849968, 15498118.166812062, 16724633.352198847]
    Error between .sp3 and calculated position 12.048038938256509
    Delta between .sp3 and calculated position [9.775849968194962, 7.019187938421965, 0.5648011527955532]

    Navigation  2018-08-24 01:30:00
    Calculation 2018-08-24 01:00:00
    Calculated Satellite Position [-13205378.92409278, 15498114.71283738, 16724627.009004006]
    Error between .sp3 and calculated position 13.507719929033005
    Delta between .sp3 and calculated position [5.005092781037092, 10.47316262125969, 6.90799599327147]
    """

    pos_calc.calc_whole_day_sats_pos()

    # df = pos_calc.analysis_errors()
    # df.to_csv('errors/errors.csv')
    pass
