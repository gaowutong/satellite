import pandas as pd
from datetime import datetime, timedelta
from math import sqrt, sin, atan2, cos
import numpy as np

from data_structure.nav_data_structure import RINEXNavigationData

# Predefined Parameters
GM = 3.986005e14
omega_e = 7.2921151467e-5  # angular velocity of earth rotation

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

PRN_LIST = [f'G{i:02n}' for i in range(1, 33)]


class SatellitePositionCalc:

    def __init__(self, nav_file, sp3_file):
        self.RINEX = RINEXNavigationData(nav_file)
        self.sp3 = pd.read_table(sp3_file, sep=r'\s+', header=0)

    def instant_pos_calc(self, prn, nav_time, calc_time):
        """
        Calculate instant satellite position in calc_time using ephemeris in nav_time
        """

        orbit_data = next(
            filter(lambda bl: bl.toc == nav_time and bl.PRN == prn, self.RINEX.blocks)
        )

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

        coord = [
            x * cos(L) - y * cos(i) * sin(L),
            x * sin(L) + y * cos(i) * cos(L),
            y * sin(i)
        ]
        return coord

    def instant_pos_eval(self, prn, calc_time, computed_pos, allow_print=True):
        """Evaluate error of instant satellite position by compare calculated and .sp3 data"""
        sod = self.sod(calc_time)
        sp3_err = np.nan
        sp3_delta = [np.nan for _ in range(3)]
        try:
            xyz_sp3 = self.sp3[(self.sp3['prn'] == prn) & (self.sp3['sod'] == sod)].values[0, -3:]

        except Exception:
            print(f"No Such SP3 Data in {calc_time}(sod {sod}) of prn {prn}")
        else:
            sp3_err, sp3_delta = self.calc_3d_point_error(xyz_sp3, computed_pos)
            if allow_print:
                print(f'Calculated Position of {prn} in {calc_time}\n{computed_pos}')
                print(f'Difference of X Y Z between .sp3 and calculated position\n{sp3_delta}')
                print(f'Error between .sp3 and calculated position {sp3_err}')
                print()
        return sp3_err, sp3_delta

    # def diff_nav_calc_times(self, prn, nav_times, calc_times):
    #     err = []
    #     for nav_time in nav_times:
    #         for calc_time in calc_times:
    #             print(f"Navigation  {nav_time}\n"
    #                   f"Calculation {calc_time}")
    #             # Second of day
    #             tgt_sod = self.sod(calc_time)
    #             xyz = self.instant_pos_calc(prn, nav_time, calc_time)
    #             err, _ = self.instant_pos_eval(prn, tgt_sod, xyz)
    #             err.append(err)
    #             print()
    #
    #     return np.array(err).reshape(len(nav_times), len(calc_times))

    def sat_orbit_calc(self):
        start_date_time = datetime(2018, 8, 24, 0, 0, 0)
        end_date_time = datetime(2018, 8, 24, 23, 45, 0)
        nums = int((end_date_time - start_date_time).total_seconds() / 30)
        calc_times = [start_date_time + timedelta(seconds=30 * i) for i in range(nums + 1)]

        calc_times_idx = pd.DatetimeIndex(calc_times)
        columns = ['sod', 'prn', "x", "y", "z", 'ephemeris toc']
        sats_orbit = []
        for prn in PRN_LIST:
            pos = []
            prn_nav_data = self.RINEX.get_nav_data(PRN=prn)
            for calc_time in calc_times:
                # Find the nearest ephemeris to current calculated time
                nav_time = min(prn_nav_data, key=lambda bl: abs(bl.toc - calc_time)).toc
                xyz = self.instant_pos_calc(prn, nav_time, calc_time)
                # self.instant_pos_eval(prn, nav_time, xyz)

                pos.append([self.sod(calc_time), prn, *xyz, nav_time])

            sat_orbit = pd.DataFrame(np.array(pos), columns=columns, index=calc_times_idx)
            sats_orbit.append(sat_orbit)
            print(f'{prn} finished')
        return pd.concat(sats_orbit)

    @staticmethod
    def calc_3d_point_error(p1, p2):
        dx_dy_dz = [i - j for i, j in zip(p1, p2)]
        ds = sqrt(sum([k ** 2 for k in dx_dy_dz]))
        return ds, dx_dy_dz

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
    calc = SatellitePositionCalc(NAV_FILE, SP3_FILE)

    calc.sat_orbit_calc()
    pass
