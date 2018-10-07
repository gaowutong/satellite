from datetime import timedelta, datetime
from math import sqrt, sin, atan2, cos
import numpy as np
import pandas as pd

from model.nav_data_structure import RINEXNavData

__all__ = ['SatellitePositionCalc']
# Predefined Parameters
GM = 3.986005e14
omega_e = 7.2921151467e-5  # angular velocity of earth rotation


def calc_sod(t):
    return int(t.hour * 3600 + t.minute * 60 + t.second)


class SatellitePositionCalc:
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

    def __init__(self, nav_file):
        self.RINEX = RINEXNavData(nav_file)

    def calc_specified_time(self, prn, nav_time, calc_time):
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
        ] = [getattr(orbit_data, name) for name in self.data_format]

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

    def calc_nearest_ephemeris(self, prn_nav_data, prn, calc_time):
        nav_time = min(prn_nav_data, key=lambda bl: abs(bl.toc - calc_time)).toc
        xyz = self.calc_specified_time(prn, nav_time, calc_time)
        return xyz

    def calc_sat_orbit(self, prn, start_date_time: datetime, end_date_time: datetime, delta=30):
        nums = int((end_date_time - start_date_time).total_seconds() / delta)
        calc_times = [start_date_time + timedelta(seconds=delta * i) for i in range(nums + 1)]

        calc_times_idx = pd.DatetimeIndex(calc_times)
        position_list = []

        prn_nav_data = self.RINEX.get_nav_data(PRN=prn)

        for calc_time in calc_times:
            # Find the nearest ephemeris to current calculated time
            xyz = self.calc_nearest_ephemeris(prn_nav_data, prn, calc_time)
            position_list.append([calc_sod(calc_time), prn, *xyz])

        df_orbit = pd.DataFrame(
            np.array(position_list),
            columns=['sod', 'prn', "x", "y", "z"], index=calc_times_idx
        )
        # TODO control column types when df is created
        df_orbit['sod'] = df_orbit['sod'].astype(int)
        df_orbit['prn'] = df_orbit['prn'].astype(str)
        df_orbit[['x', 'y', 'z']] = df_orbit[['x', 'y', 'z']].astype(float)
        return df_orbit

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
