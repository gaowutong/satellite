from datetime import datetime
import re

__all__ = ['RINEXNavigationData', 'RINEXHeaderData', 'NavBlockData']

NAV_FILE = 'data/brdc2360.18n'
SP3_FILE = 'data/brdc2360.sp3'
POS_FILE = 'data/brdc2360.pos_calc'


class RINEXNavigationData:
    """
    The Whole RINEX Data File Structure
    """

    def __init__(self, file_name):
        with open(file_name) as f:
            header_str = ''.join([f.readline() for _ in range(8)])
            self.header = RINEXHeaderData(header_str)

            self.blocks = []

            while True:
                i = f.tell()
                block_str = ''.join([f.readline() for _ in range(8)])
                if i == f.tell():
                    break
                block = NavBlockData(block_str)
                self.blocks.append(block)

    def get_nav_data(self, PRN=None):
        return list(filter(lambda bl: bl.PRN == PRN, self.blocks))

    pass


class RINEXHeaderData:
    """
    RINEX Header Data Structure
    Normally 8 lines
    """

    def __init__(self, str_data: str):
        self.lines = str_data.strip().split('\n')
        assert len(self.lines) == 8
        assert self.lines[-1].strip() == 'END OF HEADER'

    pass


class NavBlockData:
    """
    Single Block Data Structure
    Start From PRN number TOC
    """

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

    a = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
         'IODE', 'Crs', 'DeltaN', 'M0',
         'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
         'TimeEph', 'Cic', 'OMEGA', 'CIS',
         'Io', 'Crc', 'omega', 'OMEGA DOT',
         'IDOT', 'CodesL2', 'GPSWeek', 'L2Pflag',
         'SVacc', 'SVhealth', 'TGD', 'IODC',
         'TransTime', 'FitIntvl'
         ]

    def __init__(self, str_data: str):
        PRN = int(str_data[:2])
        self.PRN = f'G{PRN:02n}'

        _y, m, d, hh, mm = map(int, str_data[2:17].split())
        ss = int(str_data[17:20].strip())
        ms = int(str_data[21]) * 1000
        y = 2000 + _y
        self.toc = datetime(y, m, d, hh, mm, ss, ms)

        remain_data = list(map(float, re.findall(r'[-| ]\d\.\d{12}e[-|+]\d{2}', str_data.replace('D', 'e'))))
        for index, name in enumerate(self.data_format[2:]):
            data = remain_data[index]
            setattr(self, name, data)

    def __getitem__(self, item):
        return getattr(self, item, None)

    def __repr__(self):
        repr_list = {}

        for name in ["PRN", "toc"]:
            repr_list[name] = getattr(self, name, None)
        return ' '.join([self.__class__.__name__, repr_list.__repr__()])


def show_ordered_data():
    nav = RINEXNavigationData(NAV_FILE)
    nav.blocks.sort(key=lambda bl: bl.PRN)
    for block in nav.blocks:
        print(block)
    pass


if __name__ == '__main__':
    show_ordered_data()
    pass
