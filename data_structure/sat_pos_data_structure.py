__all__ = ['PositionData']

SP3_FILE = 'data/brdc2360.sp3'
POS_FILE = 'data/brdc2360.pos_calc'


class PositionItem:
    def __init__(self, sod, prn, x, y, z):
        self.SOD = float(sod)
        self.PRN = str(prn)
        self.XYZ = list(map(float, [x, y, z]))

    def __repr__(self):
        return ' '.join([super().__repr__(), *map(str, [self.SOD, self.PRN, self.XYZ])])

    def __str__(self):
        return ' '.join(["Position Item", *map(str, [self.SOD, self.PRN, self.XYZ])])


class PositionData:
    def __init__(self, file_name, data_type):
        self.data = []
        self.type = data_type
        with open(file_name) as f:
            # header
            f.readline()

            for line in f:
                item = PositionItem(*line.split())
                self.data.append(item)

    def __iter__(self):
        return self.data.__iter__()

    def __str__(self):
        return ' '.join([f'{self.type}', f'Length {len(self.data)}'])

    def get_pos_item(self, PRN=None, SOD=None):
        return self.get_pos_items(PRN, SOD)[0]

    def get_pos_items(self, PRN=None, SOD=None):
        """kwargs available: PRN, SOD"""
        kws = {}
        if PRN is not None:
            kws['PRN'] = PRN
        if SOD is not None:
            kws['SOD'] = SOD
        filtered_data = self.data
        for k, v in kws.items():
            filtered_data = list(filter(lambda dat: getattr(dat, k) == v, filtered_data))

        return list(filtered_data)


def test_pos_data():
    sp3 = PositionData(SP3_FILE, "SP3 Position Data")
    print(sp3)
    # for prn in sp3:
    #     print(prn)

    pos = PositionData(POS_FILE, "Ephemeris Position Data")
    print(pos)

    g01 = pos.get_pos_items(PRN='G01')
    print(f'length of G01: {len(g01)}')

    g01_pos_in_0100 = pos.get_pos_item(PRN="G01", SOD=1 * 60 * 60)
    g01_sp3_in_0100 = sp3.get_pos_item(PRN="G01", SOD=1 * 60 * 60)
    print(f'pos_calc: {g01_pos_in_0100}\nsp3: {g01_sp3_in_0100}')
    """
    SP3 Position Data Length 91200
    Ephemeris Position Data Length 91200
    length of G01: 2850
    pos_calc: Position Item 3600.0 G01 [-13205369.02, 15498123.151, 16724627.009]
    sp3: Position Item 3600.0 G01 [-13205373.919, 15498125.186, 16724633.917]
    """


if __name__ == '__main__':
    test_pos_data()
    pass
