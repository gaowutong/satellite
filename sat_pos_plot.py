import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CALC_POS_FILE = 'data/orbit.txt'

# sp3 = pd.read_table(SP3_FILE, sep=r'\s+', header=0)
# pos_calc = pd.read_table(POS_FILE, sep=r'\s+', header=0)
calc = pd.read_table(CALC_POS_FILE, sep=r'\s+', header=0)


def plot_orbit(sat, ax):
    g01_calc = calc[calc['prn'] == sat]
    ax.plot(xs=g01_calc['x'], ys=g01_calc['y'], zs=g01_calc['z'],
            label=sat)


prns = [f'G{i:02n}' for i in range(1, 33)]


def plot_sat_orbit():
    for prn in prns:
        fig = plt.figure(figsize=(12, 8), dpi=90)
        ax = Axes3D(fig)

        plot_orbit(prn, ax)
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        ax.set_zlabel("Z(m)")

        fig.suptitle(f"Satellite {prn} Orbit in Earth Coordinates", fontsize=16)
        fig.legend()
        fig.savefig(f'orbit_fig/{prn}.jpg')


def plot_sats_orbit():
    fig = plt.figure(figsize=(12, 8), dpi=90)
    ax = Axes3D(fig)
    ax.set_xlabel("X(m)", fontsize=14)
    ax.set_ylabel("Y(m)", fontsize=14)
    ax.set_zlabel("Z(m)", fontsize=14)
    for prn in prns:

        plot_orbit(prn, ax)

    fig.suptitle(f"Satellites Orbit in Earth Coordinates", fontsize=16)
    fig.legend()
    fig.savefig(f'orbit_fig/G01-G32.jpg')


if __name__ == '__main__':
    # plot_sat_orbit()
    plot_sats_orbit()