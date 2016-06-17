__doc__ = "Visualize HDF5 files generating during propagation"

import numpy as np

# load tools for creating animation
import sys
import matplotlib
import h5py

if sys.platform == 'darwin':
    # only for MacOS
    matplotlib.use('TKAgg')

import matplotlib.animation
import matplotlib.pyplot as plt

class VisualizeDynamicsPhaseSpace:
    """
    Class to visualize dynamics in phase space previously saved into the file
    """

    def __init__(self, fig, file_results):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        :param file_results: HDF5 file to save results
        """
        self.fig = fig
        self.file_results = file_results

        #################################################################
        #
        # Load some quantum system's settings
        #
        #################################################################

        self.quant_sys = dict(
            (key, val[...]) for key, val in self.file_results["settings"].iteritems()
        )

        self.quant_sys.update(
            (key, val[...]) for key, val in self.file_results["ehrenfest"].iteritems()
        )

        # Convert into object
        self.quant_sys = type('', (object,), self.quant_sys)()

        # final propagation time
        self.T_final = 8 * 2 * np.pi / self.quant_sys.omega

        # Load names of frames and sort them
        self.frame_names = map(int, self.file_results["frames"])
        self.frame_names.sort(reverse=True)

        # Save total number of frams
        self.num_frames = len(self.frame_names)

        # Create group where each frame will be saved
        self.frames_grp = self.file_results["frames"]

        # List to save times
        self.times = []

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        ax = fig.add_subplot(211)

        #ax.set_title(
        #    'Wigner function evolution $W(x,p,t)$\nwith $\\gamma^{-1} = $ %.2f and $kT = $ %.2f (a.u.)'
        #    % (1. / self.quant_sys._gamma, self.quant_sys.kT)
        #)
        ax.set_title("Wigner function")

        extent = [
            self.quant_sys.X_wigner.min(), self.quant_sys.X_wigner.max(),
            self.quant_sys.P_wigner.min(), self.quant_sys.P_wigner.max()
        ]

        # import utility to visualize the wigner function
        from wigner_normalize import WignerNormalize, WignerSymLogNorm

        # generate empty plot
        self.img = ax.imshow(
            [[]],
            extent=extent,
            origin='lower',
            aspect=5,
            cmap='bwr',
            #norm=WignerSymLogNorm(linthresh=1e-4, vmin=-0.3, vmax=0.3),
            norm=WignerNormalize(vmin=-0.0001, vmax=0.0001)
        )

        ax.set_xlim([-25, 25])
        ax.set_ylim([-3, 3])

        self.fig.colorbar(self.img)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(212)

        self.X = self.quant_sys.X_wigner.reshape(-1)
        self.X_rho = np.sqrt(2.) * self.X
        self.wigner_dP = self.quant_sys.P_wigner[1] - self.quant_sys.P_wigner[0]

        self.line1, = ax.semilogy([self.X_rho.min(), self.X_rho.max()], [1e-14, 1.], 'r', label="Rho")
        self.line2, = ax.semilogy([self.X_rho.min(), self.X_rho.max()], [1e-14, 1.], 'b-', label="W")
        #self.fig.legend(ax)
        #F = self.quant_sys.F
        #self.laser_filed_plot, = ax.plot([0., self.T_final], [-F, F])
        #ax.set_xlabel('time (a.u.)')
        #ax.set_ylabel('Laser field $E(t)$ (a.u.)')

    def empty_frame(self):
        """
        Make empty frame and reinitialize quantum system
        :param self:
        :return: image object
        """
        self.img.set_array([[]])
        #self.laser_filed_plot.set_data([], [])
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        return self.img, self.line1, self.line2 #self.laser_filed_plot

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        # Load frame to be visualzie
        frame_grp = self.file_results["frames/%d" % self.frame_names.pop()]

        # Get the Wigner function
        wigner = frame_grp["wigner"][...]
        self.img.set_array(wigner)

        wigner_x_marginal = wigner.sum(axis=0) * self.wigner_dP
        self.line2.set_data(self.X, wigner_x_marginal)

        prob = frame_grp["prob"][...]
        self.line1.set_data(self.X_rho, prob.real)
        #self.laser_filed_plot.set_data( self.quant_sys.X_wigner, prob.real)

        #print("Purity: 1 - %.1e" % (1 - self.quant_sys.get_purity()))

        #print("Frame : %d / %d" % (self.current_frame_num, self.num_frames))
        #self.current_frame_num += 1

        #self.times.append(self.quant_sys.t)

        #t = np.array(self.times)
        #self.laser_filed_plot.set_data(t, self.quant_sys.E(t))

        #self.fig.save_figure("result.png")

        return self.img, self.line1, self.line2 #self.laser_filed_plot

with h5py.File('../strong_field_physics_1024.hdf5', 'r') as file_results:
    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig, file_results)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=visualizer.num_frames, init_func=visualizer.empty_frame, blit=True, repeat=True
    )

    #plt.show()

    # Set up formatting for the movie files
    writer = matplotlib.animation.ImageMagickFileWriter()
    #writer = matplotlib.animation.writers['mencoder'](fps=20, metadata=dict(artist='Denys Bondar'), bitrate=-1)

    # Save animation into the file
    animation.save('wigner', writer=writer)

    # extract the reference to quantum system
    quant_sys = visualizer.quant_sys

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = dt * np.arange(len(quant_sys.X_average)) + dt

    plt.subplot(131)
    plt.title("Ehrenfest 1")
    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p + \\gamma x \\rangle$')

    plt.legend(loc='upper left')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("Ehrenfest 2")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(
        times, quant_sys.P_average_RHS, 'b--',
        label='$\\langle -\\partial V/\\partial x  + \\gamma p \\rangle$'
    )

    plt.legend(loc='upper left')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, quant_sys.hamiltonian_average)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()

    #################################################################
    #
    # Plot HHG spectra as FFT(<P>)
    #
    #################################################################

    N = len(quant_sys.P_average)

    # the windowed fft of the evolution
    # to remove the spectral leaking. For details see
    # rhttp://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    from scipy import fftpack
    from scipy.signal import blackman

    # obtain the dipole
    J = quant_sys.P_average
    fft_J = fftpack.fft(blackman(N) * J)
    spectrum = np.abs(fftpack.fftshift(fft_J))**2
    omegas = fftpack.fftshift(fftpack.fftfreq(N, quant_sys.dt/(2*np.pi))) / quant_sys.omega


    spectrum /= spectrum.max()

    plt.semilogy(omegas, spectrum)
    plt.ylabel('spectrum FFT($\\langle p \\rangle$)')
    plt.xlabel('frequency / $\\omega$')
    plt.xlim([0, 100.])
    plt.ylim([1e-20, 1.])

    plt.show()
