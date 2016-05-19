import pyfftw # A pythonic wrapper of FFTW
import numpy as np
from types import MethodType, FunctionType # this is used to dynamically add method to class


class WignerMoyalFTTW1D:
    """
    The second-order split-operator propagator for the Moyal equation for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t)
    (K and V may not depend on time.)
    This implementation stores the Wigner function as a 2D real array.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum
            V(self, x) - potential energy (as a function) may depend on time
            diff_V(self, x) (optional) -- the derivative of the potential energy for the Ehrenfest theorem calculations
            K(self, p) - momentum dependent part of the hamiltonian (as a function) may depend on time
            diff_K(self, p) (optional) -- the derivative of the kinetic energy for the Ehrenfest theorem calculations
            dt - time step
            t (optional) - initial value of time

            alpha (optional) - the absorbing boundary smoothing parameter

            FFTW settings (for details see https://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html#pyfftw.FFTW)
            ffw_flags (optional) - a list of strings and is a subset of the flags that FFTW allows for the planners
            fftw_threads (optional) - how many threads to use when invoking FFTW, with a default of 1
            fftw_wisdom - a tuple of strings returned by pyfftw.export_wisdom() for efficient simulations
        """
        # save all attributes
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert self.X_gridDIM % 2 == 0, "Coordinate grid size (X_gridDIM) must be even"

        try:
            self.P_gridDIM
        except AttributeError:
            raise AttributeError("Momentum grid size (P_gridDIM) was not specified")

        assert self.P_gridDIM % 2 == 0, "Momentum grid size (P_gridDIM) must be even"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.P_amplitude
        except AttributeError:
            raise AttributeError("Momentum grid range (P_amplitude) was not specified")

        try:
            # dynamically assign the method of calculating potential energy
            self.V = MethodType(self.V, self, self.__class__)
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            # dynamically assign the method of calculating kinetic energy
            self.K = MethodType(self.K, self, self.__class__)
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        ##########################################################################################
        #
        # Generating grids
        #
        ##########################################################################################

        # get coordinate and momentum step sizes
        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.dP = 2. * self.P_amplitude / self.P_gridDIM

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)
        self.X = self.X[np.newaxis, :]

        # Lambda grid (variable conjugate to the coordinate)
        self.Lambda = np.fft.fftfreq(self.X_gridDIM, self.dX / (2 * np.pi))

        # take only first half, as required by the real fft
        self.Lambda = self.Lambda[:(1 + self.X_gridDIM // 2)]
        #
        self.Lambda = self.Lambda[np.newaxis, :]

        # momentum grid
        self.P = np.linspace(-self.P_amplitude, self.P_amplitude - self.dP, self.P_gridDIM)
        self.P = self.P[:, np.newaxis]

        # Theta grid (variable conjugate to the momentum)
        self.Theta = np.fft.fftfreq(self.P_gridDIM, self.dP / (2 * np.pi))

        # take only first half, as required by the real fft
        self.Theta = self.Theta[:(1 + self.P_gridDIM // 2)]
        #
        self.Theta = self.Theta[:, np.newaxis]

        ##########################################################################################
        #
        # Pre-calculate absorbing boundary
        #
        ##########################################################################################

        # auxiliary grids
        X_minus = self.X - 0.5 * self.Theta
        X_plus = self.X + 0.5 * self.Theta

        try:
            self.alpha
            # if user specified  the absorbing boundary smoothing parameter (alpha)
            # then generate the absorbing boundary

            xmin = min(X_minus.min(), X_plus.min())
            xmax = max(X_minus.max(), X_plus.max())

            self.xtheta_extent = (X_minus.min(), X_minus.max(), X_plus.min(), X_plus.max())

            self.abs_boundary = np.sin(np.pi * (X_plus - xmin) / (xmax - xmin))
            self.abs_boundary *= np.sin(np.pi * (X_minus - xmin) / (xmax - xmin))

            np.abs(self.abs_boundary, out=self.abs_boundary)
            self.abs_boundary **= abs(self.alpha * self.dt)

        except AttributeError:
            # if the absorbing boundary smoothing parameter was not specified
            # then we should not use the absorbing boundary
            self.abs_boundary = 1

        ##########################################################################################
        #
        # Pre-calculate exponents
        #
        ##########################################################################################

        try:
            # Cache the potential energy exponent, if the potential is time independent
            self._expV = np.exp(-self.dt * 0.5j * (self.V(X_minus) - self.V(X_plus)))

            # Apply absorbing boundary
            self._expV *= self.abs_boundary

            # Dynamically assign the method self.get_exp_v(t) to access the cached exponential
            self.get_exp_v = MethodType(lambda self, t: self._expV, self, self.__class__)

        except TypeError:
            # If exception is generated, then the potential is time-dependent and caching is not possible,
            # thus, dynamically assign the method self.get_exp_v(t) to recalculate the exponent for every t

            self.X_minus = X_minus
            self.X_plus = X_plus

            def get_exp_v(self, t):
                result = -self.dt * 0.5j * (self.V(self.X_minus, t) - self.V(self.X_plus, t))
                np.exp(result, out=result)
                # Apply absorbing boundary
                result *= self.abs_boundary
                return result

            self.get_exp_v = MethodType(get_exp_v, self, self.__class__)

        ##########################################################################################

        try:
            # Cache the kinetic energy exponent, if the potential is time independent
            self._expK = np.exp(
                -self.dt * 1j * (self.K(self.P + 0.5 * self.Lambda) - self.K(self.P - 0.5 * self.Lambda))
            )

            # Dynamically assign the method self.get_exp_k(t) to access the cached exponential
            self.get_exp_k = MethodType(lambda self, t: self._expK, self, self.__class__)

        except TypeError:
            # If exception is generated, then the kinetic term is time-dependent and caching is not possible,
            # thus, dynamically assign the method self.get_exp_k(t) to recalculate the exponent for every t

            self.P_minus = self.P - 0.5 * self.Lambda
            self.P_plus = self.P + 0.5 * self.Lambda

            def get_exp_k(self, t):
                result = -self.dt * 1j * (self.K(self.P_plus, t) - self.K(self.P_minus, t))
                np.exp(result, out=result)
                return result

            self.get_exp_k = MethodType(get_exp_k, self, self.__class__)

        ##########################################################################################
        #
        #   Ehrenfest theorems (optional)
        #
        ##########################################################################################

        try:
            # Check whether the necessary terms are specified to calculate the Ehrenfest theorems

            # Pre-calculate RHS if time independent (using similar ideas as in self.get_exp_v above)
            try:
                self._diff_V = self.diff_V(self, self.X)
                self.get_diff_v = MethodType(lambda self, t: self._diff_V, self, self.__class__)
            except TypeError:
                self.get_diff_v = MethodType(
                    lambda self, t: self.diff_V(self, self.X, t),
                    self,
                    self.__class__
                )

            # Pre-calculate RHS if time independent (using similar ideas as in self.get_exp_v above)
            try:
                self._diff_K = self.diff_K(self, self.P)
                self.get_diff_k = MethodType(lambda self, t: self._diff_K, self, self.__class__)
            except TypeError:
                self.get_diff_k = MethodType(
                    lambda self, t: self.diff_K(self, self.P, t),
                    self,
                    self.__class__
                )

            # Pre-calculate the potential and kinetic energies for
            # calculating the expectation value of Hamiltonian
            try:
                self._V = self.V(self.X)
                self.get_v = MethodType(lambda self, t: self._V, self, self.__class__)
            except TypeError:
                self.get_v = MethodType(lambda self, t: self.V(self.X, t), self, self.__class__)

            try:
                self._K = self.K(self.P)
                self.get_k = MethodType(lambda self, t: self._K, self, self.__class__)
            except TypeError:
                self.get_k = MethodType(lambda self, t: self.K(self.P, t), self, self.__class__)

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

        ##########################################################################################
        #
        #   FTTW set-up
        #
        ##########################################################################################

        # Check for FFTW flags
        try:
            self.ffw_flags
        except AttributeError:
            # otherwise assign some default values
            self.ffw_flags = ('FFTW_ESTIMATE',)

        # Allow to destroy data in input arrays during FFT to speed up calculations
        self.ffw_flags = self.ffw_flags + ('FFTW_DESTROY_INPUT',)

        # Numer of threads used for FFTW
        try:
            self.fftw_threads
        except AttributeError:
            self.fftw_threads = 1

        # load FFTW wisdom, if provided
        try:
            pyfftw.import_wisdom(self.fftw_wisdom)
        except AttributeError:
            pass

        # allocate memory for the Wigner function
        self.wignerfunction = pyfftw.empty_aligned((self.P.size, self.X.size), dtype=np.float)

        # create a pointer to the wigner function in the theta x representation
        self.wigner_theta_x = pyfftw.empty_aligned((self.Theta.size, self.X.size), dtype=np.complex)

        # plan the FFT for the  p x -> theta x transform
        self.p2theta_transform = pyfftw.FFTW(
            self.wignerfunction, self.wigner_theta_x,
            axes=(0,),
            direction='FFTW_FORWARD',
            flags=self.ffw_flags,
            threads=self.fftw_threads
        )

        # plan the FFT for the theta x -> p x  transform
        self.theta2p_transform = pyfftw.FFTW(
            self.wigner_theta_x, self.wignerfunction,
            axes=(0,),
            direction='FFTW_BACKWARD',
            flags=self.ffw_flags,
            threads=self.fftw_threads
        )

        # create a pointer to the wigner function in the theta x representation
        self.wigner_p_lambda = pyfftw.empty_aligned((self.P.size, self.Lambda.size), dtype=np.complex)

        # plan the FFT for the p x  ->  p lambda transform
        self.x2lambda_transform = pyfftw.FFTW(
            self.wignerfunction, self.wigner_p_lambda,
            axes=(1,),
            direction='FFTW_FORWARD',
            flags=self.ffw_flags,
            threads=self.fftw_threads
        )

        # plan the FFT for the p lambda  ->  p x transform
        self.lambda2x_transform = pyfftw.FFTW(
            self.wigner_p_lambda, self.wignerfunction,
            axes=(1,),
            direction='FFTW_BACKWARD',
            flags=self.ffw_flags,
            threads=self.fftw_threads
        )

    def set_wignerfunction(self, new_wigner_func):
        """
        Set the initial Wigner function
        :param new_wigner_func: 2D numpy array contaning the wigner function
                            or function of X and P [F(x,p)] which will be evalued as F(self.X, self.P)
        :return: self
        """
        if isinstance(new_wigner_func, np.ndarray):
            # perform the consistency checks
            assert new_wigner_func.shape == (self.P.size, self.X.size), \
                "The grid sizes does not match with the Wigner function"

            assert new_wigner_func.dtype == np.float, "Supplied Wigner function must be real"

            # copy wigner function
            self.wignerfunction[:] = new_wigner_func

        elif isinstance(new_wigner_func, FunctionType):
            # user supplied the function which will return the Wigner function
            self.wignerfunction[:] = new_wigner_func(self, self.X, self.P)

        else:
            raise NotImplementedError("new_wigner_func must be either function or numpy.array")

        # normalize
        self.wignerfunction /= self.wignerfunction.sum() * self.dX * self.dP

        return self

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        expV = self.get_exp_v(self.t)

        self.p2theta_transform()
        self.wigner_theta_x *= expV
        self.theta2p_transform()

        self.x2lambda_transform()
        self.wigner_p_lambda *= self.get_exp_k(self.t)
        self.lambda2x_transform()

        self.p2theta_transform()
        self.wigner_theta_x *= expV
        self.theta2p_transform()

        # increment current time
        self.t += self.dt

        return self.wignerfunction

    def propagate(self, time_steps=1):
        """
        Time propagate the Wigner function saved in self.wignerfunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wignerfunction
        """
        # pre-compute the volume element in phase space
        dXdP = self.dX * self.dP

        for _ in xrange(time_steps):

            # advance by one time step
            self.single_step_propagation()

            # normalization
            self.wignerfunction /= self.wignerfunction.sum() * dXdP

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest(self.t)

        return self.wignerfunction

    def get_Ehrenfest(self, t):
        """
        Calculate observables entering the Ehrenfest theorems at time (t)
        """
        if self.isEhrenfest:
            # calculate the coordinate density
            density_coord = self.wignerfunction.real.sum(axis=0)
            # normalize
            density_coord /= density_coord.sum()

            # save the current value of <X>
            self.X_average.append(
                np.dot(density_coord, self.X.reshape(-1))
            )
            self.P_average_RHS.append(
                -np.dot(density_coord, self.get_diff_v(t).reshape(-1))
            )

            # calculate density in the momentum representation
            density_momentum = self.wignerfunction.real.sum(axis=1)
            # normalize
            density_momentum /= density_momentum.sum()

            # save the current value of <P>
            self.P_average.append(
                np.dot(density_momentum, self.P.reshape(-1))
            )
            self.X_average_RHS.append(
                np.dot(density_momentum, self.get_diff_k(t).reshape(-1))
            )

            # save the current expectation value of energy
            self.hamiltonian_average.append(
                np.dot(density_momentum, self.get_k(t).reshape(-1))
                +
                np.dot(density_coord, self.get_v(t).reshape(-1))
            )

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(WignerMoyalFTTW1D.__doc__)

    # load tools for creating animation
    import sys
    import matplotlib

    if sys.platform == 'darwin':
        # only for MacOS
        matplotlib.use('TKAgg')

    import matplotlib.animation
    import matplotlib.pyplot as plt

    class VisualizeDynamicsPhaseSpace:
        """
        Class to visualize the Wigner function function dynamics in phase space.
        """
        def __init__(self, fig):
            """
            Initialize all propagators and frame
            :param fig: matplotlib figure object
            """
            #  Initialize systems
            self.set_quantum_sys()

            #################################################################
            #
            # Initialize plotting facility
            #
            #################################################################

            self.fig = fig

            ax = fig.add_subplot(111)

            ax.set_title('Wigner function, $W(x,p,t)$')
            extent = [self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()]

            # import utility to visualize the wigner function
            from wigner_normalize import WignerNormalize

            # generate empty plot
            self.img = ax.imshow(
                [[]],
                extent=extent,
                origin='lower',
                cmap='seismic',
                norm=WignerNormalize(vmin=-0.01, vmax=0.1)
            )

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            # Create propagator
            self.quant_sys = WignerMoyalFTTW1D(
                t=0,
                dt=0.01,
                X_gridDIM=512,
                X_amplitude=10.,
                P_gridDIM=512,
                P_amplitude=10,

                # randomized parameter
                omega_square=np.random.uniform(0.5, 3),

                # parameter controlling the width of the initial wigner function
                sigma=np.random.uniform(0.5, 4.),

                # kinetic energy part of the hamiltonian
                K=lambda _, p: 0.5 * p ** 2,

                # potential energy part of the hamiltonian
                V=lambda self, x: 0.5 * self.omega_square * x ** 2,

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K=lambda _, p: p,
                diff_V=lambda self, x: self.omega_square * x
            )

            # set randomised initial condition
            self.quant_sys.set_wignerfunction(
                lambda self, x, p: np.exp(
                    # randomized position
                    -self.sigma * (x + np.random.uniform(-1., 1.)) ** 2
                    # randomized initial velocity
                    - (1. / self.sigma) * (p + np.random.uniform(-1., 1.)) ** 2
                )
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.set_quantum_sys()
            self.img.set_array([[]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the wigner function
            self.img.set_array(self.quant_sys.propagate(20))
            return self.img,


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=np.arange(100), init_func=visualizer.empty_frame, repeat=True, blit=True
    )

    plt.show()

    # Set up formatting for the movie files
    # writer = matplotlib.animation.writers['mencoder'](fps=5, metadata=dict(artist='Denys Bondar'))

    # Save animation into the file
    # animation.save('harmonic_oscilator.mp4', writer=writer)

    # extract the reference to quantum system
    quant_sys = visualizer.quant_sys

    # Analyze how well the energy was preserved
    h = np.array(quant_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %f percent" % ((1. - h.min() / h.max()) * 100)
    )

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = dt * np.arange(len(quant_sys.X_average)) + dt

    plt.subplot(131)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='$\\langle -\\partial \\partial V/\\partial x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, h)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()