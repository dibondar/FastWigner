import pyfftw # A pythonic wrapper of FFTW
import numpy as np
from types import MethodType # this is used to dynamically add method to class

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
            fftw_kwargs (optional) - dictionary listing arguments of pyfftw.FFTW. For further details see
                https://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html#pyfftw.FFTW
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
        # Pre-calculate exponents
        #
        ##########################################################################################

        try:
            # Cache the potential energy exponent, if the potential is time independent
            self._expV = np.exp(
                -self.dt * 0.5j * (self.V(self.X - 0.5*self.Theta) - self.V(self.X + 0.5*self.Theta))
            ) / self.P.size

            # Dynamically assign the method self.get_exp_v(t) to access the cached exponential
            self.get_exp_v = MethodType(lambda self, t: self._expV, self, self.__class__)

        except TypeError:
            # If exception is generated, then the potential is time-dependent and caching is not possible,
            # thus, dynamically assign the method self.get_exp_v(t) to recalculate the exponent for every t
            def get_exp_v(self, t):
                result = -self.dt * 0.5j * (
                    self.V(self.X - 0.5 * self.Theta, t) - self.V(self.X + 0.5 * self.Theta, t)
                )
                np.exp(result, out=result)
                result /= self.P.size
                return result

            self.get_exp_v = MethodType(get_exp_v, self, self.__class__)

        ##########################################################################################

        try:
            # Cache the kinetic energy exponent, if the potential is time independent
            self._expK = np.exp(
                -self.dt * 1j * (self.K(self.P + 0.5 * self.Lambda) - self.K(self.P - 0.5 * self.Lambda))
            ) / self.X.size

            # Dynamically assign the method self.get_exp_k(t) to access the cached exponential
            self.get_exp_k = MethodType(lambda self, t: self._expK, self, self.__class__)

        except TypeError:
            # If exception is generated, then the kinetic term is time-dependent and caching is not possible,
            # thus, dynamically assign the method self.get_exp_k(t) to recalculate the exponent for every t
            def get_exp_k(self, t):
                result = -self.dt * 1j * (
                    self.K(self.P + 0.5 * self.Lambda, t) - self.K(self.P - 0.5 * self.Lambda, t)
                )
                np.exp(result, out=result)
                result /= self.X.size
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

        try:
            # Check whether the user specified fftw settings
            self.fftw_kwargs
        except AttributeError:
            # otherwise assign some default values
            self.fftw_kwargs = dict(flags=('FFTW_ESTIMATE',))

        # Allow to destroy data in input arrays during FFT
        self.fftw_kwargs['flags'] = self.fftw_kwargs['flags'] + ('FFTW_DESTROY_INPUT',)

        # allocate memory for the Wigner function
        self.wignerfunction = pyfftw.empty_aligned((self.P.size, self.X.size), dtype=np.float)

        # create a pointer to the wigner function in the theta x representation
        self.wigner_theta_x = pyfftw.empty_aligned((self.Theta.size, self.X.size), dtype=np.complex)

        # plan the FFT for the  p x -> theta x transform
        self.p2theta_transform = pyfftw.FFTW(
            self.wignerfunction, self.wigner_theta_x,
            axes=(0,),
            direction='FFTW_FORWARD',
            **self.fftw_kwargs
        )

        # plan the FFT for the theta x -> p x  transform
        self.theta2p_transform = pyfftw.FFTW(
            self.wigner_theta_x, self.wignerfunction,
            axes=(0,),
            direction='FFTW_BACKWARD',
            **self.fftw_kwargs
        )

        # create a pointer to the wigner function in the theta x representation
        self.wigner_p_lambda = pyfftw.empty_aligned((self.P.size, self.Lambda.size), dtype=np.complex)

        # plan the FFT for the p x  ->  p lambda transform
        self.x2lambda_transform = pyfftw.FFTW(
            self.wignerfunction, self.wigner_p_lambda,
            axes=(1,),
            direction='FFTW_FORWARD',
            **self.fftw_kwargs
        )

        # plan the FFT for the p lambda  ->  p x transform
        self.lambda2x_transform = pyfftw.FFTW(
            self.wigner_p_lambda, self.wignerfunction,
            axes=(1,),
            direction='FFTW_BACKWARD',
            **self.fftw_kwargs
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

        else:
            # user supplied the function which will return the Wigner function
            self.wignerfunction[:] = new_wigner_func(self.X, self.P)

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
        pass
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
                -np.dot(density_coord, self.get_diff_V(t).reshape(-1))
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
                np.dot(density_momentum, self.get_diff_K(t).reshape(-1))
            )

            # save the current expectation value of energy
            self.hamiltonian_average.append(
                np.dot(density_momentum, self.get_K(t).reshape(-1))
                +
                np.dot(density_coord, self.get_V(t).reshape(-1))
            )
        """
##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    quant_sys = WignerMoyalFTTW1D(
        t=0,
        dt=0.05,
        X_gridDIM=256,
        X_amplitude=8.,
        P_gridDIM=256,
        P_amplitude=7.,

        # some parameter
        omega_square = 1.,

        # kinetic energy part of the hamiltonian
        K=lambda _, p: 0.5 * p ** 2,

        # potential energy part of the hamiltonian
        V=lambda self, x: 0.5 * self.omega_square * x ** 2,

        # these functions are used for evaluating the Ehrenfest theorems
        diff_K=lambda _, p: p,
        diff_V=lambda self, x: self.omega_square * x
    )

    quant_sys.set_wignerfunction(
        lambda x, p: np.exp(-(x-1.)**2 -p**2)
    )

    plt.subplot(121)

    plt.imshow(quant_sys.wignerfunction.copy())

    plt.subplot(122)
    plt.imshow(quant_sys.propagate(1000))

    plt.show()