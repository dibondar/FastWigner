import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from types import MethodType, FunctionType
import cufft


class WignerMoyalCUDA1D:
    """
    The second-order split-operator propagator for the Moyal equation for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t) using CUDA.

    This implementation stores the Wigner function as a 2D real gpu array.

    This implementation is based on the algorithm described in
        R. Cabrera, D. I. Bondar, K. Jacobs, and H. A. Rabitz, Phys. Rev. A 92, 042122 (2015)
        (http://dx.doi.org/10.1103/PhysRevA.92.042122)
    """
    def __init__(self, **kwargs):
        """
        The following parameters are to be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum
            t (optional) - initial value of time (default t = 0)
            consts (optional) - a string of the C code declaring the constants
            functions (optional) -  a string of the C code declaring auxiliary functions
            V - a string of the C code specifying potential energy. Coordinate (X) and time (t) variables are declared.
            K - a string of the C code specifying kinetic energy. Momentum (P) and time (t) variables are declared.
            diff_V (optional) - a string of the C code specifying the potential energy derivative w.r.t. X
                                    for the Ehrenfest theorem calculations
            diff_K (optional) - a string of the C code specifying the kinetic energy derivative w.r.t. P
                                    for the Ehrenfest theorem calculations
            dt - time step
            alpha (optional) - the absorbing boundary smoothing parameter.
                                If not specified, absorbing boundary not used.
            max_thread_block (optional) - the maximum number of GPU processes to be used (default 512)
        """
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
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
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
            del kwargs['t']
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        self.t = np.float64(self.t)

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
        self.dLambda = self.Lambda[1] - self.Lambda[0]

        # take only first half, as required by the real fft
        self.Lambda = self.Lambda[:(1 + self.X_gridDIM // 2)]
        #
        self.Lambda = self.Lambda[np.newaxis, :]

        # momentum grid
        self.P = np.linspace(-self.P_amplitude, self.P_amplitude - self.dP, self.P_gridDIM)
        self.P = self.P[:, np.newaxis]

        # Theta grid (variable conjugate to the momentum)
        self.Theta = np.fft.fftfreq(self.P_gridDIM, self.dP / (2 * np.pi))
        self.dTheta = self.Theta[1] - self.Theta[0]

        # take only first half, as required by the real fft
        self.Theta = self.Theta[:(1 + self.P_gridDIM // 2)]
        #
        self.Theta = self.Theta[:, np.newaxis]

        # pre-compute the volume element in phase space
        self.dXdP = self.dX * self.dP

        ##########################################################################################
        #
        # Absorbing boundary
        #
        ##########################################################################################

        try:
            self.alpha
            # if user specified  the absorbing boundary smoothing parameter (alpha)
            # then generate the absorbing boundary

            # find the spatial bounda in the extendent coordinate
            self.abs_ampl = self.X.max() + 0.5 * self.Theta.max()
            kwargs.update(abs_ampl=self.abs_ampl)

            # specify absorbing boundary
            self.abs_boundary = "pow(abs(sin(M_PI_2 * (X - abs_ampl) / abs_ampl)), dt * alpha)"
        except AttributeError:
            # if the absorbing boundary smoothing parameter was not specified
            # then we should not use the absorbing boundary
            self.abs_boundary = "1"

        ##########################################################################################
        #
        # Save CUDA constants
        #
        ##########################################################################################

        kwargs.update(dX=self.dX, dP=self.dP, dLambda=self.dLambda, dTheta=self.dTheta)

        self.cuda_consts = ""

        # Convert real constants into CUDA code
        for name, value in kwargs.items():
            if isinstance(value, int):
                self.cuda_consts += "    const int %s = %d;\n" % (name, value)
            elif isinstance(value, float):
                self.cuda_consts += "    const double %s = %.15e;\n" % (name, value)

        # Append user defined constants, if specified
        try:
            self.cuda_consts += self.consts
        except AttributeError:
            pass

        # Append user defined functions, if specified
        try:
            self.cuda_consts += self.functions
        except AttributeError:
            pass

        ##########################################################################################
        #
        # Generate CUDA functions applying the exponents
        #
        ##########################################################################################

        print("\n================================ Compiling expK ================================\n")

        self.expK = SourceModule(
            self.expK_cuda_source.format(cuda_consts=self.cuda_consts, K=self.K),
        ).get_function("Kernel")

        print("\n================================ Compiling expV ================================\n")

        self.expV = SourceModule(
            self.expV_cuda_source.format(
                cuda_consts=self.cuda_consts,
                V=self.V,
                abs_boundary=self.abs_boundary
            )
        ).get_function("Kernel")

        ##########################################################################################
        #
        # Set-up CUDA FFT
        #
        ##########################################################################################

        self.plan_D2Z_Axes0 = cufft.Plan2DAxis0((self.P.size, self.X.size), cufft.CUFFT_D2Z)
        self.plan_Z2D_Axes0 = cufft.Plan2DAxis0((self.P.size, self.X.size), cufft.CUFFT_Z2D)
        self.plan_D2Z_Axes1 = cufft.Plan2DAxis1((self.P.size, self.X.size), cufft.CUFFT_D2Z)
        self.plan_Z2D_Axes1 = cufft.Plan2DAxis1((self.P.size, self.X.size), cufft.CUFFT_Z2D)

        ##########################################################################################
        #
        #  Allocate memory for the Wigner functions and marginals
        #
        ##########################################################################################

        self.wignerfunction = gpuarray.GPUArray((self.P.size, self.X.size), np.float64)
        self.wigner_theta_x = gpuarray.GPUArray((self.Theta.size, self.X.size), np.complex128)
        self.wigner_p_lambda = gpuarray.GPUArray((self.P.size, self.Lambda.size), np.complex128)

        ##########################################################################################
        #
        #   Ehrenfest theorems (optional)
        #
        ##########################################################################################

        try:
            # Check whether the necessary terms are specified to calculate the Ehrenfest theorems
            self.diff_K
            self.diff_V

            print("\n================================ Compiling Ehrenfest functions ================================\n")

            self.get_p = SourceModule(
                self.weighted_func_cuda_code.format(cuda_consts=self.cuda_consts, func="P"),
            ).get_function("Kernel")

            self.get_diff_K = SourceModule(
                self.weighted_func_cuda_code.format(cuda_consts=self.cuda_consts, func=self.diff_K),
            ).get_function("Kernel")

            self.get_x = SourceModule(
                self.weighted_func_cuda_code.format(cuda_consts=self.cuda_consts, func="X"),
            ).get_function("Kernel")

            self.get_diff_V = SourceModule(
                self.weighted_func_cuda_code.format(cuda_consts=self.cuda_consts, func=self.diff_V),
            ).get_function("Kernel")

            hamiltonian_str = self.K + " + " + self.V
            self.get_hamiltonian = SourceModule(
                self.weighted_func_cuda_code.format(cuda_consts=self.cuda_consts, func=hamiltonian_str),
            ).get_function("Kernel")

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # This array is used for Ehrenfest theorem calculations
            self.weighted = gpuarray.empty_like(self.wignerfunction)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

        ##########################################################################################
        #
        #   Define block and grid parameters for CUDA kernel
        #
        ##########################################################################################

        #  Make sure that self.max_thread_block is defined
        # i.e., the maximum number of GPU processes to be used (default 512)
        try:
            self.max_thread_block
        except AttributeError:
            self.max_thread_block = 512

        if self.X_gridDIM <= self.max_thread_block:
            # If the X grid size is smaller or equal to the max numer of CUDA threads
            # then use all self.X_gridDIM processors
            nproc = self.X_gridDIM
        else:
            # otherwise number of processor to be used is the greatest common divisor of these two attributes
            from fractions import gcd
            nproc = gcd(self.X_gridDIM, self.max_thread_block)

        if nproc == 1:
            print("Warning: Parallelization is not possible given the specified values of "
                  "the parameters self.X_gridDIM and self.max_thread_block")

        # CUDA block and grid for functions that act on the whole Wigner function
        self.wigner_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(self.X_gridDIM // nproc, self.P_gridDIM)
        )

        print self.wigner_mapper_params

        # CUDA block and grid for function self.expV
        self.expV_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(self.X_gridDIM // nproc, self.P_gridDIM // 2 + 1)
        )

        print self.expV_mapper_params

        # CUDA block and grid for function self.expK
        self.expK_mapper_params = dict(
            block=(self.X_gridDIM // 2 + 1, 1, 1),
            grid=(1, self.P_gridDIM)
        )

        print self.expK_mapper_params

        print '         GPU memory Total       ', pycuda.driver.mem_get_info()[1] / float(2 ** 30), 'GB'
        print '         GPU memory Free        ', pycuda.driver.mem_get_info()[0] / float(2 ** 30), 'GB'

    def p2theta_transform(self):
        """
        the  p x -> theta x transform
        """
        cufft.cu_fft_D2Z(self.wignerfunction, self.wigner_theta_x, self.plan_D2Z_Axes0)

    def theta2p_transform(self):
        """
        the theta x -> p x  transform
        """
        cufft.cu_ifft_Z2D(self.wigner_theta_x, self.wignerfunction, self.plan_Z2D_Axes0)

    def x2lambda_transform(self):
        """
        the p x  ->  p lambda transform
        """
        cufft.cu_fft_D2Z(self.wignerfunction, self.wigner_p_lambda, self.plan_D2Z_Axes1)

    def lambda2x_transform(self):
        """
        the p lambda  ->  p x transform
        """
        cufft.cu_ifft_Z2D(self.wigner_p_lambda, self.wignerfunction, self.plan_Z2D_Axes1)

    def set_wignerfunction(self, new_wigner_func):
        """
        Set the initial Wigner function
        :param new_wigner_func: 2D numpy array, 2D GPU array contaning the wigner function,
                    a string of the C code specifying the initial condition,
                    a python function of the form F(self, x, p), or a float number
                    Coordinate (X) and momentum (P) variables are declared.
        :return: self
        """
        if isinstance(new_wigner_func, (np.ndarray, gpuarray.GPUArray)):
            # perform the consistency checks
            assert new_wigner_func.shape == (self.P.size, self.X.size), \
                "The grid sizes does not match with the Wigner function"

            assert new_wigner_func.dtype == np.float, "Supplied Wigner function must be real"

            # copy wigner function
            self.wignerfunction[:] = new_wigner_func

        elif isinstance(new_wigner_func, FunctionType):
            # user supplied the function which will return the Wigner function
            self.wignerfunction[:] = new_wigner_func(self, self.X, self.P)

        elif isinstance(new_wigner_func, str):
            # user specified C code
            print("\n================================ Compiling init_wigner ================================\n")
            init_wigner = SourceModule(
                self.init_wigner_cuda_source.format(cuda_consts=self.cuda_consts, new_wigner_func=new_wigner_func),
            ).get_function("Kernel")
            init_wigner(self.wignerfunction, **self.wigner_mapper_params)

        elif isinstance(new_wigner_func, float):
            # user specified a constant
            self.wignerfunction[:] = new_wigner_func
        else:
            raise NotImplementedError("new_wigner_func must be either function or numpy.array")

        # normalize
        self.wignerfunction /= gpuarray.sum(self.wignerfunction).get() * self.dX * self.dP

        return self

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        self.p2theta_transform()
        self.expV(self.wigner_theta_x, self.t, **self.expV_mapper_params)
        self.theta2p_transform()

        self.x2lambda_transform()
        self.expK(self.wigner_p_lambda, self.t, **self.expK_mapper_params)
        self.lambda2x_transform()

        self.p2theta_transform()
        self.expV(self.wigner_theta_x, self.t, **self.expV_mapper_params)
        self.theta2p_transform()

        return self.wignerfunction

    def propagate(self, steps=1):
        """
        Time propagate the Wigner function saved in self.wignerfunction
        :param steps: number of self.dt time increments to make
        :return: self.wignerfunction
        """
        for _ in xrange(steps):
            # increment current time
            self.t += self.dt

            # advance by one time step
            self.single_step_propagation()

            # normalization
            self.wignerfunction /= gpuarray.sum(self.wignerfunction).get() * self.dXdP

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest(self.t)

        return self.wignerfunction

    def get_Ehrenfest(self, t):
        """
        Calculate observables entering the Ehrenfest theorems at time
        :param t: current time
        :return: coordinate and momentum densities, if the Ehrenfest theorems were calculated; otherwise, return None
        """
        if self.isEhrenfest:
            # save the current value of <X>
            self.get_x(self.wignerfunction, self.weighted, t, **self.wigner_mapper_params)
            self.X_average.append(
                gpuarray.sum(self.weighted).get() * self.dXdP
            )

            # save the current value of <diff_K>
            self.get_diff_K(
                self.wignerfunction, self.weighted, t,
                block=(self.X_gridDIM, 1, 1), grid=(1, self.P_gridDIM)
            )
            self.X_average_RHS.append(
                gpuarray.sum(self.weighted).get() * self.dXdP
            )

            # save the current value of <P>
            self.get_p(self.wignerfunction, self.weighted, t, **self.wigner_mapper_params)
            self.P_average.append(
                gpuarray.sum(self.weighted).get() * self.dXdP
            )

            # save the current value of <-diff_V>
            self.get_diff_V(self.wignerfunction, self.weighted, t, **self.wigner_mapper_params)
            self.P_average_RHS.append(
                -gpuarray.sum(self.weighted).get() * self.dXdP
            )

            # save the current expectation value of energy
            self.get_hamiltonian(self.wignerfunction, self.weighted, t, **self.wigner_mapper_params)
            self.hamiltonian_average.append(
                gpuarray.sum(self.weighted).get() * self.dXdP
            )

    expK_cuda_source = """
    // CUDA code to define the action of the kinetic energy exponent
    // onto the wigner function in P Lambda representation

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    // Kinetic energy
    __device__ double K(double P, double t)
    {{
        return ({K});
    }}

    __global__ void Kernel(cuda_complex *Z, double t)
    {{
        const size_t j = threadIdx.x;
        const size_t i = blockIdx.y;
        const size_t indexTotal = j + i * blockDim.x;

        const double Lambda = dLambda*j;
        const double P = dP * (i - 0.5 * P_gridDIM);

        const double phase = -dt*(
            K(P + 0.5 * Lambda, t) - K(P - 0.5 * Lambda, t)
        );

        Z[indexTotal] *= cuda_complex(cos(phase), sin(phase));
    }}
    """

    expV_cuda_source = """
    // CUDA code to define the action of the potential energy exponent
    // onto the wigner function in Theta X representation

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    // Absorbing boundary condition
    __device__ double abs_boundary(double X)
    {{
        return ({abs_boundary});
    }}

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    __global__ void Kernel(cuda_complex *Z, double t)
    {{
        const size_t j = threadIdx.x;
        const size_t i = blockIdx.y;
        const size_t indexTotal = j + i * blockDim.x;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double Theta = dTheta*i;

        const double X_minus = X - 0.5 * Theta;
        const double X_plus = X + 0.5 * Theta;

        const double phase = -0.5 * dt * (V(X_minus, t) - V(X_plus, t));

        Z[indexTotal] *= cuda_complex(cos(phase), sin(phase))
                        * abs_boundary(X_minus) * abs_boundary(X_plus);
    }}
    """

    init_wigner_cuda_source = """
    // CUDA code to initialize the wigner function

    #include<math.h>
    #define _USE_MATH_DEFINES

    {cuda_consts}

    __global__ void Kernel(double *W)
    {{
        const size_t j = threadIdx.x;
        const size_t i = blockIdx.y;
        const size_t indexTotal = j + i * blockDim.x;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double P = dP * (i - 0.5 * P_gridDIM);

        W[indexTotal] = ({new_wigner_func});
    }}
    """

    weighted_func_cuda_code = """
    // CUDA code to calculate
    //      weighted = W(X, P, t) * func(X, P, t).
    // This is used in the Ehrenfest theorems verification since
    // weighted.sum()*dX*dP is the average of func(X, P, t) over the Wigner function

    #include<math.h>
    #define _USE_MATH_DEFINES

    {cuda_consts}

    __global__ void Kernel(const double *W, double *weighted, double t)
    {{
        const size_t j = threadIdx.x;
        const size_t i = blockIdx.y;
        const size_t indexTotal = j + i * blockDim.x;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double P = dP * (i - 0.5 * P_gridDIM);

        weighted[indexTotal] = W[indexTotal] * ({func});
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(WignerMoyalCUDA1D.__doc__)

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
            self.quant_sys = WignerMoyalCUDA1D(
                t=0.,
                dt=0.01,
                max_thread_block=512,
                X_gridDIM=1024,
                X_amplitude=10.,
                P_gridDIM=512,
                P_amplitude=10.,

                # randomized parameter
                omega_square=np.random.uniform(2., 6.),

                # randomized parameters for initial condition
                sigma=np.random.uniform(0.5, 4.),
                p0=np.random.uniform(-1., 1.),
                x0=np.random.uniform(-1., 1.),

                # smoothing parameter for absorbing boundary
                #alpha=0.01,

                # kinetic energy part of the hamiltonian
                K="0.5 * P * P",

                # potential energy part of the hamiltonian
                V="0.5 * omega_square * X * X",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",
                diff_V="omega_square * X"
            )

            # set randomised initial condition
            self.quant_sys.set_wignerfunction(
                "exp(-sigma * pow(X - x0, 2) - (1.0 / sigma) * pow(P - p0, 2))"
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.img.set_array([[]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the wigner function
            self.img.set_array(self.quant_sys.propagate(20).get())
            return self.img,


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = matplotlib.animation.FuncAnimation(
        fig, visualizer, frames=np.arange(100), init_func=visualizer.empty_frame, repeat=True, blit=True
    )

    plt.show()

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
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='$\\langle -\\partial V/\\partial x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, h)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()