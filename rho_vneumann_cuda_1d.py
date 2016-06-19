import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype
from pycuda.reduction import ReductionKernel
import numpy as np
from fractions import gcd
from types import MethodType, FunctionType
import cufft

import skcuda.linalg as cu_linalg
cu_linalg.init()


class RhoVNeumannCUDA1D:
    """
    The second-order split-operator propagator for the von Neumann equation for the denisty matrix rho(x,x',t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t) using CUDA.
    """
    def __init__(self, **kwargs):
        """
        The following parameters are to be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
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

            abs_boundary_p (optional) - a string of the C code specifying function of P,
                                    which will be applied to the density matrix at each propagation step
            abs_boundary_x (optional) - a string of the C code specifying function of X,
                                    which will be applied to the density matrix at each propagation step

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

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

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

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)

        # P grid (variable conjugate to the coordinate)
        self.P = np.fft.fftfreq(self.X_gridDIM, self.dX / (2 * np.pi))
        self.dP = self.P[1] - self.P[0]

        # Girds for the Wigner function that you gte via sdlf.get_wignerfunction()
        self.P_wigner = self.P / np.sqrt(2.)
        self.X_wigner = self.X / np.sqrt(2.)

        self.wigner_dxdp = (self.X_wigner[1] - self.X_wigner[0]) * (self.P_wigner[1] - self.P_wigner[0])

        self.P_wigner = np.fft.fftshift(self.P_wigner)[:, np.newaxis]
        self.X_wigner = self.X_wigner[np.newaxis, :]

        ##########################################################################################
        #
        # Save CUDA constants
        #
        ##########################################################################################

        kwargs.update(dX=self.dX, dP=self.dP)

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

        ##########################################################################################
        #
        # Absorbing boundaries in different representations
        #
        ##########################################################################################

        try:
            self.abs_boundary_x
        except AttributeError:
            self.abs_boundary_x = "1."

        try:
            self.abs_boundary_p
        except AttributeError:
            self.abs_boundary_p = "1."

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

        # If the X grid size is smaller or equal to the max number of CUDA threads
        # then use all self.X_gridDIM processors
        # otherwise number of processor to be used is the greatest common divisor of these two attributes
        size_x = self.X_gridDIM
        nproc = (size_x if size_x <= self.max_thread_block else gcd(size_x, self.max_thread_block))

        # CUDA block and grid for functions that act on the whole density matrix
        self.rho_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(size_x // nproc, self.X_gridDIM)
        )

        # CUDA block and grid for extracting the density matrix diagonal
        self.diag_mapper_params = dict(
            block=(nproc, 1, 1),
            grid=(size_x // nproc, 1)
        )

        ##########################################################################################
        #
        # Generate CUDA functions applying the exponents
        #
        ##########################################################################################

        # Append user defined functions
        try:
            self.cuda_consts += self.functions
        except AttributeError:
            pass

        print("\n================================ Compiling expK and expV ================================\n")

        expK_expV_compiled = SourceModule(
            self.expK_expV_cuda_source.format(
                cuda_consts=self.cuda_consts, K=self.K, V=self.V,
                abs_boundary_p=self.abs_boundary_p, abs_boundary_x=self.abs_boundary_x
            )
        )

        self.expK = expK_expV_compiled.get_function("expK")
        self.expV = expK_expV_compiled.get_function("expV")

        ##########################################################################################
        #
        # Set-up CUDA FFT
        #
        ##########################################################################################

        self.plan_Z2Z = cufft.PlanZ2Z((self.X.size, self.X.size))

        ##########################################################################################
        #
        # Allocate memory for the wigner function in the theta x and p lambda representations
        # by reusing the memory
        #
        ##########################################################################################

        # the density matrix (central object) in the coordinate representation
        self.rho = gpuarray.GPUArray((self.X.size, self.X.size), np.complex128)

        ##########################################################################################
        #
        #   Initialize facility for calculating expectation values of the curent density matrix
        #   see the implementation of self.get_average
        #
        ##########################################################################################

        # This array is used for expectation value calculation
        self._tmp = gpuarray.empty_like(self.rho)

        # hash table of cuda compiled functions that calculate an average of specified observable
        self._compiled_observable = dict()

        # Create the plan for FFT/iFFT
        self.plan_Z2Z_ax0 = cufft.Plan_Z2Z_2D_Axis0(self.rho.shape)
        self.plan_Z2Z_ax1 = cufft.Plan_Z2Z_2D_Axis1(self.rho.shape)

        ##########################################################################################
        #
        #   Ehrenfest theorems (optional)
        #
        ##########################################################################################

        try:
            # Check whether the necessary terms are specified to calculate the Ehrenfest theorems
            self.diff_K
            self.diff_V

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # List for saving phase space time integral
            self.wigner_time = []

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

        self.print_memory_info()

    @classmethod
    def print_memory_info(cls):
        """
        Print the CUDA memory info
        :return:
        """
        print(
            "\n\n\t\tGPU memory Total %.2f GB\n\t\tGPU memory Free %.2f GB\n" % \
            tuple(np.array(pycuda.driver.mem_get_info()) / 2. ** 30)
        )

    def x2p_transform(self):
        """
        The x -> p transform
        :return:
        """
        cufft.fft_Z2Z(self.rho, self.rho, self.plan_Z2Z)

    def p2x_transform(self):
        """
        The p -> x transform
        :return:
        """
        cufft.ifft_Z2Z(self.rho, self.rho, self.plan_Z2Z)

    def set_rho(self, new_rho):
        """
        Set the initial Wigner function
        :param new_rho: 2D numpy array, 2D GPU array contaning the density matrix,
                    a string of the C code specifying the initial condition in the coordinate representation,
                    a python function of the form F(self, X, X_prime), or a float number
                    Coordinate variables X and X_prime are declared.
        :return: self
        """
        if isinstance(new_rho, (np.ndarray, gpuarray.GPUArray)):
            # perform the consistency checks
            assert new_rho.shape == self.rho.shape, \
                "The grid sizes does not match with the density matrix"

            # copy the density matrix
            self.rho[:] = new_rho

        elif isinstance(new_rho, FunctionType):
            # user supplied the function which will return the density matrix
            self.rho[:] = new_rho(self, self.X[:,np.newaxis], self.X[np.newaxis,:])

        elif isinstance(new_rho, str):
            # user specified C code
            print("\n================================ Compiling init_rho ================================\n")
            SourceModule(
                self.init_rho_cuda_source.format(cuda_consts=self.cuda_consts, new_rho=new_rho)
            ).get_function("Kernel")(self.rho, **self.rho_mapper_params)

        elif isinstance(new_rho, float):
            # user specified a constant
            self.rho.fill(np.complex128(new_rho))
        else:
            raise NotImplementedError("new_rho must be either function or numpy.array")

        # normalize
        self.rho /= cu_linalg.trace(self.rho) * self.dX

        return self

    def single_step_propagation(self):
        """
        Perform a single step propagation. The final density matrix function is not normalized.
        :return: self.rho
        """
        self.expV(self.rho, self.t, **self.rho_mapper_params)

        self.x2p_transform()
        self.expK(self.rho, self.t, **self.rho_mapper_params)
        self.p2x_transform()

        self.expV(self.rho, self.t, **self.rho_mapper_params)

        return self.rho

    def propagate(self, steps=1):
        """
        Time propagate the density matrix saved in self.rho
        :param steps: number of self.dt time increments to make
        :return: self.rho
        """
        for _ in xrange(steps):
            # increment current time
            self.t += self.dt

            # advance by one time step
            self.single_step_propagation()

            # normalization
            self.rho /= cu_linalg.trace(self.rho) * self.dX

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest(self.t)

        return self.rho

    def get_Ehrenfest(self, t):
        """
        Calculate observables entering the Ehrenfest theorems at time
        :param t: current time
        :return: coordinate and momentum densities, if the Ehrenfest theorems were calculated; otherwise, return None
        """
        if self.isEhrenfest:
            # save the current value of <X>
            self.X_average.append(
                self.get_average(("X",))
            )

            # save the current value of <diff_K>
            self.X_average_RHS.append(
                self.get_average((None, self.diff_K))
            )

            # save the current value of <P>
            self.P_average.append(
                self.get_average((None, "P"))
            )

            # save the current value of <-diff_V>
            self.P_average_RHS.append(
                -self.get_average((self.diff_V,))
            )

            # save the current expectation value of energy
            self.hamiltonian_average.append(
                self.get_average((None, self.K)) + self.get_average((self.V,))
            )

            # save Wigner time
            self.wigner_current = self.get_wignerfunction()
            self.wigner_time.append(
                self.get_wigner_time(self.wigner_current, self.wigner_initial, self.t)
            )

    def get_observable(self, observable_str):
        """
        Return the compiled observable
        :param observable_str: (str)
        :return: float
        """
        # Compile the corresponding cuda functions, if it has not been done
        try:
            func = self._compiled_observable[observable_str]
        except KeyError:
            print("\n============================== Compiling [%s] ==============================\n" % observable_str)
            func = self._compiled_observable[observable_str] = SourceModule(
                self.apply_observable_cuda_source.format(cuda_consts=self.cuda_consts, observable=observable_str),
            ).get_function("Kernel")

        return func

    def get_average(self, observable):
        """
        Return the expectation value of an observable.
            observable = (coordinate obs, momentum obs, coordinate obs, momentum obs, ...)

        Example 1:
            To calculate Tr[rho F1(x) g1(p) F2(X)], we use observable = ("F1(x)", "g1(p)", "F2(X)")

        Example 2:
            To calculate Tr[rho g(p) F(X)], we use observable = (None, "g(p)", "F(X)")

        :param observable: tuple of strings
        :return: float
        """

        # Boolean flag indicated the representation
        is_x_observable = False

        # Make a copy of the density matrix
        gpuarray._memcpy_discontig(self._tmp, self.rho)

        for obs_str in observable:
            is_x_observable = not is_x_observable

            if obs_str:
                if is_x_observable:
                    # Apply observable in the coordinate representation
                    self.get_observable(obs_str)(self._tmp, self.t, **self.rho_mapper_params)
                else:
                    # Going to the momentum representation
                    cufft.fft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax1)

                    # Normalize
                    self._tmp /= self._tmp.shape[1]

                    # Apply observable in the momentum representation
                    self.get_observable(obs_str)(self._tmp, self.t, **self.rho_mapper_params)

                    # Going back to the coordinate representation
                    cufft.ifft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax1)

        return cu_linalg.trace(self._tmp).real * self.dX

    def get_purity(self):
        """
        Return the purity of the current density matrix Tr[rho**2]
        :return: float
        """
        # If kernel calculating the purity is not present, compile it
        try:
            purity_kernel = self._purity_kernel
        except AttributeError:
            purity_kernel = self._purity_kernel = ReductionKernel(
                np.float64, neutral="0", reduce_expr="a + b",
                map_expr="pow(abs(R[i]), 2)", arguments="const %s *R" % dtype_to_ctype(self.rho.dtype)
            )

        return purity_kernel(self.rho).get() * self.dX**2

    def get_sigma_x_sigma_p(self):
        """
        Return the product of standard deviation of coordinate and momentum,
        the LHS of the Heisenberg uncertainty principle:
            sigma_p * sigma_p >= 0.5
        :return: float
        """
        return np.sqrt(
            (self.get_average(("X * X",)) - self.get_average(("X",))**2) *
            (self.get_average((None, "P * P")) - self.get_average((None, "P"))**2)
        )

    def get_wignerfunction(self):
        """
        Return the Wigner function obtained via numercal transformation of the density matrix (self.rho)
        :return: gpuarray
        """
        # Check whether phase for shearing compiled
        try:
            self.phase_shearX
        except AttributeError:
            # Compile the functions
            wigner_util_compiled = SourceModule(
                self.wigner_util_cuda_source.format(cuda_consts=self.cuda_consts)
            )
            self.phase_shearX = wigner_util_compiled.get_function("phase_shearX")
            self.phase_shearY = wigner_util_compiled.get_function("phase_shearY")
            self.sign_flip = wigner_util_compiled.get_function("sign_flip")

        """
            ####################################################################
            #
            # The documented pice of code is to test the the numerical Wigner transform
            # is equivalent to the denisty matrix
            #
            ####################################################################

            self._tmp1 = gpuarray.empty_like(self._tmp)
            self.av_P_wigner = SourceModule(
                self.weighted_func_cuda_code.format(cuda_consts=self.cuda_consts, func="X * P")
            ).get_function("Kernel")

        rho_obs = 0.5 * (
            self.get_average(("X","P")) + self.get_average((None,"P", "X"))
        )
        """

        ####################################################################################
        #
        # Step 1: Perform the 45 degree rotation of the density matrix
        # using method from
        #   K. G. Larkin,  M. A. Oldfield, H. Klemm, Optics Communications, 139, 99 (1997)
        #   (http://www.sciencedirect.com/science/article/pii/S0030401897000977)
        #
        ####################################################################################

        # Shear X
        cufft.fft_Z2Z(self.rho, self._tmp, self.plan_Z2Z_ax1)
        self.phase_shearX(self._tmp, **self.rho_mapper_params)
        cufft.ifft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax1)
        self._tmp /= self._tmp.shape[1]

        # Shear Y
        cufft.fft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax0)
        self.phase_shearY(self._tmp, **self.rho_mapper_params)
        cufft.ifft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax0)
        self._tmp /= self._tmp.shape[0]

        # Shear X
        cufft.fft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax1)
        self.phase_shearX(self._tmp, **self.rho_mapper_params)
        cufft.ifft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax1)
        self._tmp /= self._tmp.shape[1]

        ####################################################################################
        #
        # Step 2: Perform the FFT over the roated matrix
        # using method from
        #   D. H. Bailey and P. N. Swarztrauber, SIAM J. Sci. Comput. 15, 1105 (1994)
        #   (http://epubs.siam.org/doi/abs/10.1137/0915067)
        #
        ####################################################################################

        self.sign_flip(self._tmp, **self.rho_mapper_params)
        cufft.ifft_Z2Z(self._tmp, self._tmp, self.plan_Z2Z_ax0)
        self.sign_flip(self._tmp, **self.rho_mapper_params)

        # normalize the wigner function
        self._tmp /= gpuarray.sum(self._tmp).get() * self.wigner_dxdp

        """
        ####################################################################
        #
        # The documented pice of code is to test the the numerical Wigner transform
        # is equivalent to the denisty matrix
        #
        ####################################################################
        self.av_P_wigner(self._tmp, self._tmp1, self.t, **self.rho_mapper_params)
        r = float(gpuarray.sum(self._tmp1).get().real * self.wigner_dxdp)
        print "diff between density matrix and wigner = %.3e (abs val of obs = %.3e)"% (r - rho_obs, r)
        """

        return self._tmp

    def get_wigner_time(self, wigner_current, wigner_init, t):
        """
        Calculate the integral:

            int_{H(x, p, t) > -Ip} [wigner_current(x,p) - wigner_init(x,p)] dxdp

        :param wigner_current: gpuarray containing current Wigner function
        :param wigner_init: gpuarray containing initial Wigner function
        :param t: current time
        :return: float
        """
        # If kernel calculating the wigner time is not present, compile it
        try:
            wigner_time_mapper = self._wigner_time_mapper
        except AttributeError:
            # Allocate memory to map
            self._tmp_wigner_time = gpuarray.empty(self.rho.shape, np.float64)

            wigner_time_mapper = self._wigner_time_mapper = SourceModule(
                self.wigner_time_mapper_cuda_code.format(
                    cuda_consts=self.cuda_consts, K=self.K, V=self.V
                ),
            ).get_function("Kernel")

        wigner_time_mapper(self._tmp_wigner_time, wigner_current, wigner_init, t, **self.rho_mapper_params)

        return gpuarray.sum(self._tmp_wigner_time).get() * self.wigner_dxdp

    expK_expV_cuda_source = """
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

    // Potential energy
    __device__ double V(double X, double t)
    {{
        return ({V});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the kinetic energy exponent
    // onto the density matrix in the momentum representation < P | rho | P_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expK(cuda_complex *rho, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        // fft shifting momentum
        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);
        const double P_prime = dP * ((i + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);

        const double phase = -dt * (K(P, t) - K(P_prime, t));

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) * ({abs_boundary_p});
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to define the action of the potential energy exponent
    // onto the density matrix in the coordinate representation < X | rho | X_prime >
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void expV(cuda_complex *rho, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double X_prime = dX * (i - 0.5 * X_gridDIM);

        const double phase = -0.5 * dt * (V(X, t) - V(X_prime, t));

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase)) * ({abs_boundary_x});
    }}
    """

    init_rho_cuda_source = """
    // CUDA code to initialize the densiy matrix in the coordinate representation
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double X_prime = dX * (i - 0.5 * X_gridDIM);

        rho[indexTotal] = ({new_rho});
    }}
    """

    apply_observable_cuda_source = """
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to apply the observable onto the density function
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void Kernel(cuda_complex *rho, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);
        const double X = dX * (j - 0.5 * X_gridDIM);

        rho[indexTotal] *= ({observable});
    }}
    """

    wigner_util_cuda_source = """
    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to apply phase factors to perform X and Y shearing
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void phase_shearX(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double P = dP * ((j + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);
        const double X_prime = dX * (i - 0.5 * X_gridDIM);

        // perform rotation by theta: const double a = tan(0.5 * theta);
        const double a = tan(M_PI / 8.);
        const double phase = -a * P * X_prime;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase));
    }}

    __global__ void phase_shearY(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM);
        const double P_prime = dP * ((i + X_gridDIM / 2) % X_gridDIM - 0.5 * X_gridDIM);

        // perform rotation by theta: const double b = -sin(theta);
        const double b = -sin(M_PI / 4.);
        const double phase = -b * P_prime * X;

        rho[indexTotal] *= cuda_complex(cos(phase), sin(phase));
    }}

    ////////////////////////////////////////////////////////////////////////////
    //
    // CUDA code to multiply with (-1)^i in order for FFT
    // to approximate the Fourier integral over theta
    //
    ////////////////////////////////////////////////////////////////////////////

    __global__ void sign_flip(cuda_complex *rho)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        // rho *= pow(-1, i)
        rho[indexTotal] *= 1 - 2 * int(i % 2);
    }}
    """

    weighted_func_cuda_code = """
    // CUDA code to calculate
    //      weighted = W(X, P, t) * func(X, P, t).
    // This is used in self.get_average
    // weighted.sum()*dX*dP is the average of func(X, P, t) over the Wigner function

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(const cuda_complex *W, cuda_complex *weighted, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM) / sqrt(2.);
        const double P = dP * (i - 0.5 * X_gridDIM) / sqrt(2.);

        weighted[indexTotal] = W[indexTotal] * ({func});
    }}
    """

    wigner_time_mapper_cuda_code = """
    // CUDA code to calculate
    //      out = Heaviside(H(x, p, t) > -Ip) * (wigner_current(x,p) - wigner_init(x,p))

    #include<pycuda-complex.hpp>
    #include<math.h>
    #define _USE_MATH_DEFINES

    typedef pycuda::complex<double> cuda_complex;

    {cuda_consts}

    __global__ void Kernel(double *out, const cuda_complex *wigner_current, const double *wigner_init, double t)
    {{
        const size_t i = blockIdx.y;
        const size_t j = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t indexTotal = j + i * X_gridDIM;

        const double X = dX * (j - 0.5 * X_gridDIM) / sqrt(2.);
        const double P = dP * (i - 0.5 * X_gridDIM) / sqrt(2.);

        const double hamiltonian = {K} + {V};

        // Heaviside function of H(x, p, t) > -Ip
        const double indicator_func = 0.5 * (1 + int(signbit(hamiltonian + Ip)));

        out[indexTotal] = indicator_func * (real(wigner_current[indexTotal]) - wigner_init[indexTotal]);
    }}
    """

##########################################################################################
#
# Example
#
##########################################################################################

if __name__ == '__main__':

    print(RhoVNeumannCUDA1D.__doc__)

    # load tools for creating animation
    import sys
    import matplotlib

    if sys.platform == 'darwin':
        # only for MacOS
        matplotlib.use('TKAgg')

    import matplotlib.animation
    import matplotlib.pyplot as plt

    #np.random.seed(3156)

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
            extent = [
                self.quant_sys.X_wigner.min(), self.quant_sys.X_wigner.max(),
                self.quant_sys.P_wigner.min(), self.quant_sys.P_wigner.max()
            ]

            # import utility to visualize the wigner function
            from wigner_normalize import WignerNormalize

            # generate empty plot
            self.img = ax.imshow(
                [[]],
                extent=extent,
                aspect=0.5,
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
            self.quant_sys = RhoVNeumannCUDA1D(
                t=0.,
                dt=0.01,
                X_gridDIM=512,
                X_amplitude=20.,

                # randomized parameter
                omega_square=np.random.uniform(2., 6.),

                # randomized parameters for initial condition
                sigma=np.random.uniform(2., 4.),
                p0=np.random.uniform(-1., 1.),
                x0=np.random.uniform(-1., 1.),

                # kinetic energy part of the hamiltonian
                K="0.5 * P * P",

                # potential energy part of the hamiltonian
                V="0.5 * omega_square * X * X",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="P",
                diff_V="omega_square * X"
            )

            # set randomised initial condition
            self.quant_sys.set_rho(
                "exp("
                "   -sigma *( pow(X - x0, 2) + pow(X_prime - x0, 2)) + cuda_complex(0., p0*(X - X_prime))"
                ")"
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
            self.img.set_array(self.quant_sys.get_wignerfunction().get().real)

            self.quant_sys.propagate(50)

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
