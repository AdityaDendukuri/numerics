# The package ships a static archive and headers only: no debuginfo to extract, and an
# LTO-only archive would not link with other compilers.
%global debug_package %{nil}
%global _lto_cflags %{nil}

Name:           numerics
Version:        1.1.0
Release:        1%{?dist}
Summary:        C++20 library for scientific computing
License:        MIT
URL:            https://github.com/AdityaDendukuri/numerics
Source0:        %{url}/archive/v%{version}/%{name}-%{version}.tar.gz

BuildRequires:  cmake >= 3.20
BuildRequires:  gcc-c++
BuildRequires:  make

%description
numerics is a C++20 library for scientific computing: dense and sparse linear algebra,
iterative solvers, eigenvalues, ODE and PDE integrators, FFTs, quadrature, root finding
and stochastic methods.

%package        devel
Summary:        Development files for numerics
Provides:       %{name}-static = %{version}-%{release}

%description    devel
Headers, the static library and the CMake package for numerics. Use it with
find_package(numerics) and target_link_libraries(app PRIVATE numerics::numerics).

%prep
%autosetup

%build
%cmake \
    -DBUILD_SHARED_LIBS=OFF \
    -DNUMERICS_BUILD_EXAMPLES=OFF \
    -DNUMERICS_BUILD_TESTS=OFF \
    -DNUMERICS_BUILD_IO=OFF \
    -DNUMERICS_ENABLE_CUDA=OFF \
    -DNUMERICS_ENABLE_MPI=OFF \
    -DNUMERICS_USE_BLAS=OFF \
    -DNUMERICS_USE_LAPACK=OFF \
    -DNUMERICS_USE_OPENMP=OFF \
    -DNUMERICS_USE_FFTW=OFF \
    -DNUMERICS_USE_SUITESPARSE=OFF
%cmake_build

%install
%cmake_install

%files devel
%license LICENSE
%doc README.md
%{_includedir}/numerics/
%{_libdir}/libnumerics.a
%{_libdir}/cmake/numerics/

%changelog
* Wed Nov 25 2026 Aditya Dendukuri <dendukuri.aditya123@outlook.com> - 1.1.0-1
- Initial package
