Name:           numerics
Version:        1.2.0
Release:        1%{?dist}
Summary:        Header-only C++20 library for scientific computing
License:        MIT
URL:            https://github.com/AdityaDendukuri/numerics
Source0:        %{url}/archive/v%{version}/%{name}-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  cmake >= 3.20
BuildRequires:  gcc-c++
BuildRequires:  make

%description
numerics is a header-only C++20 library for scientific computing.

%package        devel
Summary:        Header-only C++20 library for scientific computing

%description    devel
Headers and the CMake package for numerics: find_package(numerics) and
target_link_libraries(app PRIVATE numerics::numerics).

%prep
%autosetup

%build
%cmake -DNUMERICS_BUILD_EXAMPLES=OFF \
    -DNUMERICS_ENABLE_CUDA=OFF -DNUMERICS_ENABLE_MPI=OFF \
    -DNUMERICS_USE_BLAS=OFF -DNUMERICS_USE_LAPACK=OFF -DNUMERICS_USE_OPENMP=OFF
   

%install
%cmake_install

%files devel
%license LICENSE
%{_includedir}/numerics/
%{_datadir}/cmake/numerics/

%changelog
* Wed Nov 25 2026 Aditya Dendukuri <dendukuri.aditya123@outlook.com> - 1.2.0-1
- Header-only
