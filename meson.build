project(
  'fastgl',
  'cpp',
  # Keep in sync with the static version in pyproject.toml (which meson-python
  # uses for wheel naming); this fixes "Project version: undefined" in logs.
  version: '0.1.12',
  default_options: ['cpp_std=c++17'],
  meson_version: '>=1.1.0',
)

py = import('python').find_installation(pure: false)
cpp = meson.get_compiler('cpp')

pybind = dependency('pybind11', version: '>=2.11')

inc = include_directories('src')

sources = files(
  'src/fastgl.cpp',          # original FastGL implementation
  'src/fastgl_bindings.cpp', # pybind11 glue
)

# --- OpenMP -------------------------------------------------------------------
# Requirement: every binary must be OpenMP-threaded. Both branches below are
# hard-required, so configuration FAILS if OpenMP cannot be enabled -- this can
# never silently produce a serial build.
if host_machine.system() == 'darwin' and not cpp.has_argument('-fopenmp')
  # Apple clang: rejects bare -fopenmp ("clang: error: unsupported option").
  # Enable the pragmas in the frontend and link libomp explicitly.
  # Prefer an explicitly provided runtime: CI sets LIBOMP_PREFIX to a
  # conda-forge llvm-openmp whose minimum macOS is old enough for delocate to
  # vendor it into wheels tagged MACOSX_DEPLOYMENT_TARGET=13.0. (Homebrew's
  # bottle requires the build machine's own macOS version, which delocate
  # rejects.) Fall back to Homebrew for local development builds.
  libomp_prefix = run_command(
    py, '-c', 'import os; print(os.environ.get("LIBOMP_PREFIX", ""))',
    check: true,
  ).stdout().strip()

  if libomp_prefix == ''
    brew = find_program('brew', required: false)
    if not brew.found()
      error('OpenMP is required. Set LIBOMP_PREFIX to an OpenMP install prefix, or: brew install libomp')
    endif
    libomp_prefix = run_command(brew, '--prefix', 'libomp', check: true).stdout().strip()
  endif

  omp_dep = declare_dependency(
    compile_args: ['-Xpreprocessor', '-fopenmp', '-I' + libomp_prefix / 'include'],
    dependencies: cpp.find_library('omp', dirs: libomp_prefix / 'lib'),
  )
else
  # GCC / LLVM clang / MSVC: meson's builtin detection supplies the correct
  # compile AND link flags for this compiler, so no hand-rolled -fopenmp.
  omp_dep = dependency('openmp')
endif

extension = py.extension_module(
  '_fastgl',
  sources: sources,
  dependencies: [pybind, omp_dep],
  include_directories: inc,
  subdir: 'fastgl',
  install: true,
)

# Install the pure-Python package tree itself
install_subdir(
  'fastgl',
  install_dir: py.get_install_dir(),
  exclude_directories: ['__pycache__'],
)
