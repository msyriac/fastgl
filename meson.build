project('fastgl',
        ['cpp'],
        default_options : ['cpp_std=c++17'])

py = import('python').find_installation(pure: false)
pybind = dependency('pybind11', version : '>=2.11')

inc    = include_directories('src')

sources = files(
  'src/fastgl.cpp',          # original FastGL implementation
  'src/fastgl_bindings.cpp', # pybind11 glue
)

# --- find OpenMP ----------------
omp_dep = dependency('openmp')

# Enable OpenMP for C++ manually
add_project_arguments('-fopenmp', language: 'cpp')
add_project_link_arguments('-fopenmp', language: 'cpp')


extension = py.extension_module(
  '_fastgl',
  sources             : sources,
  dependencies        : [pybind, omp_dep],
  include_directories : inc,
  subdir              : 'fastgl',      
  install             : true,
)

# Install the pure-Python package tree itself
install_subdir('fastgl', install_dir : py.get_install_dir())
