EnsureSConsVersion(1,2)

import os

import inspect
import platform
import re
import subprocess
from SCons import SConf

def haveOcelot():
	try:
		which('OcelotConfig')
		
		return True
	except:
		return False

def getOcelotPaths():
	"""Determines Ocelot {bin,lib,include,cflags,lflags,libs} paths
	
	returns (bin_path,lib_path,inc_path,cflags,lflags,libs)
	"""
	
	try:
		ocelot_config_path = which('OcelotConfig')
	except:
		raise ValueError, 'Error: Failed to find OcelotConfig, make sure ' + \
			'it is on your PATH'
	
	# determine defaults
	bin_path = os.popen('OcelotConfig --bindir').read().split()
	lib_path = os.popen('OcelotConfig --libdir').read().split()
	inc_path = os.popen('OcelotConfig --includedir').read().split()
	cflags   = os.popen('OcelotConfig --cppflags').read().split()
	lflags   = os.popen('OcelotConfig --ldflags').read().split()
	libs     = os.popen('OcelotConfig --libs').read().split()
	
	return (bin_path,lib_path,inc_path,cflags,lflags,libs)

	
def getTools():
	result = []
	if os.name == 'nt':
		result = ['default', 'msvc']
	elif os.name == 'posix':
		result = ['default', 'c++', 'g++']
	else:
		result = ['default']

	return result;


OldEnvironment = Environment;


# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
		'gcc' : {'warn_all' : '-Wall',
			'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g', 
			'exception_handling' : '', 'standard': ''},
		'g++' : {'warn_all' : '-Wall',
			'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g', 
			'exception_handling' : '', 'standard': '-std=c++0x'},
		'c++' : {'warn_all' : '-Wall',
			'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g',
			'exception_handling' : '',
			'standard': ['-stdlib=libc++', '-std=c++0x', '-pthread']},
		'cl'  : {'warn_all' : '/Wall',
				 'warn_errors' : '/WX', 
		         'optimization' : ['/Ox', '/MD', '/Zi', '/DNDEBUG'], 
				 'debug' : ['/Zi', '/Od', '/D_DEBUG', '/RTC1', '/MDd'], 
				 'exception_handling': '/EHsc', 
				 'standard': ['/GS', '/GR', '/Gd', '/fp:precise',
				 	'/Zc:wchar_t','/Zc:forScope', '/DYY_NO_UNISTD_H']}
	}


# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
		'gcc'  : {'debug' : ''},
		'g++'  : {'debug' : ''},
		'c++'  : {'debug' : ''},
		'link' : {'debug' : '/debug'}
	}

def getCFLAGS(mode, warn, warnings_as_errors, CC):
	result = []
	if mode == 'release':
		# turn on optimization
		result.append(gCompilerOptions[CC]['optimization'])
	elif mode == 'debug':
		# turn on debug mode
		result.append(gCompilerOptions[CC]['debug'])
		result.append('-DVANAHEIMR_DEBUG')

	if warn:
		# turn on all warnings
		result.append(gCompilerOptions[CC]['warn_all'])

	if warnings_as_errors:
		# treat warnings as errors
		result.append(gCompilerOptions[CC]['warn_errors'])

	result.append(gCompilerOptions[CC]['standard'])

	return result

def getLibCXXPaths():
	"""Determines libc++ path

	returns (inc_path, lib_path)
	"""

	# determine defaults
	if os.name == 'posix':
		inc_path = '/usr/include'
		lib_path = '/usr/lib/libc++.so'
	else:
		raise ValueError, 'Error: unknown OS.  Where is libc++ installed?'

	# override with environement variables
	if 'LIBCXX_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['LIBCXX_INC_PATH'])
	if 'LIBCXX_LIB_PATH' in os.environ:
		lib_path = os.path.abspath(os.environ['LIBCXX_LIB_PATH'])

	return (inc_path, lib_path)

def getCXXFLAGS(mode, warn, warnings_as_errors, CXX):
	result = []
	if mode == 'release':
		# turn on optimization
		result.append(gCompilerOptions[CXX]['optimization'])
	elif mode == 'debug':
		# turn on debug mode
		result.append(gCompilerOptions[CXX]['debug'])
	# enable exception handling
	result.append(gCompilerOptions[CXX]['exception_handling'])

	if warn:
		# turn on all warnings
		result.append(gCompilerOptions[CXX]['warn_all'])

	if warnings_as_errors:
		# treat warnings as errors
		result.append(gCompilerOptions[CXX]['warn_errors'])

	result.append(gCompilerOptions[CXX]['standard'])

	return result

def getLINKFLAGS(mode, LINK):
	result = []
	if mode == 'debug':
		# turn on debug mode
		result.append(gLinkerOptions[LINK]['debug'])

	if LINK == 'c++':
		result.append(getLibCXXPaths()[1]);

	return result

def getExtraLibs():
	if os.name == 'nt':
		return []
	else:
		return []

def getVersion(base):
	try:
		svn_path = which('svn')
	except:
		print 'Failed to get subversion revision'
		return base + '.0'

	process = subprocess.Popen('svn info ..', shell=True,
		stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	(svn_info, std_err_data) = process.communicate()
	
	match = re.search('Revision: ', svn_info)
	revision = 'unknown'
	if match:
		end = re.search('\n', svn_info[match.end():])
		if end:
			revision = svn_info[match.end():match.end()+end.start()]
	else:
		print 'Failed to get SVN repository version!'

	return base + '.' + revision

def fixPath(path):
	if (os.name == 'nt'):
		return path.replace('\\', '\\\\')
	else:
		return path

import collections

def flatten(l):
	for el in l:
		if isinstance(el, collections.Iterable) and not isinstance(
			el, basestring):
			for sub in flatten(el):
				yield sub
		else:
			yield el
			
def defineConfigFlags(env):
	
	include_path = os.path.join(env['INSTALL_PATH'], "include")
	library_path = os.path.join(env['INSTALL_PATH'], "lib")
	bin_path     = os.path.join(env['INSTALL_PATH'], "bin")

	configFlags = env['CXXFLAGS'] + " ".join( 
		["%s\"\\\"%s\\\"\"" % x for x in (
			('-DVANAHEIMR_CXXFLAGS=', " ".join(flatten(env['CXXFLAGS']))),
			('-DPACKAGE=', 'vanaheimr'),
			('-DVERSION=', env['VERSION']),
			('-DVANAHEIMR_PREFIX_PATH=', fixPath(env['INSTALL_PATH'])),
			('-DVANAHEIMR_LDFLAGS=', fixPath(env['VANAHEIMR_LDFLAGS'])),
			('-L', fixPath(library_path)),
			('-DVANAHEIMR_INCLUDE_PATH=', fixPath(include_path)),
			('-DVANAHEIMR_LIB_PATH=', fixPath(library_path)),
			('-DVANAHEIMR_BIN_PATH=', fixPath(bin_path))	)])

	env.Replace(VANAHEIMR_CONFIG_FLAGS = configFlags)


def importEnvironment():
	env = {  }
	
	if 'PATH' in os.environ:
		env['PATH'] = os.environ['PATH']
	
	if 'CXX' in os.environ:
		env['CXX'] = os.environ['CXX']
	
	if 'CC' in os.environ:
		env['CC'] = os.environ['CC']
	
	if 'TMP' in os.environ:
		env['TMP'] = os.environ['TMP']
	
	if 'LD_LIBRARY_PATH' in os.environ:
		env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

	return env

def Environment():
	vars = Variables()

	# add a variable to handle RELEASE/DEBUG mode
	vars.Add(EnumVariable('mode', 'Release versus debug mode', 'debug',
		allowed_values = ('release', 'debug')))

	# add a variable to handle warnings
	vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 1))
	
	# shared or static libraries
	libraryDefault = 'shared'
	
	vars.Add(EnumVariable('library', 'Build shared or static library',
		libraryDefault, allowed_values = ('shared', 'static')))
	
	# add a variable to treat warnings as errors
	vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 1))

	# add a variable to determine the install path
	default_install_path = '/usr/local'
	
	if 'VANAHEIMR_INSTALL_PATH' in os.environ:
		default_install_path = os.environ['VANAHEIMR_INSTALL_PATH']
		
	vars.Add(PathVariable('install_path', 'The vanaheimr install path',
		default_install_path))

	vars.Add(BoolVariable('install', 'Include vanaheimr install path in default '
		'targets that will be built and configure to install in the '
		'install_path (defaults to false unless one of the targets is '
		'"install")', 0))
	
	# create an Environment
	env = OldEnvironment(ENV = importEnvironment(), \
		tools = getTools(), variables = vars)

	# set the version
	env.Replace(VERSION = getVersion("0.1"))
	
	# always link with the c++ compiler
	if os.name != 'nt':
		env['LINK'] = env['CXX']
	
	# get C compiler switches
	env.AppendUnique(CFLAGS = getCFLAGS(env['mode'], env['Wall'], \
		env['Werror'], env.subst('$CC')))

	# get CXX compiler switches
	env.AppendUnique(CXXFLAGS = getCXXFLAGS(env['mode'], env['Wall'], \
		env['Werror'], env.subst('$CXX')))

	# get linker switches
	env.AppendUnique(LINKFLAGS = getLINKFLAGS(env['mode'], env.subst('$LINK')))

	# Install paths
	if env['install']:
		env.Replace(INSTALL_PATH = os.path.abspath(env['install_path']))
	else:
		env.Replace(INSTALL_PATH = os.path.abspath('.'))

	# get libc++
	if env['CXX'] == 'c++':
		env.AppendUnique(CPPPATH = getLibCXXPaths()[0])
	
	# set extra libs 
	env.Replace(EXTRA_LIBS=getExtraLibs())

	# get ocelot paths
	if haveOcelot():
		(ocelot_exe_path,ocelot_lib_path,ocelot_inc_path,ocelot_cflags,\
			ocelot_lflags,ocelot_libs) = getOcelotPaths()
		env.AppendUnique(LIBPATH = ocelot_lib_path)
		env.AppendUnique(CPPPATH = ocelot_inc_path)
		env.AppendUnique(CXXFLAGS = ocelot_cflags)
		env.AppendUnique(LINKFLAGS = ocelot_lflags)
		env.AppendUnique(EXTRA_LIBS = ocelot_libs)
		env.Replace(HAVE_OCELOT=True)
	else:
		env.Replace(HAVE_OCELOT=False)

	# set vanaheimr libs
	vanaheimr_libs = '-lvanaheimr'
	
	for lib in env['EXTRA_LIBS']:
		vanaheimr_libs += ' ' + lib
	
	env.Replace(VANAHEIMR_LDFLAGS=vanaheimr_libs)
	
	# generate vanaheimr-config flags
	defineConfigFlags(env)

	# set the build path
	env.Replace(BUILD_ROOT = str(env.Dir('.')))

	# set vanaheimr include path
	env.AppendUnique(LIBPATH = os.path.abspath('.'))
	
	# we need librt on linux
	if os.name == 'posix':
		env.AppendUnique(EXTRA_LIBS = ['-lrt']) 

	# generate help text
	Help(vars.GenerateHelpText(env))

	return env

